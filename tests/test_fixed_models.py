"""Fixed-model contracts, real ONNX replicas, compatible scheduling and traces."""

import asyncio
import base64
from io import BytesIO
import json

import httpx
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, WorkerSettings
from app.coordinator.registry import RegistrationConflict, WorkerRegistry
from app.schemas.inference import MODEL_NAME
from app.schemas.models import FACE_MODEL, HOUSE_MODEL, MAX_IMAGE_BYTES
from app.schemas.workers import WorkerRegistration
from app.worker.fixed_models import face_tensor
from app.worker.main import create_app
from app.coordinator.main import create_app as create_gateway
from app.core.config import GatewaySettings
from test_gateway import gateway_client
from test_lamport_cluster import assert_causal_trace, cluster

HOUSE = dict(area_sqft=1800, bedrooms=3, full_bathrooms=2, garage_spaces=2,
             year_built=2000, overall_quality=7)
MODELS = [MODEL_NAME, HOUSE_MODEL, FACE_MODEL]
MODEL_DIR = PROJECT_ROOT / "models" / "demo"


def encoded_image(size=(64, 64), format="PNG", **kwargs):
    stream = BytesIO()
    Image.new("RGB", size, "white").save(stream, format=format, **kwargs)
    return {"image_base64": base64.b64encode(stream.getvalue()).decode()}


def require_artifacts():
    if not (MODEL_DIR / "house_price/house_price.onnx").is_file() or not (MODEL_DIR / "male_female/male_female.onnx").is_file():
        pytest.skip("Prepare the two local demo models for real-artifact integration tests")


@pytest.mark.parametrize("key,value", [
    ("area_sqft", 0), ("area_sqft", 5643), ("area_sqft", True), ("area_sqft", "1800"),
    ("area_sqft", float("nan")), ("area_sqft", float("inf")),
    ("bedrooms", -1), ("bedrooms", 9), ("bedrooms", 3.5), ("bedrooms", True),
    ("full_bathrooms", 5), ("garage_spaces", 5), ("year_built", 2026),
    ("year_built", 1800), ("overall_quality", 0), ("overall_quality", 11), ("extra", 1),
])
def test_invalid_house_never_contacts_a_worker(key, value):
    with gateway_client(lambda request: pytest.fail("Invalid input was forwarded")) as (client, _, _):
        response = client.post("/predict/house", content=json.dumps({**HOUSE, key: value}),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.headers["X-Trace-ID"]


@pytest.mark.parametrize("payload", [
    {}, {"image_base64": ""}, {"image_base64": "%%%"}, {"image_base64": "é"},
    {"image_base64": "data:image/png;base64,AAAA"}, {"image_base64": "YWJj"},
    {"image_base64": True}, encoded_image((16, 16)), encoded_image(format="GIF"),
    {**encoded_image(), "filename": "extra.png"},
])
def test_invalid_face_never_contacts_a_worker(payload):
    with gateway_client(lambda request: pytest.fail("Invalid image was forwarded")) as (client, _, _):
        response = client.post("/predict/face", json=payload)
        assert response.status_code == 422
        trace = client.get("/traces/" + response.headers["X-Trace-ID"]).json()
        assert trace["complete"] is False and len(trace["events"]) == 2


def test_face_limits_reject_decoded_oversize_and_streamed_body():
    with gateway_client(lambda request: pytest.fail("Oversize image was forwarded")) as (client, _, _):
        response = client.post("/predict/face", json={
            "image_base64": base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode(),
        })
        assert response.status_code == 422
        response = client.post("/predict/face", content=b"x" * (7 * 1024 * 1024 + 1))
        assert response.status_code == 413
        assert response.headers["X-Trace-ID"]


def test_animated_and_excessive_dimension_images_are_rejected():
    stream = BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (32, 32), "black")], duration=100)
    with pytest.raises(ValueError, match="still"):
        face_tensor(stream.getvalue())
    # A large image is small on disk; the dimension check precedes pixel decoding.
    with pytest.raises(ValueError, match="20 megapixels"):
        face_tensor(base64.b64decode(encoded_image((4500, 4500))["image_base64"]))


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_serving_image_preprocessing_matches_prepared_model_contract(tmp_path, format):
    import numpy as np
    from scripts.demo_models import face_tensor as prepared_tensor
    path = tmp_path / ("face.png" if format == "PNG" else "face.jpg")
    image = Image.new("RGB", (64, 96), "white")
    image.paste("black", (0, 0, 32, 96))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, format=format, exif=exif)
    np.testing.assert_array_equal(face_tensor(path.read_bytes()), prepared_tensor(path))


@pytest.mark.asyncio
async def test_chunked_image_request_replay_and_size_limit():
    app = create_gateway(GatewaySettings())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
            body = json.dumps(encoded_image()).encode()
            async def chunks():
                for index in range(0, len(body), 17):
                    yield body[index:index + 17]
            response = await asyncio.wait_for(client.post("/predict/face", content=chunks(),
                headers={"Content-Type": "application/json"}), timeout=2)
            # Successful decode/validation reaches the empty-registry check.
            assert response.status_code == 503
            yielded = 0
            async def large_chunks():
                nonlocal yielded
                for _ in range(20):
                    yielded += 1
                    yield b"x" * (2 * 1024 * 1024)
            response = await asyncio.wait_for(client.post("/predict/face", content=large_chunks()), timeout=2)
            assert response.status_code == 413 and yielded == 4
            assert (await client.get("/models")).status_code == 200


@pytest.mark.parametrize("path,payload", [("/predict/house", {**HOUSE, "bedrooms": True}),
                                         ("/predict/face", {"image_base64": "YWJj"})])
def test_worker_also_validates_new_model_inputs(worker_client, path, payload):
    assert worker_client.post(path, json=payload).status_code == 422


def test_missing_or_corrupt_artifacts_keep_iris_available(tmp_path):
    (tmp_path / "house_price").mkdir()
    (tmp_path / "house_price/house_price.onnx").write_bytes(b"broken model")
    app = create_app(WorkerSettings(gateway_url=None, demo_model_dir=tmp_path))
    with TestClient(app) as client:
        assert app.state.demo_models == {}
        assert client.get("/health").json()["ready"] is True
        assert client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}).status_code == 200
        assert client.post("/predict/house", json=HOUSE).status_code == 503
        assert client.post("/predict/face", json=encoded_image()).status_code == 503
    assert app.state.demo_models == {}


def test_compatible_scheduling_and_cursors_are_per_model():
    registry = WorkerRegistry()
    for worker_id, models in (("iris-only", [MODEL_NAME]), ("all-a", MODELS), ("all-b", MODELS)):
        registry.register(WorkerRegistration(worker_id=worker_id, worker_url=f"http://{worker_id}.test",
                                             models=models, instance_id="process-1"))
    assert [registry.select(HOUSE_MODEL).worker_id, registry.select(FACE_MODEL).worker_id,
            registry.select(HOUSE_MODEL).worker_id, registry.select(FACE_MODEL).worker_id] == [
                "all-a", "all-a", "all-b", "all-b"]
    assert registry.select().worker_id == "iris-only"
    with pytest.raises(RegistrationConflict):
        registry.register(WorkerRegistration(worker_id="all-a", worker_url="http://all-a.test",
                                             models=[MODEL_NAME], instance_id="process-1"))
    replacement, created = registry.register(WorkerRegistration(worker_id="all-a", worker_url="http://all-a.test",
                                                                models=[MODEL_NAME], instance_id="process-2"))
    assert not created and replacement.models == [MODEL_NAME]
    assert {registry.select(HOUSE_MODEL).worker_id for _ in range(4)} == {"all-b"}


@pytest.mark.parametrize("models", [[MODEL_NAME, "unknown"], [], [HOUSE_MODEL], [MODEL_NAME, MODEL_NAME]])
def test_registration_rejects_invalid_capabilities(models):
    with gateway_client(lambda request: pytest.fail("Registration contacted a worker")) as (client, _, _):
        assert client.post("/workers/register", json={"worker_id": "invalid", "worker_url": "http://invalid.test",
                                                      "models": models}).status_code == 422


def test_legacy_iris_registration_does_not_receive_other_models():
    with gateway_client(lambda request: pytest.fail("Incompatible worker contacted")) as (client, _, _):
        assert [item["available_workers"] for item in client.get("/models").json()] == [1, 0, 0]
        assert client.post("/predict/house", json=HOUSE).status_code == 503
        assert client.post("/predict/face", json=encoded_image()).status_code == 503


@pytest.mark.parametrize("path,payload,result", [
    ("/predict/house", HOUSE, {"predicted_price_usd": "bad"}),
    ("/predict/house", HOUSE, {"predicted_price_usd": float("inf")}),
    ("/predict/house", HOUSE, {"predicted_price_usd": 123., "currency": "INR"}),
    ("/predict/house", HOUSE, {"predicted_price_usd": 123., "model": FACE_MODEL}),
    ("/predict/house", HOUSE, {"predicted_price_usd": 123., "worker_id": "wrong"}),
    ("/predict/house", HOUSE, {"predicted_price_usd": 123., "request_id": "wrong"}),
    ("/predict/face", encoded_image(), {"prediction": "male", "probabilities": {"male": .8}}),
    ("/predict/face", encoded_image(), {"prediction": "male", "probabilities": {"female": .8, "male": .2}}),
    ("/predict/face", encoded_image(), {"prediction": "male", "probabilities": {"female": .5, "male": .8}}),
])
def test_gateway_checks_new_model_response_contracts(path, payload, result):
    def handler(request):
        body = {"model": HOUSE_MODEL if path.endswith("house") else FACE_MODEL,
                "worker_id": "test-worker", "request_id": request.headers["X-Request-ID"], **result}
        # Deliberately exercise invalid nonfinite JSON too.
        return httpx.Response(200, content=json.dumps(body).encode())
    with gateway_client(handler, registered=False) as (client, _, _):
        client.post("/workers/register", json={"worker_id": "test-worker", "worker_url": "http://worker.test",
                                               "models": MODELS})
        assert client.post(path, json=payload).status_code == 502


@pytest.mark.asyncio
async def test_all_three_real_models_round_robin_concurrency_and_traces():
    require_artifacts()
    samples = {label: {"image_base64": base64.b64encode(
        (MODEL_DIR / "male_female/samples" / f"sample_{label}.png").read_bytes()).decode()}
        for label in ("female", "male")}
    async with cluster() as (client, gateway, workers):
        assert [item["available_workers"] for item in (await client.get("/models")).json()] == [2, 2, 2]
        sessions = [dict(worker.state.demo_models) for worker in workers]
        for path, payload, expected in (
            ("/predict", {"features": [5.1, 3.5, 1.4, 0.2]}, "setosa"),
            ("/predict/house", HOUSE, None),
            ("/predict/face", samples["female"], "female"),
        ):
            responses = [await client.post(path, json=payload) for _ in range(4)]
            assert [r.json()["worker_id"] for r in responses] == ["worker-1", "worker-2"] * 2
            for response in responses:
                assert response.status_code == 200
                if expected:
                    assert response.json()["prediction"] == expected
                else:
                    assert response.json()["predicted_price_usd"] == pytest.approx(232247.41, abs=.1)
                trace = (await client.get("/traces/" + response.headers["X-Trace-ID"])).json()
                assert_causal_trace(trace, response)
        assert (await client.post("/predict/face", json=samples["male"])).json()["prediction"] == "male"
        # Mixed concurrent requests retain model identity and distinct causal traces.
        jobs = [("/predict/house", HOUSE), ("/predict/face", samples["female"])] * 8
        responses = await asyncio.wait_for(asyncio.gather(*(
            client.post(path, json=payload, headers={"X-Request-ID": "reused"}) for path, payload in jobs
        )), timeout=10)
        assert all(r.status_code == 200 for r in responses)
        assert len({r.headers["X-Trace-ID"] for r in responses}) == 16
        for response, (path, _) in zip(responses, jobs):
            assert response.json()["model"] == (HOUSE_MODEL if path.endswith("house") else FACE_MODEL)
            assert_causal_trace((await client.get("/traces/" + response.headers["X-Trace-ID"])).json(), response)
        assert all(worker.state.demo_models == original for worker, original in zip(workers, sessions))
        gateway.state.registry.set_available(gateway.state.registry.list_workers()[0], False)
        assert (await client.post("/predict/house", json=HOUSE)).json()["worker_id"] == "worker-2"
        assert (await client.post("/predict/face", json=samples["female"])).json()["worker_id"] == "worker-2"
    assert all(worker.state.demo_models == {} for worker in workers)
