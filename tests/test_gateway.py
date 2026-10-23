import asyncio
import json
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient

from app.coordinator.main import create_app
from app.core.config import GatewaySettings, WorkerSettings
from app.worker.main import create_app as create_worker

from test_worker import INVALID_PAYLOADS


FEATURES = [5.1, 3.5, 1.4, 0.2]
WORKER_HEALTH = {
    "status": "ok", "service": "worker", "ready": True,
    "worker_id": "test-worker", "model": "iris-logistic-regression-v1",
}


def prediction(request_id="gateway-test"):
    return {
        "prediction": "setosa", "class_id": 0,
        "probabilities": {"setosa": 0.98, "versicolor": 0.01, "virginica": 0.01},
        "model": "iris-logistic-regression-v1", "worker_id": "test-worker",
        "request_id": request_id,
    }


@contextmanager
def gateway_client(handler, registered=True):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    application = create_app(
        GatewaySettings(request_timeout_seconds=0.25),
        client=upstream,
    )
    with TestClient(application) as client:
        try:
            if registered:
                assert client.post("/workers/register", json={
                    "worker_id": "test-worker", "worker_url": "http://worker.test:8001",
                    "model": "iris-logistic-regression-v1",
                }).status_code == 201
            yield client, application, upstream
        finally:
            client.portal.call(upstream.aclose)


def test_gateway_forwards_features_request_id_and_configured_timeout():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json=WORKER_HEALTH)
        assert request.method == "POST"
        assert request.url == "http://worker.test:8001/predict"
        assert json.loads(request.content) == {"features": FEATURES}
        assert request.headers["x-request-id"] == "gateway-test"
        assert all(value == 0.25 for value in request.extensions["timeout"].values())
        return httpx.Response(200, json=prediction())

    with gateway_client(handler) as (client, application, upstream):
        response = client.post(
            "/predict", json={"features": FEATURES}, headers={"X-Request-ID": "gateway-test"}
        )
        assert response.status_code == 200
        assert response.json() == prediction()
        assert response.headers["x-request-id"] == "gateway-test"
        assert not upstream.is_closed
    assert any(request.url.path == "/predict" for request in requests)


@pytest.mark.asyncio
async def test_gateway_calls_real_worker_in_process():
    worker = create_worker(WorkerSettings(worker_id="real-cpu-worker", gateway_url=None))
    gateway_settings = GatewaySettings()
    async with worker.router.lifespan_context(worker):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=worker)) as upstream:
            gateway = create_app(gateway_settings, client=upstream)
            async with gateway.router.lifespan_context(gateway):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=gateway), base_url="http://gateway.test"
                ) as client:
                    registered = await client.post("/workers/register", json={
                        "worker_id": "real-cpu-worker", "worker_url": "http://worker.test",
                        "model": "iris-logistic-regression-v1",
                    })
                    assert registered.status_code == 201
                    health = await client.get("/health")
                    assert health.status_code == 200
                    assert health.json()["worker_available"] is True
                    response = await client.post(
                        "/predict", json={"features": FEATURES},
                        headers={"X-Request-ID": "end-to-end-trace"},
                    )
                    assert response.status_code == 200
                    assert response.json()["prediction"] == "setosa"
                    assert response.json()["worker_id"] == "real-cpu-worker"
                    assert response.json()["request_id"] == "end-to-end-trace"
                    assert response.headers["x-request-id"] == "end-to-end-trace"
            # An injected client belongs to its caller, even after gateway shutdown.
            assert not upstream.is_closed


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_gateway_validates_before_contacting_worker(payload):
    def handler(request):
        pytest.fail("Invalid input must not be forwarded to a worker")

    with gateway_client(handler) as (client, _, _):
        response = client.post(
            "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]


@pytest.mark.parametrize(
    ("error_class", "status"),
    [(httpx.ConnectError, 503), (httpx.ReadTimeout, 504), (httpx.ConnectTimeout, 504),
     (httpx.RemoteProtocolError, 502)],
)
def test_gateway_maps_transport_failures(error_class, status):
    calls = []

    def handler(request):
        calls.append(request)
        raise error_class("simulated worker failure", request=request)

    with gateway_client(handler) as (client, _, _):
        response = client.post("/predict", json={"features": FEATURES})
        assert response.status_code == status
        assert response.json()["detail"]
        assert response.headers["x-request-id"]
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["worker_available"] is False
    # Bounded work: a failed prediction and one health probe, no retry storm.
    assert len(calls) <= 3


@pytest.mark.parametrize("upstream_status", [503, 500, 422, 404])
def test_gateway_maps_upstream_http_errors(upstream_status):
    def handler(request):
        return httpx.Response(upstream_status, json={"detail": "internal upstream details"})

    with gateway_client(handler) as (client, _, _):
        response = client.post("/predict", json={"features": FEATURES})
        assert response.status_code == (503 if upstream_status == 503 else 502)
        assert "internal upstream details" not in response.text


INVALID_RESPONSES = [
    {}, {**prediction(), "prediction": "unknown"},
    {**prediction(), "class_id": 3},
    {**prediction(), "class_id": 1},
    {**prediction(), "probabilities": {"setosa": 0.98}},
    {**prediction(), "probabilities": {"setosa": 2, "versicolor": 0, "virginica": 0}},
    {**prediction(), "probabilities": {"setosa": 0.5, "versicolor": 0.1, "virginica": 0.1}},
    {**prediction(), "worker_id": ""},
    {**prediction(), "request_id": "wrong-trace"},
    {**prediction(), "model": "unknown-model"},
    {**prediction(), "unexpected": "secret"},
]


@pytest.mark.parametrize("upstream_payload", INVALID_RESPONSES)
def test_gateway_rejects_invalid_upstream_responses(upstream_payload):
    def handler(request):
        return httpx.Response(200, json=upstream_payload)

    with gateway_client(handler) as (client, _, _):
        response = client.post(
            "/predict", json={"features": FEATURES}, headers={"X-Request-ID": "gateway-test"}
        )
        assert response.status_code == 502
        assert response.json()["detail"]


def test_gateway_rejects_non_json_upstream_response():
    with gateway_client(lambda request: httpx.Response(200, text="worker returned HTML")) as (
        client, _, _
    ):
        assert client.post("/predict", json={"features": FEATURES}).status_code == 502


@pytest.mark.parametrize("health_payload", [
    WORKER_HEALTH, {**WORKER_HEALTH, "ready": False}, {"ready": False}, {}, {"ready": True}
])
def test_gateway_health_probes_worker(health_payload):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=health_payload)

    with gateway_client(handler) as (client, _, _):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["service"] == "gateway"
        assert body["ready"] is True
        assert body["worker_available"] is (health_payload == WORKER_HEALTH)
        assert body["status"] == ("ok" if health_payload == WORKER_HEALTH else "degraded")
    assert requests and all(request.url.path == "/health" for request in requests)


def test_gateway_owned_client_is_reused_and_closed():
    application = create_app(GatewaySettings())
    with TestClient(application) as client:
        upstream = application.state.client
        assert not upstream.is_closed
        assert application.state.client is upstream
    assert upstream.is_closed


def test_gateway_rejects_mismatched_upstream_header():
    def handler(request):
        return httpx.Response(200, json=prediction(), headers={"X-Request-ID": "wrong-trace"})

    with gateway_client(handler) as (client, _, _):
        response = client.post(
            "/predict", json={"features": FEATURES}, headers={"X-Request-ID": "gateway-test"}
        )
        assert response.status_code == 502
        assert response.headers["x-request-id"] == "gateway-test"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/predict", "/health"])
async def test_gateway_enforces_total_deadline_for_delayed_worker(path):
    cancelled = False
    calls = 0

    async def delayed_worker(request):
        nonlocal cancelled, calls
        calls += 1
        try:
            # MockTransport ignores HTTPX's per-I/O timeout. Only the gateway's
            # total deadline can cancel this wait before a valid reply arrives.
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            cancelled = True
            raise
        payload = WORKER_HEALTH if path == "/health" else prediction("deadline-test")
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(delayed_worker)) as upstream:
        application = create_app(
            GatewaySettings(request_timeout_seconds=0.05),
            client=upstream,
        )
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://gateway.test"
            ) as client:
                assert (await client.post("/workers/register", json={
                    "worker_id": "test-worker", "worker_url": "http://worker.test",
                    "model": "iris-logistic-regression-v1",
                })).status_code == 201
                request = (
                    client.post(path, json={"features": FEATURES}, headers={"X-Request-ID": "deadline-test"})
                    if path == "/predict" else client.get(path)
                )
                # A generous outer bound prevents a broken deadline hanging the suite.
                response = await asyncio.wait_for(request, timeout=1.0)
                if path == "/predict":
                    assert response.status_code == 504
                    assert response.headers["x-request-id"] == "deadline-test"
                else:
                    assert response.status_code == 200
                    assert response.json()["status"] == "degraded"
                    assert response.json()["worker_available"] is False
                assert cancelled
                assert calls == 1
