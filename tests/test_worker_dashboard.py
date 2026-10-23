"""Real failure/recovery behaviour and honest worker dashboard observations."""

import asyncio
from contextlib import contextmanager
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.coordinator.main import create_app as create_gateway
from app.coordinator.registry import WorkerRegistry
from app.core.config import GatewaySettings, WorkerSettings
from app.core.lamport import LamportClock, TIME_HEADER, clocked_request
from app.schemas.workers import WorkerRegistration
from app.worker.main import create_app as create_worker
from test_gateway import FEATURES, gateway_client, prediction
from test_lamport_cluster import assert_causal_trace, cluster

TOKEN = "worker-demo-test-" + "x" * 32
REGISTRATION = dict(worker_id="test-worker", worker_url="http://worker.test",
                    instance_id="instance-one", demo_control_available=True)
HOUSE = dict(area_sqft=1800, bedrooms=3, full_bathrooms=2, garage_spaces=2, year_built=2000, overall_quality=7)


@contextmanager
def controlled_gateway(handler, remote="127.0.0.1", enabled=True, registration=None):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_gateway(GatewaySettings(demo_control_token=TOKEN if enabled else None), client=upstream)
    with TestClient(app, client=(remote, 50000)) as client:
        try:
            if registration is not None:
                client.post("/workers/register", json=registration)
            yield client, app
        finally:
            client.portal.call(upstream.aclose)


def acknowledgement(request):
    assert request.headers["X-Demo-Token"] == TOKEN
    assert request.url.path == "/demo/control"
    payload = json.loads(request.content)
    return httpx.Response(200, json=dict(worker_id="test-worker", instance_id=payload["instance_id"],
                                       paused=payload["action"] == "pause"))


def test_worker_page_html_json_and_static_assets_remain_available():
    with gateway_client(lambda request: pytest.fail("The dashboard must not probe workers")) as (client, _, _):
        assert isinstance(client.get("/workers").json(), list)
        html = client.get("/workers", headers={"Accept": "text/html"})
        assert html.status_code == 200 and "See each worker" in html.text
        assert html.headers["Vary"] == "Accept"
        assert "demo_control_token" not in html.text
        assert client.get("/workers/view").status_code == 200
        assert isinstance(client.get("/workers?format=json", headers={"Accept": "text/html"}).json(), list)
        assert client.get("/static/workers.js").status_code == 200
        assert client.get("/workers/status").json()["demo_controls_enabled"] is False


def test_dashboard_and_demo_control_cannot_renew_heartbeat_or_skip_detection():
    with controlled_gateway(acknowledgement) as (client, app):
        elapsed = [0.]
        app.state.registry = WorkerRegistry(clock=lambda: elapsed[0])
        client.post("/workers/register", json=REGISTRATION)
        worker = app.state.registry.list_workers()[0]
        elapsed[0] = 3
        response = client.post("/workers/test-worker/demo", json={"action": "pause", "instance_id": "instance-one"})
        assert response.status_code == 200
        state = client.get("/workers/status").json()
        assert state["nodes"][0]["worker"]["demo_paused"] is True
        assert state["nodes"][0]["lease_remaining_seconds"] == 5
        assert worker.available is None and not worker.heartbeat_expired
        assert app.state.registry.select() is worker  # The demo endpoint did not edit scheduling.
        elapsed[0] = 8
        state = client.get("/workers/status").json()
        assert state["nodes"][0]["worker"]["heartbeat_expired"] is True
        assert state["nodes"][0]["worker"]["available"] is False
        assert state["nodes"][0]["lease_remaining_seconds"] == 0
        assert TOKEN not in json.dumps(state)
        assert client.post("/workers/test-worker/demo", json={"action": "resume", "instance_id": "instance-one"}).status_code == 200
        assert worker.available is False and worker.heartbeat_expired  # Await a genuine heartbeat.


@pytest.mark.parametrize("remote,origin", [("192.168.1.40", None), ("127.0.0.1", "http://unrelated.test")])
def test_demo_gateway_rejects_remote_or_cross_origin_control(remote, origin):
    with controlled_gateway(lambda request: pytest.fail("Forbidden control was forwarded"), remote=remote,
                            registration=REGISTRATION) as (client, _):
        response = client.post("/workers/test-worker/demo", json={"action": "pause", "instance_id": "instance-one"},
                               headers={"Origin": origin} if origin else {})
        assert response.status_code == 403


@pytest.mark.parametrize("case,status", [("disabled", 404), ("unknown", 404), ("stale", 409), ("unsupported", 409)])
def test_control_rejects_disabled_unknown_stale_or_unsupported_workers(case, status):
    registration = {**REGISTRATION, "demo_control_available": case != "unsupported"}
    with controlled_gateway(lambda request: pytest.fail("Invalid control was forwarded"), enabled=case != "disabled",
                            registration=registration) as (client, _):
        worker = "unknown" if case == "unknown" else "test-worker"
        instance = "old-process" if case == "stale" else "instance-one"
        assert client.post(f"/workers/{worker}/demo", json={"action": "pause", "instance_id": instance}).status_code == status


@pytest.mark.parametrize("payload", [{}, {"action": "terminate", "instance_id": "instance-one"},
    {"action": "pause", "instance_id": "instance-one", "extra": True}, {"action": True, "instance_id": "instance-one"}])
def test_demo_control_input_is_explicit_and_bounded(payload):
    with controlled_gateway(lambda request: pytest.fail("Invalid action was forwarded"), registration=REGISTRATION) as (client, _):
        assert client.post("/workers/test-worker/demo", json=payload).status_code == 422


@pytest.mark.parametrize("changes", [{"worker_id": "wrong"}, {"instance_id": "wrong"}, {"paused": False}, {"paused": "true"}])
def test_invalid_control_acknowledgement_does_not_claim_a_successful_pause(changes):
    def handler(request):
        return httpx.Response(200, json={"worker_id": "test-worker", "instance_id": "instance-one", "paused": True, **changes})
    with controlled_gateway(handler, registration=REGISTRATION) as (client, app):
        assert client.post("/workers/test-worker/demo", json={"action": "pause", "instance_id": "instance-one"}).status_code == 502
        assert app.state.registry.list_workers()[0].demo_paused is False


@pytest.mark.parametrize("exception,status", [(httpx.ConnectError, 503), (httpx.ReadTimeout, 504), (httpx.RemoteProtocolError, 502)])
def test_demo_control_reports_transport_failure_without_editing_eligibility(exception, status):
    def handler(request):
        raise exception("simulated", request=request)
    with controlled_gateway(handler, registration=REGISTRATION) as (client, app):
        assert client.post("/workers/test-worker/demo", json={"action": "pause", "instance_id": "instance-one"}).status_code == status
        worker = app.state.registry.list_workers()[0]
        assert worker.available is None and worker.demo_paused is False


def test_prediction_counts_and_bounded_history_only_record_validated_successes():
    with gateway_client(lambda request: httpx.Response(200, json=prediction(request.headers["X-Request-ID"]))) as (client, app, _):
        for index in range(25):
            assert client.post("/predict", json={"features": FEATURES}, headers={"X-Request-ID": f"request-{index}"}).status_code == 200
        state = client.get("/workers/status").json()
        assert state["nodes"][0]["worker"]["completed_predictions"] == 25
        assert len(state["recent_predictions"]) == 20
        assert state["recent_predictions"][0]["request_id"] == "request-24"
        assert state["recent_predictions"][-1]["request_id"] == "request-5"
        assert "features" not in json.dumps(state)
        worker = app.state.registry.list_workers()[0]
        app.state.registry.register(WorkerRegistration(worker_id=worker.worker_id, worker_url=worker.worker_url))
        app.state.registry.record_prediction(worker, worker.model, "same-process-in-flight", "same-process-trace")
        assert app.state.registry.list_workers()[0].completed_predictions == 26
        # A new process resets its count; an old in-flight reply cannot increment it.
        app.state.registry.register(WorkerRegistration(worker_id=worker.worker_id, worker_url=worker.worker_url, instance_id="new-process"))
        app.state.registry.record_prediction(worker, worker.model, "old-request", "old-trace")
        assert app.state.registry.list_workers()[0].completed_predictions == 0
    with gateway_client(lambda request: httpx.Response(200, json={})) as (client, _, _):
        assert client.post("/predict", json={"features": FEATURES}).status_code == 502
        state = client.get("/workers/status").json()
        assert state["nodes"][0]["worker"]["completed_predictions"] == 0
        assert state["recent_predictions"] == []


@pytest.mark.parametrize("enabled,supplied,status", [(False, TOKEN, 404), (True, "wrong", 403), (True, "", 403)])
def test_worker_control_is_opt_in_and_requires_launch_secret(enabled, supplied, status, tmp_path):
    app = create_worker(WorkerSettings(gateway_url=None, demo_model_dir=tmp_path, demo_control_token=TOKEN if enabled else None))
    with TestClient(app) as client:
        response = client.post("/demo/control", json={"action": "pause", "instance_id": app.state.lamport.instance_id},
                               headers={"X-Demo-Token": supplied})
        assert response.status_code == status
        assert client.get("/health").status_code == 200
        assert TOKEN not in repr(app.state.settings)
        assert "demo_control_token" not in app.state.settings.model_dump()


@pytest.mark.asyncio
async def test_demo_headers_are_forwarded_without_overriding_logical_clock_identity():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        clock = LamportClock("gateway")
        await clocked_request(client, clock, "POST", "http://worker.test/demo/control", request_id="real-request", peer_id="worker",
                              headers={"X-Demo-Token": TOKEN, "X-Request-ID": "wrong", TIME_HEADER: "999"})
    assert seen[0].headers["X-Demo-Token"] == TOKEN
    assert seen[0].headers["X-Request-ID"] == "real-request"
    assert seen[0].headers[TIME_HEADER] == "1"


@pytest.mark.asyncio
async def test_real_worker_pause_expiry_survivor_routing_and_resume_without_fake_gateway_updates():
    async with cluster(interval=.02, demo_token=TOKEN, heartbeat_timeout=.2) as (client, gateway, workers):
        async def wait_for(predicate):
            deadline = asyncio.get_running_loop().time() + 3
            while asyncio.get_running_loop().time() < deadline:
                state = (await client.get("/workers/status")).json()
                if predicate(state):
                    return state
                await asyncio.sleep(.01)
            pytest.fail("Worker state did not recover/expire")
        await wait_for(lambda state: all(node["worker"]["heartbeat_sequence"] >= 2 for node in state["nodes"]))
        first = workers[0]
        instance = first.state.lamport.instance_id
        responses = [await client.post("/predict", json={"features": FEATURES}) for _ in range(6)]
        assert [r.json()["worker_id"] for r in responses] == ["worker-1", "worker-2"] * 3
        pause = await client.post("/workers/worker-1/demo", json={"action": "pause", "instance_id": instance})
        assert pause.status_code == 200 and pause.json()["paused"] is True
        assert first.state.demo_paused.is_set() and first.state.ready is True  # Process remains running.
        assert first.state.heartbeat_task is not None and not first.state.heartbeat_task.done()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=first), base_url="http://worker-1.test") as direct:
            assert (await direct.get("/health")).status_code == 503
            for path, payload in (("/predict", {"features": FEATURES}), ("/predict/house", HOUSE),
                                  ("/predict/face", {"image_base64": "YWJj"})):
                assert (await direct.post(path, json=payload)).status_code == 503
            assert (await direct.post("/demo/control", json={"action": "resume", "instance_id": "old-process"},
                                       headers={"X-Demo-Token": TOKEN})).status_code == 409
        await asyncio.sleep(.05)  # Allow an already-sent heartbeat to finish.
        sequence = gateway.state.registry.list_workers()[0].heartbeat_sequence
        state = await wait_for(lambda state: state["nodes"][0]["worker"]["heartbeat_expired"])
        assert state["nodes"][0]["worker"]["heartbeat_sequence"] == sequence
        assert state["nodes"][0]["worker"]["available"] is False
        assert state["nodes"][1]["worker"]["available"] is True
        for _ in range(6):
            response = await client.post("/predict", json={"features": FEATURES})
            assert response.status_code == 200 and response.json()["worker_id"] == "worker-2"
            assert_causal_trace((await client.get("/traces/" + response.headers["X-Trace-ID"])).json(), response)
        assert (await client.post("/workers/worker-1/demo", json={"action": "resume", "instance_id": instance})).status_code == 200
        state = await wait_for(lambda state: state["nodes"][0]["worker"]["available"] is True)
        assert state["nodes"][0]["worker"]["heartbeat_sequence"] > sequence
        assert state["nodes"][0]["worker"]["instance_id"] == instance
        responses = [await client.post("/predict", json={"features": FEATURES}) for _ in range(6)]
        assert [r.json()["worker_id"] for r in responses] in (["worker-1", "worker-2"] * 3, ["worker-2", "worker-1"] * 3)
        # Pausing both leaves no eligible worker after actual heartbeat expiry.
        for worker in workers:
            await client.post(f"/workers/{worker.state.settings.worker_id}/demo", json={"action": "pause", "instance_id": worker.state.lamport.instance_id})
        await wait_for(lambda state: all(node["worker"]["heartbeat_expired"] for node in state["nodes"]))
        assert (await client.post("/predict", json={"features": FEATURES})).status_code == 503
    assert all(worker.state.heartbeat_task is None for worker in workers)
