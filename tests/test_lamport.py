"""Clock rules, process identities, HTTP boundaries, and bounded trace storage."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import WorkerSettings
from app.core.lamport import (
    INSTANCE_HEADER, RECEIVED_HEADER, TIME_HEADER, TRACE_HEADER,
    ClockEvent, InferenceTrace, LamportClock, TraceStore, clocked_request, parse_time,
)
from app.worker.main import create_app
from test_gateway import FEATURES, gateway_client, prediction


def test_lamport_clock_merges_remote_values_without_going_backwards():
    clock = LamportClock("test")
    assert clock.value == 0
    assert clock.tick() == 1
    assert clock.receive(10) == 11
    assert clock.receive(3) == 12
    assert clock.tick() == 13
    assert clock.receive(13) == 14


@pytest.mark.parametrize("remote", [-1, True, 1.5, "2", None])
def test_invalid_receive_does_not_modify_clock(remote):
    clock = LamportClock("test")
    clock.tick()
    with pytest.raises(ValueError):
        clock.receive(remote)
    assert clock.value == 1


def test_clock_updates_are_atomic_across_worker_threads():
    clock = LamportClock("threaded-worker")
    with ThreadPoolExecutor(max_workers=12) as pool:
        values = list(pool.map(lambda index: clock.tick() if index % 2 else clock.receive(0), range(2000)))
    assert sorted(values) == list(range(1, 2001))


@pytest.mark.parametrize("invalid", ["", "-1", "+1", "1.5", "1e3", " 1", "1 ", "01", "NaN", "true", "١", "9" * 40])
def test_rejects_malformed_clock_headers(invalid):
    with pytest.raises(ValueError):
        parse_time(invalid)


def test_worker_receives_clock_and_sends_higher_time_without_changing_prediction_body():
    with TestClient(create_app(WorkerSettings(gateway_url=None))) as client:
        response = client.post("/predict", json={"features": FEATURES}, headers={TIME_HEADER: "500"})
        assert response.status_code == 200
        assert int(response.headers[RECEIVED_HEADER]) == 501
        assert int(response.headers[TIME_HEADER]) == 502
        assert response.json()["prediction"] == "setosa"
        assert "trace" not in response.json()
        assert response.headers[TRACE_HEADER]


def test_duplicate_or_invalid_incoming_clock_is_rejected_before_inference(monkeypatch):
    worker = create_app(WorkerSettings(gateway_url=None))
    with TestClient(worker) as client:
        def unexpected(*args):
            pytest.fail("Malformed clock headers must not reach model inference")
        monkeypatch.setattr(worker.state.model, "predict_proba", unexpected)
        for headers in ({TIME_HEADER: "bad"}, [(TIME_HEADER, "1"), (TIME_HEADER, "2")]):
            response = client.post("/predict", json={"features": FEATURES}, headers=headers)
            assert response.status_code == 400
            assert response.headers["X-Request-ID"]
            assert int(response.headers[TIME_HEADER]) > int(response.headers[RECEIVED_HEADER])


def test_worker_clock_identity_resets_on_restart():
    worker = create_app(WorkerSettings(gateway_url=None))
    with TestClient(worker) as client:
        first = client.get("/health", headers={TIME_HEADER: "900"})
        previous = worker.state.lamport
        assert int(first.headers[TIME_HEADER]) == 902
    with TestClient(worker) as client:
        second = client.get("/health")
        assert second.headers[INSTANCE_HEADER] != first.headers[INSTANCE_HEADER]
        assert worker.state.lamport is not previous
        assert int(second.headers[RECEIVED_HEADER]) == 1
        assert int(second.headers[TIME_HEADER]) == 2


@pytest.mark.asyncio
async def test_outbound_send_and_receive_use_the_same_clock_and_do_not_modify_payload():
    clock = LamportClock("gateway")
    peer_instance = str(uuid4())
    def handler(request):
        assert request.headers[TIME_HEADER] == "1"
        assert request.headers[INSTANCE_HEADER] == clock.instance_id
        assert request.headers["X-Request-ID"] == "logical-request"
        return httpx.Response(200, headers={TIME_HEADER: "101", RECEIVED_HEADER: "100", INSTANCE_HEADER: peer_instance})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        _, exchange = await clocked_request(client, clock, "GET", "http://worker.test/health",
                                            request_id="logical-request", peer_id="worker")
    assert (exchange.sent, exchange.peer_received, exchange.peer_sent, exchange.received) == (1, 100, 101, 102)
    assert clock.value == 102


@pytest.mark.parametrize("kind", ["incomplete", "bad-number", "bad-order", "bad-instance", "wrong-trace", "duplicate"])
def test_gateway_rejects_peer_clock_metadata_that_cannot_describe_this_exchange(kind):
    def handler(request):
        sent = int(request.headers[TIME_HEADER])
        headers = {TIME_HEADER: str(sent + 2), RECEIVED_HEADER: str(sent + 1),
                   INSTANCE_HEADER: str(uuid4()), TRACE_HEADER: request.headers[TRACE_HEADER]}
        if kind == "incomplete":
            del headers[RECEIVED_HEADER]
        elif kind == "bad-number":
            headers[TIME_HEADER] = "NaN"
        elif kind == "bad-order":
            headers[RECEIVED_HEADER] = str(sent)
        elif kind == "bad-instance":
            headers[INSTANCE_HEADER] = "not-a-process"
        elif kind == "wrong-trace":
            headers[TRACE_HEADER] = str(uuid4())
        elif kind == "duplicate":
            headers = list(headers.items()) + [(TIME_HEADER, str(sent + 3))]
        return httpx.Response(200, json=prediction(request.headers["X-Request-ID"]), headers=headers)
    with gateway_client(handler) as (client, _, _):
        response = client.post("/predict", json={"features": FEATURES})
        assert response.status_code == 502
        trace = client.get("/traces/" + response.headers[TRACE_HEADER]).json()
        assert trace["complete"] is False
        assert trace["status_code"] == 502


def test_failed_prediction_keeps_an_honest_partial_trace():
    def handler(request):
        raise httpx.ConnectError("worker stopped", request=request)
    with gateway_client(handler) as (client, _, _):
        response = client.post("/predict", json={"features": FEATURES})
        assert response.status_code == 503
        trace = client.get("/traces/" + response.headers[TRACE_HEADER]).json()
        assert trace["complete"] is False
        assert [item["event"] for item in trace["events"]] == ["request_received", "request_forwarded", "response_sent"]
        assert trace["request_id"] == response.headers["X-Request-ID"]


def test_legacy_peer_without_clock_metadata_still_predicts_but_has_no_invented_worker_events():
    with gateway_client(lambda request: httpx.Response(200, json=prediction(request.headers["X-Request-ID"]))) as (client, _, _):
        response = client.post("/predict", json={"features": FEATURES})
        assert response.status_code == 200
        trace = client.get("/traces/" + response.headers[TRACE_HEADER]).json()
        assert trace["complete"] is False
        assert all(item["node_id"] == "gateway" for item in trace["events"])


def test_trace_storage_is_bounded_and_returns_immutable_events():
    store = TraceStore(capacity=2)
    clock = LamportClock("gateway")
    for index in range(3):
        trace = InferenceTrace(trace_id=str(index), request_id="same-id", status_code=200, complete=False,
                               events=(ClockEvent(event="response_sent", node_id="gateway", instance_id=clock.instance_id, timestamp=clock.tick()),))
        store.add(trace)
    assert store.get("0") is None
    assert store.get("1").request_id == store.get("2").request_id
    assert isinstance(store.get("2").events, tuple)


def test_gateway_restart_discards_old_traces_and_changes_clock_instance():
    with gateway_client(lambda request: httpx.Response(200, json=prediction(request.headers["X-Request-ID"]))) as (client, app, _):
        response = client.post("/predict", json={"features": FEATURES})
        trace_id = response.headers[TRACE_HEADER]
        old_instance = app.state.lamport.instance_id
        assert client.get("/traces/" + trace_id).status_code == 200
    with TestClient(app) as client:
        assert app.state.lamport.instance_id != old_instance
        assert client.get("/traces/" + trace_id).status_code == 404
