import asyncio
from collections import Counter

import httpx
import pytest
from fastapi.testclient import TestClient

from app.coordinator.main import create_app
from app.core.config import GatewaySettings
from test_gateway import FEATURES, WORKER_HEALTH, gateway_client, prediction


def registration(worker_id="worker-a", url="http://worker-a.test:8001"):
    return {"worker_id": worker_id, "worker_url": url, "model": "iris-logistic-regression-v1"}


def no_upstream(request):
    pytest.fail("Registry operations must not contact workers during their startup")


def routed_response(request):
    worker_id = "worker-a" if request.url.host == "worker-a.test" else "worker-b"
    if request.url.path == "/health":
        return httpx.Response(200, json={**WORKER_HEALTH, "worker_id": worker_id})
    return httpx.Response(200, json={
        **prediction(request.headers["x-request-id"]), "worker_id": worker_id,
    })


def register_pair(client):
    assert client.post("/workers/register", json=registration()).status_code == 201
    assert client.post("/workers/register", json=registration(
        "worker-b", "http://worker-b.test:8002"
    )).status_code == 201


def test_empty_registry_and_prediction_without_workers():
    with gateway_client(no_upstream, registered=False) as (client, _, _):
        assert client.get("/workers").json() == []
        health = client.get("/health").json()
        assert health["ready"] is True
        assert health["status"] == "degraded"
        assert health["registered_workers"] == health["available_workers"] == 0
        assert health["worker_url"] is None
        assert client.post("/predict", json={"features": FEATURES}).status_code == 503


def test_registration_listing_and_idempotent_duplicate():
    with gateway_client(no_upstream, registered=False) as (client, _, _):
        response = client.post("/workers/register", json=registration())
        assert response.status_code == 201
        expected = response.json()
        assert expected["worker_id"] == "worker-a"
        assert expected["worker_url"] == "http://worker-a.test:8001/"
        assert expected["available"] is None
        assert expected["heartbeat_sequence"] == 0
        assert expected["heartbeat_expired"] is False
        assert client.post("/workers/register", json=registration()).status_code == 200
        assert client.get("/workers").json() == [expected]


@pytest.mark.parametrize("conflict", [
    registration("worker-a", "http://different.test:8001"),
    registration("worker-b", "http://worker-a.test:8001/"),
])
def test_conflicting_id_or_address_does_not_replace_registry(conflict):
    with gateway_client(no_upstream, registered=False) as (client, _, _):
        original = client.post("/workers/register", json=registration()).json()
        assert client.post("/workers/register", json=conflict).status_code == 409
        assert client.get("/workers").json() == [original]


@pytest.mark.parametrize("payload", [
    {}, {"worker_id": "worker-a"}, {"worker_url": "http://worker-a.test"},
    {**registration(), "worker_id": "bad id"}, {**registration(), "worker_id": "x" * 129},
    {**registration(), "worker_url": "ftp://worker-a.test"},
    {**registration(), "worker_url": "http://user:password@worker-a.test"},
    {**registration(), "worker_url": "http://worker-a.test/predict"},
    {**registration(), "worker_url": "http://worker-a.test?query=1"},
    {**registration(), "worker_url": "http://worker-a.test#fragment"},
    {**registration(), "model": "incompatible-model"},
    {**registration(), "available": True}, [], None,
])
def test_invalid_registration_is_rejected_without_mutation(payload):
    with gateway_client(no_upstream, registered=False) as (client, _, _):
        assert client.post("/workers/register", json=payload).status_code == 422
        assert client.get("/workers").json() == []


def test_six_predictions_alternate_and_health_and_duplicates_do_not_consume_turns():
    with gateway_client(routed_response, registered=False) as (client, _, _):
        register_pair(client)
        ids = []
        for index in range(6):
            health = client.get("/health").json()
            assert health["registered_workers"] == health["available_workers"] == 2
            client.get("/workers")
            assert client.post("/workers/register", json=registration()).status_code == 200
            response = client.post("/predict", json={"features": FEATURES}, headers={
                "X-Request-ID": f"alternating-{index}",
            })
            assert response.status_code == 200
            assert response.json()["request_id"] == f"alternating-{index}"
            ids.append(response.json()["worker_id"])
        assert ids == ["worker-a", "worker-b"] * 3


def test_failed_worker_is_skipped_without_retrying_failed_request_and_can_register_again():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "worker-a.test":
            raise httpx.ConnectError("stopped", request=request)
        return routed_response(request)

    with gateway_client(handler, registered=False) as (client, _, _):
        register_pair(client)
        assert client.post("/predict", json={"features": FEATURES}).status_code == 503
        assert calls == ["worker-a.test"]  # One attempt; no hidden failover or retry.
        assert client.get("/workers").json()[0]["available"] is False
        for _ in range(3):
            response = client.post("/predict", json={"features": FEATURES})
            assert response.status_code == 200
            assert response.json()["worker_id"] == "worker-b"
        assert calls.count("worker-a.test") == 1
        assert client.post("/workers/register", json=registration()).status_code == 200
        assert client.get("/workers").json()[0]["available"] is None


def test_health_rejects_wrong_worker_identity_and_can_restore_worker():
    wrong_identity = True

    def handler(request):
        if wrong_identity:
            return httpx.Response(200, json={**WORKER_HEALTH, "worker_id": "impostor"})
        return routed_response(request)

    with gateway_client(handler, registered=False) as (client, _, _):
        client.post("/workers/register", json=registration())
        assert client.get("/health").json()["worker_available"] is False
        assert client.post("/predict", json={"features": FEATURES}).status_code == 503
        wrong_identity = False
        assert client.get("/health").json()["worker_available"] is True
        assert client.post("/predict", json={"features": FEATURES}).status_code == 200


def test_prediction_rejects_wrong_worker_identity():
    with gateway_client(lambda request: httpx.Response(200, json=prediction()), registered=False) as (client, _, _):
        client.post("/workers/register", json=registration())
        response = client.post("/predict", json={"features": FEATURES}, headers={"X-Request-ID": "gateway-test"})
        assert response.status_code == 502
        assert client.get("/workers").json()[0]["available"] is False


@pytest.mark.asyncio
async def test_concurrent_predictions_are_distributed_evenly():
    async def delayed(request):
        await asyncio.sleep(0.005)
        return routed_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as upstream:
        gateway = create_app(GatewaySettings(), client=upstream)
        async with gateway.router.lifespan_context(gateway):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway.test") as client:
                for item in (registration(), registration("worker-b", "http://worker-b.test:8002")):
                    assert (await client.post("/workers/register", json=item)).status_code == 201
                responses = await asyncio.gather(*(
                    client.post("/predict", json={"features": FEATURES}, headers={"X-Request-ID": f"concurrent-{i}"})
                    for i in range(30)
                ))
                assert all(response.status_code == 200 for response in responses)
                assert Counter(response.json()["worker_id"] for response in responses) == {"worker-a": 15, "worker-b": 15}
                assert len({response.json()["request_id"] for response in responses}) == 30


def test_registry_is_cleared_when_gateway_restarts():
    gateway = create_app(GatewaySettings())

    with TestClient(gateway) as client:
        assert client.post("/workers/register", json=registration()).status_code == 201
    with TestClient(gateway) as client:
        assert client.get("/workers").json() == []


@pytest.mark.asyncio
async def test_failure_after_concurrent_success_still_marks_worker_unavailable():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if request.headers["x-request-id"] == "later-failure":
            await asyncio.sleep(0.02)
            raise httpx.ConnectError("worker stopped", request=request)
        return routed_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        gateway = create_app(GatewaySettings(), client=upstream)
        async with gateway.router.lifespan_context(gateway):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway.test") as client:
                await client.post("/workers/register", json=registration())
                responses = await asyncio.gather(*(
                    client.post("/predict", json={"features": FEATURES}, headers={"X-Request-ID": request_id})
                    for request_id in ("first-success", "later-failure")
                ))
                assert [response.status_code for response in responses] == [200, 503]
                assert (await client.get("/workers")).json()[0]["available"] is False
                assert (await client.post("/predict", json={"features": FEATURES})).status_code == 503
                assert calls == 2


@pytest.mark.asyncio
async def test_old_health_probe_cannot_overwrite_new_registration():
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        started.set()
        await release.wait()
        raise httpx.ConnectError("old worker stopped", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        gateway = create_app(GatewaySettings(), client=upstream)
        async with gateway.router.lifespan_context(gateway):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway.test") as client:
                await client.post("/workers/register", json=registration())
                probe = asyncio.create_task(client.get("/health"))
                await asyncio.wait_for(started.wait(), timeout=1)
                assert (await client.post("/workers/register", json=registration())).status_code == 200
                release.set()
                await asyncio.wait_for(probe, timeout=1)
                assert (await client.get("/workers")).json()[0]["available"] is None
