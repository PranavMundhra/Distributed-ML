import asyncio
from datetime import timezone

import httpx
import pytest

from app.coordinator.main import create_app
from app.coordinator.registry import HeartbeatConflict, UnknownWorker, WorkerRegistry
from app.core.config import GatewaySettings, WorkerSettings
from app.schemas.workers import WorkerHeartbeat, WorkerRegistration
from test_gateway import FEATURES, WORKER_HEALTH, gateway_client
from test_registration import registration


def registered(clock, timeout=8):
    registry = WorkerRegistry(timeout, clock=lambda: clock[0])
    worker, _ = registry.register(WorkerRegistration(**registration(), instance_id="instance-a"))
    return registry, worker


def beat(sequence=1, instance="instance-a", worker_id="worker-a"):
    return WorkerHeartbeat(worker_id=worker_id, instance_id=instance, sequence=sequence)


def test_registration_grace_and_expiry_at_exact_monotonic_deadline():
    clock = [100.0]
    registry, worker = registered(clock)
    clock[0] = 107.999
    assert registry.select() is worker
    clock[0] = 108.0
    assert registry.expire_workers() == ["worker-a"]
    assert worker.available is False and worker.heartbeat_expired is True
    assert registry.select() is None
    assert registry.expire_workers() == []


def test_heartbeat_extends_deadline_and_recovers_expired_worker():
    clock = [100.0]
    registry, worker = registered(clock)
    clock[0] = 107.0
    assert registry.heartbeat(beat()) is worker
    assert worker.last_heartbeat_at.tzinfo == timezone.utc
    clock[0] = 114.999
    assert registry.select() is worker
    clock[0] = 115.0
    assert registry.select() is None
    registry.heartbeat(beat(2))
    assert registry.select() is worker
    assert worker.available is True and worker.heartbeat_expired is False


@pytest.mark.parametrize("payload", [beat(1), beat(2, "old-instance")])
def test_stale_or_wrong_instance_heartbeat_does_not_extend_deadline(payload):
    clock = [100.0]
    registry, worker = registered(clock)
    registry.heartbeat(beat(1))
    clock[0] = 107.0
    with pytest.raises(HeartbeatConflict):
        registry.heartbeat(payload)
    assert worker.heartbeat_sequence == 1
    clock[0] = 108.0
    assert registry.select() is None


def test_unknown_worker_cannot_join_through_heartbeat():
    registry = WorkerRegistry()
    with pytest.raises(UnknownWorker):
        registry.heartbeat(beat())
    assert registry.list_workers() == []


def test_new_process_rejects_previous_incarnation_and_resets_sequence():
    clock = [100.0]
    registry, old = registered(clock)
    registry.heartbeat(beat(10))
    new, created = registry.register(WorkerRegistration(**registration(), instance_id="new-instance"))
    assert not created and new is not old
    assert new.heartbeat_sequence == 0
    with pytest.raises(HeartbeatConflict):
        registry.heartbeat(beat(11))
    registry.heartbeat(beat(1, "new-instance"))
    assert new.available is True


def test_repeat_registration_preserves_sequence_and_live_instance():
    clock = [100.0]
    registry, _ = registered(clock)
    registry.heartbeat(beat(10))
    new, _ = registry.register(WorkerRegistration(**registration()))
    assert new.instance_id == "instance-a" and new.heartbeat_sequence == 10
    registry.heartbeat(beat(11))
    assert new.available is True


def test_successful_http_observations_cannot_renew_expired_heartbeat():
    clock = [100.0]
    registry, worker = registered(clock)
    clock[0] = 107.0
    registry.set_available(worker, True)
    clock[0] = 108.0
    registry.set_available(worker, True)
    assert registry.select() is None
    assert worker.available is False and worker.heartbeat_expired is True


def test_heartbeat_endpoint_and_monotonic_sequence():
    with gateway_client(lambda request: pytest.fail("No outbound request expected"), registered=False) as (client, _, _):
        item = {**registration(), "instance_id": "instance-a"}
        assert client.post("/workers/register", json=item).status_code == 201
        response = client.post("/workers/heartbeat", json=beat().model_dump())
        assert response.status_code == 200
        result = response.json()
        assert result["available"] is True and result["heartbeat_sequence"] == 1
        assert result["last_heartbeat_at"] is not None
        assert client.post("/workers/heartbeat", json=beat().model_dump()).status_code == 409
        assert client.post("/workers/heartbeat", json=beat(2, "old").model_dump()).status_code == 409
        assert client.post("/workers/heartbeat", json=beat(2, worker_id="unknown").model_dump()).status_code == 404
        assert client.get("/workers").json()[0]["heartbeat_sequence"] == 1


@pytest.mark.parametrize("payload", [
    {}, {"worker_id": "worker-a"},
    {**beat().model_dump(), "sequence": 0}, {**beat().model_dump(), "sequence": -1},
    {**beat().model_dump(), "sequence": True}, {**beat().model_dump(), "sequence": 1.5},
    {**beat().model_dump(), "sequence": "1"}, {**beat().model_dump(), "worker_id": "bad id"},
    {**beat().model_dump(), "instance_id": ""},
    {**beat().model_dump(), "timestamp": "2099-01-01"}, [], None,
])
def test_invalid_heartbeat_is_rejected(payload):
    with gateway_client(lambda request: pytest.fail("No outbound request expected"), registered=False) as (client, _, _):
        assert client.post("/workers/heartbeat", json=payload).status_code == 422
        assert client.get("/workers").json() == []


@pytest.mark.asyncio
async def test_background_expiry_runs_without_any_health_or_prediction_requests():
    gateway = create_app(GatewaySettings(heartbeat_timeout_seconds=0.06, heartbeat_check_interval_seconds=0.01))
    async with gateway.router.lifespan_context(gateway):
        worker, _ = gateway.state.registry.register(WorkerRegistration(**registration(), instance_id="instance-a"))
        expiry_task = gateway.state.expiry_task
        await asyncio.sleep(0.12)
        # Read the object directly: list_workers() and select() also check expiry,
        # so calling either here could hide a broken background task.
        assert worker.heartbeat_expired is True and worker.available is False
        assert not expiry_task.done()
    assert expiry_task.cancelled()
    assert gateway.state.expiry_task is None


@pytest.mark.asyncio
async def test_health_endpoint_cannot_revive_worker_with_expired_heartbeat():
    clock = [100.0]
    registry, _ = registered(clock)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={**WORKER_HEALTH, "worker_id": "worker-a"})
    )) as upstream:
        gateway = create_app(GatewaySettings(), client=upstream)
        async with gateway.router.lifespan_context(gateway):
            gateway.state.registry = registry
            clock[0] = 108.0
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway.test") as client:
                assert (await client.get("/health")).json()["worker_available"] is False
                assert (await client.post("/predict", json={"features": FEATURES})).status_code == 503


@pytest.mark.parametrize("field", ["heartbeat_timeout_seconds", "heartbeat_check_interval_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 301])
def test_gateway_rejects_invalid_heartbeat_settings(field, value):
    with pytest.raises(ValueError):
        GatewaySettings(**{field: value})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 61])
def test_worker_rejects_invalid_heartbeat_interval(value):
    with pytest.raises(ValueError):
        WorkerSettings(heartbeat_interval_seconds=value)


def test_expiry_interval_must_not_exceed_timeout():
    with pytest.raises(ValueError):
        GatewaySettings(heartbeat_timeout_seconds=1, heartbeat_check_interval_seconds=2)
