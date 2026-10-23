import asyncio
import json

import httpx
import pytest

from app.core.config import WorkerSettings
from app.schemas.workers import RegisteredWorker
from app.worker.main import create_app


def settings():
    return WorkerSettings(worker_id="worker-a", worker_url="http://worker-a.test:8001",
                          gateway_url="http://gateway.test", heartbeat_interval_seconds=0.01,
                          request_timeout_seconds=0.04)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", [None, "connection", "timeout", "invalid", "server-error", "restart"])
async def test_periodic_heartbeats_recover_and_shutdown_cancels_task(first_failure):
    accepted = asyncio.Event()
    registrations = []
    sequences = []
    current = None

    async def handler(request):
        nonlocal current
        payload = json.loads(request.content)
        if request.url.path == "/workers/register":
            registrations.append(payload)
            current = RegisteredWorker(**payload)
            return httpx.Response(201, json=current.model_dump(mode="json"))
        sequences.append(payload["sequence"])
        assert payload["instance_id"] == registrations[0]["instance_id"]
        if len(sequences) == 1:
            if first_failure == "connection":
                raise httpx.ConnectError("offline", request=request)
            if first_failure == "timeout":
                await asyncio.sleep(0.2)
            if first_failure == "invalid":
                return httpx.Response(200, json={})
            if first_failure == "server-error":
                return httpx.Response(503)
            if first_failure == "restart":
                return httpx.Response(404)
        current.available = True
        current.heartbeat_sequence = payload["sequence"]
        if len(sequences) >= 3:
            accepted.set()
        return httpx.Response(200, json=current.model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        worker = create_app(settings(), client=upstream)
        async with worker.router.lifespan_context(worker):
            task = worker.state.heartbeat_task
            await asyncio.wait_for(accepted.wait(), timeout=2)
            assert worker.state.ready is True
            assert not task.done()
            assert sequences[:3] == [1, 2, 3]
            assert len(registrations) == (2 if first_failure == "restart" else 1)
            if len(registrations) == 2:
                assert registrations[0] == registrations[1]
        assert task.cancelled()
        assert worker.state.heartbeat_task is None
        assert not upstream.is_closed


@pytest.mark.asyncio
async def test_stale_instance_rejection_does_not_reclaim_new_worker_registration():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/workers/register":
            return httpx.Response(201, json=RegisteredWorker(**json.loads(request.content)).model_dump(mode="json"))
        return httpx.Response(409, json={"detail": "replaced worker"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        worker = create_app(settings(), client=upstream)
        async with worker.router.lifespan_context(worker):
            task = worker.state.heartbeat_task
            await asyncio.wait_for(task, timeout=1)
            assert calls == ["/workers/register", "/workers/heartbeat"]
            assert worker.state.ready is True  # Local inference remains independent.


@pytest.mark.asyncio
async def test_worker_rejects_interval_that_cannot_meet_gateway_timeout():
    def handler(request):
        record = RegisteredWorker(**json.loads(request.content), heartbeat_timeout_seconds=0.01)
        return httpx.Response(201, json=record.model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        worker = create_app(settings(), client=upstream)
        with pytest.raises(RuntimeError, match="HEARTBEAT_INTERVAL_SECONDS"):
            async with worker.router.lifespan_context(worker):
                pytest.fail("Unsafe heartbeat settings should fail startup")
        assert worker.state.ready is False
        assert worker.state.model is None


@pytest.mark.asyncio
async def test_owned_heartbeat_client_is_closed_after_task_stops(monkeypatch):
    owned = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(201, json=RegisteredWorker(**json.loads(request.content)).model_dump(mode="json"))
    ))
    monkeypatch.setattr("app.worker.main.httpx.AsyncClient", lambda **kwargs: owned)
    worker = create_app(settings())
    async with worker.router.lifespan_context(worker):
        task = worker.state.heartbeat_task
        assert not owned.is_closed
    assert task.cancelled()
    assert owned.is_closed


@pytest.mark.asyncio
async def test_standalone_worker_does_not_start_heartbeat_task():
    worker = create_app(WorkerSettings(gateway_url=None))
    async with worker.router.lifespan_context(worker):
        assert worker.state.heartbeat_task is None
