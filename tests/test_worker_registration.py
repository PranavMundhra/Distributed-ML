import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from uuid import UUID

from app.core.config import WorkerSettings
from app.worker.main import create_app


SETTINGS = WorkerSettings(
    worker_id="worker-a", worker_url="http://worker-a.test:8001",
    gateway_url="http://gateway.test:8000", request_timeout_seconds=0.05,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 201])
async def test_worker_registers_loaded_model_once_and_preserves_injected_client(status, tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url == "http://gateway.test:8000/workers/register"
        payload = json.loads(request.content)
        assert UUID(payload["instance_id"])
        assert {key: value for key, value in payload.items() if key != "instance_id"} == {
            "worker_id": "worker-a", "worker_url": "http://worker-a.test:8001/",
            "model": "iris-logistic-regression-v1",
            "models": ["iris-logistic-regression-v1"],
            "demo_control_available": False,
        }
        return httpx.Response(status, json={**payload, "available": None})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        worker = create_app(SETTINGS.model_copy(update={"demo_model_dir": tmp_path}), client=upstream)
        async with worker.router.lifespan_context(worker):
            assert worker.state.ready is True
            assert worker.state.model is not None
            assert len(requests) == 1
        assert not upstream.is_closed
        assert worker.state.ready is False
        assert worker.state.model is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "timeout", "conflict", "redirect", "invalid-json", "wrong-worker"])
async def test_failed_registration_prevents_readiness_and_cleans_up(failure):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if failure == "connection":
            raise httpx.ConnectError("gateway unavailable", request=request)
        if failure == "timeout":
            await asyncio.sleep(0.2)
        if failure == "conflict":
            return httpx.Response(409)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "http://other.test"})
        if failure == "wrong-worker":
            return httpx.Response(201, json={
                "worker_id": "different", "worker_url": "http://worker-a.test:8001",
                "model": "iris-logistic-regression-v1",
            })
        return httpx.Response(201, text="not JSON")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        worker = create_app(SETTINGS, client=upstream)
        with pytest.raises(RuntimeError, match="registration failed"):
            async with worker.router.lifespan_context(worker):
                pytest.fail("A failed registration must not finish startup")
        assert calls == 1
        assert worker.state.ready is False
        assert worker.state.model is None
        assert not upstream.is_closed


def test_worker_still_supports_standalone_inference():
    with TestClient(create_app(WorkerSettings(gateway_url=None))) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}).status_code == 200
