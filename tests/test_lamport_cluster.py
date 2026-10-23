"""Exercise real gateway and worker apps with independent clocks and shared HTTP transport."""

import asyncio
from collections import Counter, defaultdict
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import pytest

from app.coordinator.main import create_app as create_gateway
from app.core.config import GatewaySettings, WorkerSettings
from app.core.lamport import INSTANCE_HEADER, RECEIVED_HEADER, TIME_HEADER, TRACE_HEADER
from app.worker.main import create_app as create_worker
from test_gateway import FEATURES


@asynccontextmanager
async def cluster(interval=2, demo_token=None, heartbeat_timeout=8):
    applications = {}
    async def route(request):
        transport = httpx.ASGITransport(app=applications[request.url.host])
        return await transport.handle_async_request(request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as transport:
        gateway = create_gateway(GatewaySettings(demo_control_token=demo_token,
            heartbeat_timeout_seconds=heartbeat_timeout,
            heartbeat_check_interval_seconds=min(.5, heartbeat_timeout / 4)), client=transport)
        applications["gateway.test"] = gateway
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(gateway.router.lifespan_context(gateway))
            workers = []
            for index in (1, 2):
                worker = create_worker(WorkerSettings(
                    worker_id=f"worker-{index}", worker_url=f"http://worker-{index}.test",
                    gateway_url="http://gateway.test", heartbeat_interval_seconds=interval,
                    demo_control_token=demo_token,
                ), client=transport)
                applications[f"worker-{index}.test"] = worker
                workers.append(worker)
                await stack.enter_async_context(worker.router.lifespan_context(worker))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway.test") as client:
                yield client, gateway, workers


def assert_causal_trace(trace, response):
    assert trace["complete"] is True
    assert trace["status_code"] == 200
    assert trace["trace_id"] == response.headers[TRACE_HEADER]
    assert trace["request_id"] == response.json()["request_id"]
    assert [item["event"] for item in trace["events"]] == [
        "request_received", "request_forwarded", "worker_request_received",
        "worker_response_sent", "worker_response_received", "response_sent",
    ]
    times = [item["timestamp"] for item in trace["events"]]
    assert all(left < right for left, right in zip(times, times[1:]))
    assert times[0] == int(response.headers[RECEIVED_HEADER])
    assert times[-1] == int(response.headers[TIME_HEADER])
    assert trace["events"][2]["node_id"] == trace["events"][3]["node_id"] == response.json()["worker_id"]
    assert trace["events"][2]["instance_id"] == trace["events"][3]["instance_id"]
    assert trace["events"][2]["instance_id"] != response.headers[INSTANCE_HEADER]


@pytest.mark.asyncio
async def test_independent_real_workers_produce_causal_traces_and_keep_round_robin():
    async with cluster() as (client, gateway, workers):
        assert len({gateway.state.lamport.instance_id, *(worker.state.lamport.instance_id for worker in workers)}) == 3
        results = []
        for _ in range(6):
            response = await client.post("/predict", json={"features": FEATURES}, headers={"X-Request-ID": "reused-client-id"})
            assert response.status_code == 200
            trace = (await client.get("/traces/" + response.headers[TRACE_HEADER])).json()
            assert_causal_trace(trace, response)
            results.append((response, trace))
        assert [response.json()["worker_id"] for response, _ in results] == ["worker-1", "worker-2"] * 3
        assert len({trace["trace_id"] for _, trace in results}) == 6


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_mix_traces_even_when_request_ids_are_reused():
    async with cluster(interval=0.01) as (client, _, _):
        responses = await asyncio.wait_for(asyncio.gather(*(
            client.post("/predict", json={"features": FEATURES}, headers={"X-Request-ID": "same-for-all"})
            for _ in range(30)
        )), timeout=5)
        assert all(response.status_code == 200 for response in responses)
        assert Counter(response.json()["worker_id"] for response in responses) == {"worker-1": 15, "worker-2": 15}
        traces = [(await client.get("/traces/" + response.headers[TRACE_HEADER])).json() for response in responses]
        assert len({trace["trace_id"] for trace in traces}) == 30
        timestamps = defaultdict(set)
        for response, trace in zip(responses, traces):
            assert_causal_trace(trace, response)
            for item in trace["events"]:
                key = item["instance_id"]
                assert item["timestamp"] not in timestamps[key]
                timestamps[key].add(item["timestamp"])


@pytest.mark.asyncio
async def test_registration_heartbeat_and_health_share_the_inference_clock():
    async with cluster(interval=0.01) as (client, gateway, workers):
        initial = [worker.state.lamport.value for worker in workers]
        assert all(value > 0 for value in initial)  # Startup registration exchanged clocks.
        deadline = asyncio.get_running_loop().time() + 2
        while not all(worker.heartbeat_sequence >= 2 for worker in gateway.state.registry.list_workers()):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert all(worker.state.lamport.value > before for worker, before in zip(workers, initial))
        before = gateway.state.lamport.value
        health = await client.get("/health")
        assert health.json()["available_workers"] == 2
        assert int(health.headers[TIME_HEADER]) > before
        response = await client.post("/predict", json={"features": FEATURES})
        assert response.status_code == 200
        assert_causal_trace((await client.get("/traces/" + response.headers[TRACE_HEADER])).json(), response)
