"""Verify Lamport traces on real HTTP processes, including concurrent requests and restarts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request

from run_local import LOCAL_HTTP, ROOT, stop_children, wait_until_ready

FEATURES = {"features": [5.1, 3.5, 1.4, 0.2]}
EVENTS = ["request_received", "request_forwarded", "worker_request_received",
          "worker_response_sent", "worker_response_received", "response_sent"]


def request(url: str, payload=None, request_id="lamport-check", headers=None):
    values = {"X-Request-ID": request_id, **(headers or {})}
    body = None
    if payload is not None:
        values["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    try:
        response = LOCAL_HTTP.open(Request(url, data=body, headers=values), timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), response.headers


def prediction_trace(gateway_url: str, index: int):
    request_id = f"lamport-{index}"
    status, result, headers = request(gateway_url + "/predict", FEATURES, request_id)
    assert status == 200 and result["prediction"] == "setosa", result
    assert result["request_id"] == headers["X-Request-ID"] == request_id
    trace_id = headers["X-Trace-ID"]
    status, trace, _ = request(gateway_url + "/traces/" + trace_id)
    assert status == 200 and trace["complete"] is True and trace["status_code"] == 200, trace
    assert trace["trace_id"] == trace_id and trace["request_id"] == request_id
    events = trace["events"]
    assert [item["event"] for item in events] == EVENTS, trace
    times = [item["timestamp"] for item in events]
    assert all(left < right for left, right in zip(times, times[1:])), trace
    assert times[0] == int(headers["X-Lamport-Received"])
    assert times[-1] == int(headers["X-Lamport-Time"])
    assert events[2]["node_id"] == events[3]["node_id"] == result["worker_id"]
    assert events[2]["instance_id"] == events[3]["instance_id"]
    assert events[0]["instance_id"] == events[-1]["instance_id"] == headers["X-Lamport-Instance"]
    return result, trace


def verify(gateway_url: str) -> str:
    results = [prediction_trace(gateway_url, index) for index in range(1, 7)]
    ids = [result["worker_id"] for result, _ in results]
    assert len(set(ids)) == 2 and ids == [ids[0], ids[1]] * 3, ids
    for result, trace in results:
        times = [item["timestamp"] for item in trace["events"]][1:5]
        print(f"{result['worker_id']}: gateway send {times[0]} < worker receive {times[1]} < worker send {times[2]} < gateway receive {times[3]}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(lambda index: prediction_trace(gateway_url, index), range(20, 28)))
    assert len({trace["trace_id"] for _, trace in concurrent}) == 8
    status, result, _ = request(gateway_url + "/predict", FEATURES, headers={"X-Lamport-Time": "invalid"})
    assert status == 400, result
    print("PASS: alternating workers, causal timestamps, eight concurrent traces and invalid-clock rejection.", flush=True)
    return results[-1][1]["trace_id"]


def wait_for_workers(gateway_url: str):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status, workers, _ = request(gateway_url + "/workers")
        if status == 200 and len(workers) == 2 and all(item["heartbeat_sequence"] > 0 for item in workers):
            return workers
        time.sleep(0.1)
    raise RuntimeError("Workers did not register and send heartbeats")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", help="Check an existing two-worker demo without stopping any services")
    args = parser.parse_args()
    if args.gateway_url:
        verify(args.gateway_url.rstrip("/"))
        return 0
    with ExitStack() as stack:
        probes = [stack.enter_context(socket.socket()) for _ in range(3)]
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        gateway_port, *worker_ports = [probe.getsockname()[1] for probe in probes]
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    children = []
    with tempfile.TemporaryDirectory(prefix="ml-lamport-") as directory:
        path = Path(directory) / "services.log"
        with path.open("w+", encoding="utf-8") as log:
            def start(module: str, port: int, worker_id=None):
                environment = {**os.environ, "PYTHONUNBUFFERED": "1", "REQUEST_TIMEOUT_SECONDS": "5",
                               "HEARTBEAT_INTERVAL_SECONDS": "0.2", "HEARTBEAT_TIMEOUT_SECONDS": "4",
                               "HEARTBEAT_CHECK_INTERVAL_SECONDS": "0.05"}
                if worker_id:
                    environment.update({"WORKER_ID": worker_id, "WORKER_URL": f"http://127.0.0.1:{port}", "GATEWAY_URL": gateway_url})
                child = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                children.append(child)
                wait_until_ready(child, f"http://127.0.0.1:{port}/health", "worker" if worker_id else "gateway")
                return child
            try:
                gateway = start("app.coordinator.main:app", gateway_port)
                first = start("app.worker.main:app", worker_ports[0], "cpu-worker-1")
                start("app.worker.main:app", worker_ports[1], "cpu-worker-2")
                old_workers = wait_for_workers(gateway_url)
                previous_trace = verify(gateway_url)
                old_instance = next(item["instance_id"] for item in old_workers if item["worker_id"] == "cpu-worker-1")
                stop_children([first])
                start("app.worker.main:app", worker_ports[0], "cpu-worker-1")
                workers = wait_for_workers(gateway_url)
                assert next(item["instance_id"] for item in workers if item["worker_id"] == "cpu-worker-1") != old_instance
                verify(gateway_url)
                print("PASS: restarted worker has a new clock identity and rejoins causal inference.", flush=True)
                stop_children([gateway])
                start("app.coordinator.main:app", gateway_port)
                wait_for_workers(gateway_url)
                assert request(gateway_url + "/traces/" + previous_trace)[0] == 404
                verify(gateway_url)
                print("PASS: gateway restart clears old traces; live workers re-register and clocks merge correctly.", flush=True)
                return 0
            except Exception as error:
                print(f"FAIL: {error}", file=sys.stderr)
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                return 1
            finally:
                stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
