"""Start a real gateway and two workers; verify registration and alternating HTTP inference."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request

from run_local import LOCAL_HTTP, ROOT, stop_children, wait_until_ready


def request_json(url: str, payload=None, request_id="round-robin-check"):
    headers = {"X-Request-ID": request_id}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    try:
        response = LOCAL_HTTP.open(Request(url, data=body, headers=headers), timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), response.headers.get("X-Request-ID")


def verify(gateway_url: str, expected_ids: list[str], count: int, fresh_gateway: bool = True) -> None:
    status, workers, _ = request_json(f"{gateway_url}/workers")
    assert status == 200 and len(workers) == len(expected_ids) and {
        worker["worker_id"] for worker in workers
    } == set(expected_ids), workers
    ids = []
    for index in range(count):
        request_id = f"round-robin-{index + 1}"
        status, result, header = request_json(
            f"{gateway_url}/predict", {"features": [5.1, 3.5, 1.4, 0.2]}, request_id,
        )
        assert status == 200 and result["prediction"] == "setosa", result
        assert header == result["request_id"] == request_id
        ids.append(result["worker_id"])
        print(f"Request {index + 1}: {result['worker_id']} -> {result['prediction']} ({request_id})", flush=True)
    offset = 0 if fresh_gateway else expected_ids.index(ids[0])
    assert ids == [expected_ids[(offset + index) % len(expected_ids)] for index in range(count)], ids
    print(f"PASS: alternating workers and request IDs; counts={dict(Counter(ids))}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", help="Check an existing launcher instead of starting services")
    parser.add_argument("--requests", type=int, default=6)
    args = parser.parse_args()
    if args.requests < 2:
        parser.error("--requests must be at least 2")
    expected_ids = ["cpu-worker-1", "cpu-worker-2"]
    if args.gateway_url:
        verify(args.gateway_url.rstrip("/"), expected_ids, args.requests, fresh_gateway=False)
        return 0

    # Hold all sockets together so each selected temporary port is distinct.
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for probe in sockets:
            probe.bind(("127.0.0.1", 0))
        gateway_port, *worker_ports = [probe.getsockname()[1] for probe in sockets]
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    children = []
    with tempfile.TemporaryDirectory(prefix="ml-round-robin-") as directory:
        with (Path(directory) / "services.log").open("w+", encoding="utf-8") as log_file:
            def start(module, port, worker_id=None):
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                if worker_id is not None:
                    environment.update({"GATEWAY_URL": gateway_url, "WORKER_ID": worker_id,
                                        "WORKER_URL": f"http://127.0.0.1:{port}"})
                child = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                children.append(child)
                wait_until_ready(child, f"http://127.0.0.1:{port}/health", "worker" if worker_id else "gateway")

            try:
                start("app.coordinator.main:app", gateway_port)
                status, health, _ = request_json(f"{gateway_url}/health")
                assert status == 200 and health["registered_workers"] == 0
                assert request_json(f"{gateway_url}/predict", {"features": [5.1, 3.5, 1.4, 0.2]})[0] == 503
                for worker_id, port in zip(expected_ids, worker_ports):
                    start("app.worker.main:app", port, worker_id)
                status, health, _ = request_json(f"{gateway_url}/health")
                assert status == 200 and health["available_workers"] == 2, health
                verify(gateway_url, expected_ids, args.requests)
                status, registered, _ = request_json(f"{gateway_url}/workers")
                assert request_json(f"{gateway_url}/workers/register", {
                    key: registered[0][key] for key in ("worker_id", "worker_url", "model")
                })[0] == 200
                assert len(request_json(f"{gateway_url}/workers")[1]) == 2
                assert request_json(f"{gateway_url}/workers/register", {
                    "worker_id": "bad worker", "worker_url": "http://127.0.0.1:1",
                })[0] == 422
                print("PASS: automatic registration, empty-registry 503, duplicate and invalid registration.", flush=True)
                return 0
            except Exception as error:
                print(f"FAIL: {error}", file=sys.stderr)
                log_file.flush()
                log_file.seek(0)
                print(log_file.read(), file=sys.stderr)
                return 1
            finally:
                stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
