"""Demonstrate real heartbeats, expiry, worker recovery, and gateway restart recovery."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from run_local import ROOT, stop_children, wait_until_ready
from smoke_round_robin import request_json, verify


def wait_for_workers(gateway_url, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    latest = []
    while time.monotonic() < deadline:
        status, latest, _ = request_json(f"{gateway_url}/workers")
        if status == 200 and predicate(latest):
            return latest
        time.sleep(0.05)
    raise AssertionError(f"Worker state did not converge: {latest}")


def main() -> int:
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for probe in sockets:
            probe.bind(("127.0.0.1", 0))
        gateway_port, *worker_ports = [probe.getsockname()[1] for probe in sockets]
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    worker_ids = ["cpu-worker-1", "cpu-worker-2"]
    heartbeat_timeout = 1.5  # Faster than the normal 8-second teaching default.
    children = []
    with tempfile.TemporaryDirectory(prefix="ml-heartbeats-") as directory:
        log_path = Path(directory) / "services.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            def start(module, port, worker_id=None):
                environment = os.environ.copy()
                environment.update({
                    "PYTHONUNBUFFERED": "1", "REQUEST_TIMEOUT_SECONDS": "1",
                    "HEARTBEAT_INTERVAL_SECONDS": "0.15",
                    "HEARTBEAT_TIMEOUT_SECONDS": str(heartbeat_timeout),
                    "HEARTBEAT_CHECK_INTERVAL_SECONDS": "0.05",
                })
                if worker_id is not None:
                    environment.update({
                        "GATEWAY_URL": gateway_url, "WORKER_ID": worker_id,
                        "WORKER_URL": f"http://127.0.0.1:{port}",
                    })
                child = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                children.append(child)
                wait_until_ready(child, f"http://127.0.0.1:{port}/health", "worker" if worker_id else "gateway")
                return child

            try:
                gateway = start("app.coordinator.main:app", gateway_port)
                workers = [start("app.worker.main:app", port, worker_id)
                           for port, worker_id in zip(worker_ports, worker_ids)]
                initial = wait_for_workers(gateway_url, lambda items: len(items) == 2 and all(
                    item["heartbeat_sequence"] >= 2 and item["available"] is True for item in items
                ))
                old_instance = initial[0]["instance_id"]
                print("PASS: both workers send periodic heartbeats and report UTC receipt times.", flush=True)
                assert all(item["last_heartbeat_at"] for item in initial)
                verify(gateway_url, worker_ids, 6)

                log_offset = log_path.stat().st_size
                stop_children([workers[0]])
                print("Stopped Worker 1; waiting for heartbeat expiry without health or prediction requests...", flush=True)
                time.sleep(heartbeat_timeout + 0.4)
                # Inspect a separate read handle before any gateway API call:
                # this proves the background sweep detected the stop on its own.
                assert b"worker_heartbeat_expired worker_id=cpu-worker-1" in log_path.read_bytes()[log_offset:]
                state = request_json(f"{gateway_url}/workers")[1]
                assert state[0]["heartbeat_expired"] is True and state[0]["available"] is False, state
                assert state[1]["available"] is True, state
                print("PASS: background detection marks Worker 1 expired while Worker 2 remains available.", flush=True)
                for _ in range(3):
                    status, result, _ = request_json(f"{gateway_url}/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
                    assert status == 200 and result["worker_id"] == "cpu-worker-2", result
                print("PASS: new requests use only Worker 2 after expiry (no retry of failed requests).", flush=True)

                start("app.worker.main:app", worker_ports[0], worker_ids[0])
                recovered = wait_for_workers(gateway_url, lambda items: len(items) == 2 and all(
                    item["available"] is True and item["heartbeat_sequence"] >= 1 for item in items
                ))
                assert recovered[0]["instance_id"] != old_instance
                assert request_json(f"{gateway_url}/workers/heartbeat", {
                    "worker_id": worker_ids[0], "instance_id": old_instance, "sequence": 999999,
                })[0] == 409
                verify(gateway_url, worker_ids, 6, fresh_gateway=False)
                print("PASS: restarting Worker 1 restores scheduling; its old process's heartbeat is rejected.", flush=True)

                stop_children([gateway])
                start("app.coordinator.main:app", gateway_port)
                rejoined = wait_for_workers(gateway_url, lambda items: len(items) == 2 and all(
                    item["available"] is True and item["heartbeat_sequence"] >= 1 for item in items
                ))
                assert {item["worker_id"] for item in rejoined} == set(worker_ids)
                counts = Counter()
                for _ in range(6):
                    status, result, _ = request_json(f"{gateway_url}/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
                    assert status == 200, result
                    counts[result["worker_id"]] += 1
                assert counts == {worker_ids[0]: 3, worker_ids[1]: 3}, counts
                print("PASS: live workers automatically re-register after a gateway restart; predictions recover.", flush=True)
                return 0
            except Exception as error:
                print(f"FAIL: {error}", file=sys.stderr)
                print(log_path.read_text(encoding="utf-8"), file=sys.stderr)
                return 1
            finally:
                stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
