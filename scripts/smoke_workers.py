"""Verify dashboard observation, demo pause, and actual worker crash/restart on real HTTP."""

from contextlib import ExitStack
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

from run_local import ROOT, stop_children, wait_until_ready
from smoke_lamport import request, prediction_trace, wait_for_workers
from smoke_models import check_prediction, examples


def wait_status(url, predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, state, _ = request(url + "/workers/status")
        if status == 200 and predicate(state):
            return state
        time.sleep(.05)
    raise RuntimeError("Worker dashboard did not reach the expected state")


def alternating(url):
    ids = [prediction_trace(url, index)[0]["worker_id"] for index in range(6)]
    assert len(set(ids)) == 2 and ids == [ids[0], ids[1]] * 3, ids


def main():
    with ExitStack() as stack:
        probes = [stack.enter_context(socket.socket()) for _ in range(3)]
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        gateway_port, *worker_ports = [probe.getsockname()[1] for probe in probes]
    url = f"http://127.0.0.1:{gateway_port}"
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "REQUEST_TIMEOUT_SECONDS": "5",
                   "HEARTBEAT_INTERVAL_SECONDS": ".15", "HEARTBEAT_TIMEOUT_SECONDS": "1.5",
                   "HEARTBEAT_CHECK_INTERVAL_SECONDS": ".05", "DEMO_CONTROL_TOKEN": secrets.token_urlsafe(32)}
    children = []
    with tempfile.TemporaryDirectory(prefix="ml-workers-") as directory:
        with (Path(directory) / "services.log").open("w+", encoding="utf-8") as log:
            def start(module, port, worker_id=None):
                values = environment.copy()
                if worker_id:
                    values.update(WORKER_ID=worker_id, WORKER_URL=f"http://127.0.0.1:{port}", GATEWAY_URL=url)
                child = subprocess.Popen([sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=values, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                children.append(child)
                wait_until_ready(child, f"http://127.0.0.1:{port}/health", "worker" if worker_id else "gateway")
                return child
            try:
                gateway = start("app.coordinator.main:app", gateway_port)
                first = start("app.worker.main:app", worker_ports[0], "cpu-worker-1")
                second = start("app.worker.main:app", worker_ports[1], "cpu-worker-2")
                workers = wait_for_workers(url)
                first_instance = workers[0]["instance_id"]
                second_instance = workers[1]["instance_id"]
                alternating(url)
                state = request(url + "/workers/status")[1]
                assert [node["worker"]["completed_predictions"] for node in state["nodes"]] == [3, 3]
                assert len(state["recent_predictions"]) == 6
                assert "DEMO_CONTROL_TOKEN" not in str(state)
                print("PASS: six real predictions alternate; dashboard counts and recent traces match.", flush=True)
                status, result, _ = request(url + "/workers/cpu-worker-1/demo", {"action": "pause", "instance_id": first_instance})
                assert status == 200 and result["paused"]
                assert request(f"http://127.0.0.1:{worker_ports[0]}/health")[0] == 503
                # Only read registry snapshots while waiting: no gateway health probes or predictions.
                state = wait_status(url, lambda value: value["nodes"][0]["worker"]["heartbeat_expired"])
                assert state["nodes"][1]["worker"]["available"] is True
                for title, path, payload, expected in examples():
                    assert check_prediction(url, path, payload, expected)["worker_id"] == "cpu-worker-2"
                print("PASS: paused worker stops heartbeats/inference; background expiry routes every model to Worker 2.", flush=True)
                assert request(url + "/workers/cpu-worker-1/demo", {"action": "resume", "instance_id": first_instance})[0] == 200
                state = wait_status(url, lambda value: value["nodes"][0]["worker"]["available"] is True)
                assert state["nodes"][0]["worker"]["instance_id"] == first_instance
                alternating(url)
                print("PASS: resume restores scheduling through a new heartbeat, with the same process identity.", flush=True)
                stop_children([second])
                wait_status(url, lambda value: value["nodes"][1]["worker"]["heartbeat_expired"])
                assert prediction_trace(url, 100)[0]["worker_id"] == "cpu-worker-1"
                start("app.worker.main:app", worker_ports[1], "cpu-worker-2")
                state = wait_status(url, lambda value: value["nodes"][1]["worker"]["available"] is True)
                assert state["nodes"][1]["worker"]["instance_id"] != second_instance
                assert state["nodes"][1]["worker"]["completed_predictions"] == 0
                alternating(url)
                print("PASS: actual worker process stop/restart is detected; new process identity and fresh counter appear.", flush=True)
                stop_children([gateway])
                start("app.coordinator.main:app", gateway_port)
                wait_for_workers(url)
                state = request(url + "/workers/status")[1]
                assert all(node["worker"]["completed_predictions"] == 0 for node in state["nodes"])
                assert state["recent_predictions"] == []
                alternating(url)
                print("PASS: gateway restart clears dashboard counters/history; live workers re-register.", flush=True)
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
