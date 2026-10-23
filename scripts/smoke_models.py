"""Demonstrate three fixed models over real HTTP and optional worker failure."""

import argparse
import base64
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
from smoke_lamport import request, wait_for_workers, EVENTS

HOUSE = dict(area_sqft=1800, bedrooms=3, full_bathrooms=2, garage_spaces=2,
             year_built=2000, overall_quality=7)
SAMPLES = ROOT / "models" / "demo" / "male_female" / "samples"


def examples():
    return [
        ("Iris", "/predict", {"features": [5.1, 3.5, 1.4, .2]}, "setosa"),
        ("House", "/predict/house", HOUSE, None),
        *((f"Face ({label})", "/predict/face", {"image_base64": base64.b64encode(
            (SAMPLES / f"sample_{label}.png").read_bytes()).decode()}, label) for label in ("female", "male")),
    ]


def check_prediction(url, path, payload, expected):
    status, result, headers = request(url + path, payload)
    assert status == 200, result
    if expected:
        assert result["prediction"] == expected, result
    else:
        assert result["currency"] == "USD" and abs(result["predicted_price_usd"] - 232247.41) < .1, result
    status, trace, _ = request(url + "/traces/" + headers["X-Trace-ID"])
    assert status == 200 and trace["complete"] and trace["status_code"] == 200, trace
    assert [item["event"] for item in trace["events"]] == EVENTS
    times = [item["timestamp"] for item in trace["events"]]
    assert all(a < b for a, b in zip(times, times[1:])), trace
    assert trace["events"][2]["node_id"] == result["worker_id"]
    return result


def verify(url):
    status, models, _ = request(url + "/models")
    assert status == 200 and len(models) == 3 and all(item["available_workers"] == 2 for item in models), models
    for title, path, payload, expected in examples():
        results = [check_prediction(url, path, payload, expected) for _ in range(6)]
        ids = [item["worker_id"] for item in results]
        assert len(set(ids)) == 2 and ids == [ids[0], ids[1]] * 3, ids
        value = results[0].get("prediction", results[0].get("predicted_price_usd"))
        print(f"PASS: {title}: {value}; workers={dict(Counter(ids))}; six causal traces.", flush=True)
    assert request(url + "/predict/house", {**HOUSE, "year_built": 2026})[0] == 422
    assert request(url + "/predict/face", {"image_base64": "YWJj"})[0] == 422
    print("PASS: invalid house inputs and unreadable images rejected.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", help="Check an existing two-worker demo without stopping services")
    args = parser.parse_args()
    if args.gateway_url:
        verify(args.gateway_url.rstrip("/"))
        return 0
    with ExitStack() as stack:
        probes = [stack.enter_context(socket.socket()) for _ in range(3)]
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        gateway_port, *worker_ports = [probe.getsockname()[1] for probe in probes]
    url = f"http://127.0.0.1:{gateway_port}"
    children = []
    with tempfile.TemporaryDirectory(prefix="ml-models-") as directory:
        with (Path(directory) / "services.log").open("w+", encoding="utf-8") as log:
            def start(module, port, worker_id=None):
                environment = {**os.environ, "PYTHONUNBUFFERED": "1", "REQUEST_TIMEOUT_SECONDS": "5",
                               "HEARTBEAT_INTERVAL_SECONDS": "0.2", "HEARTBEAT_TIMEOUT_SECONDS": "2",
                               "HEARTBEAT_CHECK_INTERVAL_SECONDS": "0.05"}
                if worker_id:
                    environment.update(WORKER_ID=worker_id, WORKER_URL=f"http://127.0.0.1:{port}", GATEWAY_URL=url)
                child = subprocess.Popen([sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                children.append(child)
                wait_until_ready(child, f"http://127.0.0.1:{port}/health", "worker" if worker_id else "gateway")
                return child
            try:
                start("app.coordinator.main:app", gateway_port)
                first = start("app.worker.main:app", worker_ports[0], "cpu-worker-1")
                start("app.worker.main:app", worker_ports[1], "cpu-worker-2")
                wait_for_workers(url)
                verify(url)
                stop_children([first])
                print("Stopped Worker 1; waiting for background heartbeat expiry...", flush=True)
                # No HTTP health/prediction requests during the expiry window.
                time.sleep(2.5)
                for title, path, payload, expected in examples():
                    result = check_prediction(url, path, payload, expected)
                    assert result["worker_id"] == "cpu-worker-2", result
                    print(f"PASS: {title} still works on Worker 2 after expiry.", flush=True)
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
