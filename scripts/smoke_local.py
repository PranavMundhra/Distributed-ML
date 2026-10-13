"""Verify two real local services, inference, worker failure, and recovery."""

from __future__ import annotations

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


def available_ports() -> tuple[int, int]:
    # Hold both sockets until allocation is complete to avoid duplicate ports.
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


def request(url: str, payload: dict | None = None) -> tuple[int, bytes, dict]:
    headers = {"X-Request-ID": "local-smoke-request"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    try:
        response = LOCAL_HTTP.open(Request(url, data=body, headers=headers), timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read(), dict(response.headers)


def main() -> int:
    worker_port, gateway_port = available_ports()
    worker_url = f"http://127.0.0.1:{worker_port}"
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    children: list[subprocess.Popen] = []
    environment = os.environ.copy()
    environment["WORKER_URL"] = worker_url
    environment["REQUEST_TIMEOUT_SECONDS"] = "5"
    environment["WORKER_ID"] = "smoke-cpu-worker"
    environment["PYTHONUNBUFFERED"] = "1"
    with tempfile.TemporaryDirectory(prefix="ml-smoke-") as directory:
        log_path = Path(directory) / "services.log"
        with log_path.open("w+", encoding="utf-8") as log_file:
            def start(module: str, port: int, service: str) -> subprocess.Popen:
                child = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                children.append(child)
                wait_until_ready(child, f"http://127.0.0.1:{port}/health", service)
                return child

            try:
                worker = start("app.worker.main:app", worker_port, "worker")
                start("app.coordinator.main:app", gateway_port, "gateway")
                status, page, _ = request(gateway_url + "/")
                assert status == 200 and b"<html" in page.lower(), "UI page was not served"
                for asset in ("/static/style.css", "/static/app.js"):
                    assert request(gateway_url + asset)[0] == 200, f"Missing UI asset: {asset}"
                for url in (worker_url, gateway_url):
                    assert request(url + "/openapi.json")[0] == 200
                    status, body, headers = request(url + "/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
                    result = json.loads(body)
                    assert status == 200 and result["prediction"] == "setosa", result
                    assert result["worker_id"] == "smoke-cpu-worker"
                    assert result["request_id"] == "local-smoke-request"
                    assert next(v for k, v in headers.items() if k.lower() == "x-request-id") == result["request_id"]
                assert request(gateway_url + "/predict", {"features": [1, 2]})[0] == 422
                print("PASS: UI/assets, OpenAPI, direct and forwarded inference, request IDs, validation.", flush=True)
                stop_children([worker])
                status, body, _ = request(gateway_url + "/health")
                assert status == 200 and json.loads(body)["worker_available"] is False
                assert request(gateway_url + "/predict", {"features": [5.1, 3.5, 1.4, 0.2]})[0] == 503
                print("PASS: stopped worker produces degraded gateway health and HTTP 503.", flush=True)
                start("app.worker.main:app", worker_port, "worker")
                status, body, _ = request(gateway_url + "/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
                assert status == 200 and json.loads(body)["prediction"] == "setosa"
                print("PASS: restarting the worker restores inference without restarting the gateway.", flush=True)
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
