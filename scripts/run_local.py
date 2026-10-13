"""Run the local gateway and worker; stop both on Ctrl+C or startup failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
LOCAL_HTTP = build_opener(ProxyHandler({}))


def wait_until_ready(
    process: subprocess.Popen, url: str, service: str, timeout: float = 30
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{service} exited with code {process.returncode}.")
        try:
            with LOCAL_HTTP.open(url, timeout=1) as response:
                health = json.load(response)
            if health.get("service") == service and health.get("ready") is True:
                if service == "worker" or health.get("worker_available") is True:
                    return
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{service} did not become ready within {timeout:g} seconds.")


def check_ports(ports: list[int]) -> None:
    if len(set(ports)) != len(ports):
        raise ValueError("Gateway and worker ports must be different.")
    for port in ports:
        if not 1 <= port <= 65535:
            raise ValueError("Ports must be between 1 and 65535.")
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(
                    f"Port {port} is unavailable. Stop its service or choose another port."
                ) from error


def stop_children(children: list[subprocess.Popen]) -> None:
    failures = []
    for child in reversed(children):
        if child.poll() is None:
            try:
                if os.name == "nt":
                    # A Windows venv python.exe can be a redirector with a child
                    # interpreter. Stop its entire owned tree before its parent.
                    result = subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True, text=True, timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if result.returncode and child.poll() is None:
                        failures.append(f"PID {child.pid}: {result.stderr.strip()}")
                else:
                    child.terminate()
            except (OSError, subprocess.TimeoutExpired) as error:
                failures.append(f"PID {child.pid}: {error}")
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            failures.append(f"PID {child.pid} did not stop")
    if failures:
        raise RuntimeError("Could not stop all local services: " + "; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--worker-port", type=int, default=8001)
    args = parser.parse_args()
    children: list[subprocess.Popen] = []
    log_files = []
    try:
        check_ports([args.gateway_port, args.worker_port])
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        # Keep the gateway pointed at the worker this launcher actually starts.
        environment["WORKER_URL"] = f"http://127.0.0.1:{args.worker_port}"
        for service, module, port in [
            ("worker", "app.worker.main:app", args.worker_port),
            ("gateway", "app.coordinator.main:app", args.gateway_port),
        ]:
            log_path = ROOT / f"{service}.log"
            log_file = log_path.open("w", encoding="utf-8")
            log_files.append(log_file)
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
            print(f"{service.capitalize()} ready on port {port}; log: {log_path.name}", flush=True)
        print(f"Open http://127.0.0.1:{args.gateway_port}/ (Ctrl+C stops both services).", flush=True)
        while True:
            for service, child in zip(("worker", "gateway"), children):
                if child.poll() is not None:
                    raise RuntimeError(f"{service} stopped; inspect {service}.log.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping local services...", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Startup failed: {error} Check worker.log and gateway.log.", file=sys.stderr)
        return 1
    finally:
        try:
            stop_children(children)
        finally:
            for log_file in log_files:
                log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
