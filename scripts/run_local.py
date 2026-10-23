"""Run one gateway and registered CPU workers; stop all on Ctrl+C."""

from __future__ import annotations

import argparse
import json
import os
import secrets
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
                return
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{service} did not become ready within {timeout:g} seconds.")


def check_ports(ports: list[int]) -> None:
    if len(set(ports)) != len(ports):
        raise ValueError("Gateway and all worker ports must be different.")
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


def monitor_children(services: list[tuple[str, subprocess.Popen]]) -> None:
    """Keep serving after worker exits; a gateway exit stops the demo."""
    reported_exits = set()
    while True:
        for name, child in services:
            if child.poll() is not None:
                if name == "gateway":
                    raise RuntimeError("Gateway stopped; inspect gateway.log.")
                if name not in reported_exits:
                    print(f"{name} stopped; heartbeat expiry will remove it from routing. See {name}.log.", flush=True)
                    reported_exits.add(name)
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--worker-port", type=int, default=8001)
    parser.add_argument("--worker-count", type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    children: list[subprocess.Popen] = []
    log_files = []
    try:
        worker_ports = [args.worker_port + index for index in range(args.worker_count)]
        check_ports([args.gateway_port, *worker_ports])
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["DEMO_CONTROL_TOKEN"] = secrets.token_urlsafe(32)
        gateway_url = f"http://127.0.0.1:{args.gateway_port}"
        services = [("gateway", "app.coordinator.main:app", args.gateway_port, None)]
        services.extend(
            (f"worker-{index + 1}", "app.worker.main:app", port, f"cpu-worker-{index + 1}")
            for index, port in enumerate(worker_ports)
        )
        for name, module, port, worker_id in services:
            child_environment = environment.copy()
            if worker_id is not None:
                child_environment.update({
                    "GATEWAY_URL": gateway_url, "WORKER_ID": worker_id,
                    "WORKER_URL": f"http://127.0.0.1:{port}",
                })
            log_path = ROOT / f"{name}.log"
            log_file = log_path.open("w", encoding="utf-8")
            log_files.append(log_file)
            child = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT,
                env=child_environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            children.append(child)
            wait_until_ready(child, f"http://127.0.0.1:{port}/health", "worker" if worker_id else "gateway")
            print(f"{name} ready on port {port}; log: {log_path.name}", flush=True)
        with LOCAL_HTTP.open(f"{gateway_url}/health", timeout=10) as response:
            health = json.load(response)
        if health.get("available_workers") != args.worker_count:
            raise RuntimeError("Not all workers registered and became available")
        print(f"Worker dashboard and failure demo: {gateway_url}/workers", flush=True)
        print(f"Open http://127.0.0.1:{args.gateway_port}/ (Ctrl+C stops all services).", flush=True)
        monitor_children([(service[0], child) for service, child in zip(services, children)])
    except KeyboardInterrupt:
        print("\nStopping local services...", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Startup failed: {error} Check gateway.log and worker-*.log.", file=sys.stderr)
        return 1
    finally:
        try:
            stop_children(children)
        finally:
            for log_file in log_files:
                log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
