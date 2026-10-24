"""Run one gateway or worker on an explicitly selected private IPv4 interface."""

from __future__ import annotations

import argparse
from ipaddress import IPv4Address, IPv4Network
import json
from pathlib import Path
import re
import socket
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAN_NETWORKS = tuple(IPv4Network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
LAN_HTTP = build_opener(ProxyHandler({}))


def lan_ip(value: str) -> str:
    try:
        address = IPv4Address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a private IPv4 address from ipconfig, such as 192.168.1.10") from error
    if not any(address in network for network in LAN_NETWORKS):
        raise argparse.ArgumentTypeError("Use a private LAN IPv4 address; localhost, public and wildcard addresses are not LAN endpoints")
    return str(address)


def port_number(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Port must be an integer between 1 and 65535") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def worker_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
        raise argparse.ArgumentTypeError("Worker ID must contain 1-128 letters, digits, dots, underscores or hyphens")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    roles = parser.add_subparsers(dest="role", required=True)
    for role, default_port in (("gateway", 8000), ("worker", 8001)):
        command = roles.add_parser(role)
        command.add_argument("--ip", required=True, type=lan_ip, help="This PC's private IPv4 address (ipconfig)")
        command.add_argument("--port", type=port_number, default=default_port)
        command.add_argument("--check-only", action="store_true", help="Check settings/connectivity without starting a service")
        if role == "worker":
            command.add_argument("--gateway-ip", required=True, type=lan_ip, help="PC 1's private IPv4 address")
            command.add_argument("--gateway-port", type=port_number, default=8000)
            command.add_argument("--worker-id", required=True, type=worker_identifier)
    args = parser.parse_args(argv)
    if args.role == "worker" and (args.ip, args.port) == (args.gateway_ip, args.gateway_port):
        parser.error("A worker and gateway on the same PC must use different ports")
    return args


def check_interface(ip: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((ip, port))
        except OSError as error:
            raise RuntimeError(
                f"Cannot bind {ip}:{port}: {error}. Use this PC's current ipconfig address and an unused port."
            ) from error


def check_gateway(url: str) -> None:
    try:
        with LAN_HTTP.open(url + "/workers/status", timeout=5) as response:
            state = json.load(response)
        if not isinstance(state, dict) or not isinstance(state.get("nodes"), list):
            raise ValueError("Unexpected gateway status response")
    except (URLError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot reach the gateway at {url}: {error}. Start PC 1's gateway first; check its IP, port, "
            "Private-network firewall access and Wi-Fi client isolation."
        ) from error


def start_service(args) -> None:
    # Explicit settings override stale .env addresses and disable demo controls,
    # including a token inherited from a previous local-launcher session.
    import uvicorn
    from app.core.config import GatewaySettings, WorkerSettings

    if args.role == "gateway":
        from app.coordinator.main import create_app
        application = create_app(GatewaySettings(demo_control_token=None))
    else:
        from app.worker.main import create_app
        application = create_app(WorkerSettings(
            worker_id=args.worker_id, worker_url=f"http://{args.ip}:{args.port}",
            gateway_url=f"http://{args.gateway_ip}:{args.gateway_port}", demo_control_token=None,
        ))
    uvicorn.run(application, host=args.ip, port=args.port, log_level="info")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        check_interface(args.ip, args.port)
        own_url = f"http://{args.ip}:{args.port}"
        if args.role == "worker":
            gateway_url = f"http://{args.gateway_ip}:{args.gateway_port}"
            check_gateway(gateway_url)
            print(f"Worker {args.worker_id}: {own_url} -> gateway {gateway_url}", flush=True)
            print("Ctrl+C stops only this worker. Restart with the same command to rejoin.", flush=True)
        else:
            print(f"Inference: {own_url}/\nWorker dashboard: {own_url}/workers", flush=True)
        print("LAN demo controls are disabled; use each worker's terminal for real stop/restart.", flush=True)
        if args.check_only:
            print("PASS: local interface/port checked; worker mode also reached the gateway. Remote-to-worker reachability still needs check_lan.py.")
            return 0
        start_service(args)
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"LAN startup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
