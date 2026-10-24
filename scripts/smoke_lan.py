"""Rehearse the LAN launchers on ONE PC's private interface, using temporary ports."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lan import lan_ip
from scripts.run_local import stop_children, wait_until_ready
from scripts.check_lan import cluster_state, wait_for_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True, type=lan_ip, help="This PC's ipconfig IPv4 address")
    args = parser.parse_args(argv)
    print("ONE-PC REHEARSAL: this does not verify a second physical computer or its firewall.", flush=True)
    children = []
    with ExitStack() as stack:
        probes = [stack.enter_context(socket.socket()) for _ in range(3)]
        for probe in probes:
            probe.bind((args.ip, 0))
        gateway_port, *worker_ports = [probe.getsockname()[1] for probe in probes]
    url = f"http://{args.ip}:{gateway_port}"
    ids = ["cpu-worker-1", "cpu-worker-2"]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "HEARTBEAT_INTERVAL_SECONDS": ".2",
                   "HEARTBEAT_TIMEOUT_SECONDS": "1.5", "HEARTBEAT_CHECK_INTERVAL_SECONDS": ".05",
                   # Ensure LAN settings override inherited local-demo controls.
                   "DEMO_CONTROL_TOKEN": "x" * 32}
    with tempfile.TemporaryDirectory(prefix="ml-lan-") as directory:
        with (Path(directory) / "services.log").open("w+", encoding="utf-8") as log:
            def start(role, port, worker_id=None):
                command = [sys.executable, str(ROOT / "scripts/run_lan.py"), role, "--ip", args.ip, "--port", str(port)]
                if worker_id:
                    command += ["--gateway-ip", args.ip, "--gateway-port", str(gateway_port), "--worker-id", worker_id]
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                children.append(child)
                wait_until_ready(child, f"http://{args.ip}:{port}/health", role)
                return child

            def check(*extra, expect_failure=False):
                command = [sys.executable, str(ROOT / "scripts/check_lan.py"), "--gateway-ip", args.ip,
                           "--gateway-port", str(gateway_port), *extra]
                result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if expect_failure:
                    if result.returncode != 1 or "same IP" not in result.stderr:
                        raise RuntimeError(result.stdout + result.stderr)
                elif result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)

            try:
                gateway = start("gateway", gateway_port)
                start("worker", worker_ports[0], ids[0])
                second = start("worker", worker_ports[1], ids[1])
                before = wait_for_state(url, ids, None, 20)
                from scripts.check_lan import request_json
                state, _ = request_json(url + "/workers/status")
                if state["demo_controls_enabled"] or any(worker["demo_control_available"] for worker in before.values()):
                    raise RuntimeError("LAN services inherited demo controls")
                check(expect_failure=True)
                print("PASS: the default checker rejects one-IP placement; inherited demo controls are disabled.", flush=True)
                check("--allow-same-host")
                print("PASS: actual LAN-interface launchers, all models, round robin, heartbeats and traces (local rehearsal).", flush=True)
                stop_children([second])
                check("--allow-same-host", "--failed-worker", ids[1])
                print("PASS: real Worker 2 process exit, heartbeat expiry and all-model survivor routing.", flush=True)
                start("worker", worker_ports[1], ids[1])
                after = wait_for_state(url, ids, None, 20)
                if before[ids[1]]["instance_id"] == after[ids[1]]["instance_id"]:
                    raise RuntimeError("Restart did not create a new worker process identity")
                check("--allow-same-host")
                print("PASS: worker restart creates a new identity and restores routing.", flush=True)
                stop_children([gateway])
                start("gateway", gateway_port)
                check("--allow-same-host")
                print("PASS: LAN workers re-register after gateway restart. Two-PC verification remains pending.", flush=True)
                return 0
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                print(f"FAIL: {error}", file=sys.stderr)
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                return 1
            finally:
                stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
