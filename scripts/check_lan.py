"""Check a running two-worker LAN demo without starting or stopping its services."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lan import LAN_HTTP, lan_ip, port_number, worker_identifier

IRIS = "iris-logistic-regression-v1"
HOUSE = "house-price-linear-v1"
FACE = "male-female-cydral-v1"
MODELS = {IRIS, HOUSE, FACE}
EVENTS = ["request_received", "request_forwarded", "worker_request_received",
          "worker_response_sent", "worker_response_received", "response_sent"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fetch(url, payload=None, request_id=None):
    headers = {"Accept": "application/json"}
    if request_id:
        headers["X-Request-ID"] = request_id
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    try:
        with LAN_HTTP.open(Request(url, data=body, headers=headers), timeout=10) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            require(len(data) <= 8 * 1024 * 1024, f"Unexpected oversized response from {url}")
            return data, response.headers
    except HTTPError as error:
        raise RuntimeError(f"{url} returned HTTP {error.code}; inspect the gateway/worker terminal") from error
    except (URLError, OSError) as error:
        raise RuntimeError(f"Cannot reach {url}: {error}. Check service startup, addresses and Private-network firewall access") from error


def request_json(url, payload=None, request_id=None):
    data, headers = fetch(url, payload, request_id)
    return json.loads(data), headers


def worker_origin(value):
    origin = urlsplit(value)
    require(origin.scheme == "http" and not origin.username and not origin.password
            and not origin.query and not origin.fragment and origin.path in ("", "/"),
            f"Expected a plain HTTP worker origin, got {value}")
    host = lan_ip(origin.hostname or "")
    port = port_number(str(80 if origin.port is None else origin.port))
    return host, port


def validate_placement(workers, gateway_ip, allow_same_host=False):
    endpoints = [worker_origin(worker["worker_url"]) for worker in workers.values()]
    require(len(set(endpoints)) == 2, "Workers must advertise two distinct endpoints")
    hosts = {host for host, _ in endpoints}
    require(gateway_ip in hosts, "Expected one worker on the gateway PC; check advertised WORKER_URL values")
    require(len(hosts) == 2 or allow_same_host,
            "Both workers advertise the same IP. Use PC 2's LAN IP, or --allow-same-host for a local rehearsal only")


def cluster_state(url):
    state, _ = request_json(url + "/workers/status")
    require(isinstance(state, dict) and isinstance(state.get("nodes"), list), "Invalid gateway status response")
    return {node["worker"]["worker_id"]: node["worker"] for node in state["nodes"]}


def wait_for_state(url, ids, failed_worker, timeout):
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = cluster_state(url)
        require(not (set(last) - set(ids)), "Unexpected extra workers registered; stop the other demo and restart this gateway")
        if set(last) == set(ids) and all(
            (worker["heartbeat_expired"] is True and worker["available"] is False)
            if name == failed_worker else
            (worker["available"] is True and worker["heartbeat_expired"] is False and worker["heartbeat_sequence"] > 0)
            for name, worker in last.items()
        ):
            return last
        time.sleep(.2)
    summary = {name: {key: worker.get(key) for key in ("available", "heartbeat_expired", "heartbeat_sequence")}
               for name, worker in last.items()}
    raise RuntimeError(f"Timed out waiting for worker registration/heartbeats or requested failure expiry: {summary}")


def verify_heartbeats(url, before, active, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        after = cluster_state(url)
        require(all(name in after and after[name]["instance_id"] == before[name]["instance_id"] for name in before),
                "Worker process changed during verification; rerun after the cluster settles")
        if all(after[name]["available"] is True and not after[name]["heartbeat_expired"]
               and after[name]["heartbeat_sequence"] > before[name]["heartbeat_sequence"] for name in active):
            print("PASS: active workers send increasing heartbeats to the gateway.", flush=True)
            return
        time.sleep(.2)
    raise RuntimeError("Active workers did not advance their heartbeat sequences")


def prediction(url, workers, active, title, path, payload, model, expected):
    request_id = "lan-" + uuid4().hex
    result, headers = request_json(url + path, payload, request_id)
    worker_id = result["worker_id"]
    require(worker_id in active and result["model"] == model and result["request_id"] == request_id
            and headers["X-Request-ID"] == request_id, "Prediction identity/model mismatch")
    if expected is not None:
        require(result["prediction"] == expected, f"Unexpected {title} result")
    else:
        require(result["currency"] == "USD" and math.isfinite(result["predicted_price_usd"]), "Invalid house-price result")
    trace_id = headers["X-Trace-ID"]
    trace, _ = request_json(url + "/traces/" + trace_id)
    events = trace["events"]
    require(trace["complete"] is True and trace["status_code"] == 200 and trace["trace_id"] == trace_id
            and trace["request_id"] == request_id and [event["event"] for event in events] == EVENTS,
            "Missing or incomplete inference trace")
    times = [event["timestamp"] for event in events]
    require(all(a < b for a, b in zip(times, times[1:])), "Lamport events are not causally ordered")
    require(all(events[index]["node_id"] == worker_id and events[index]["instance_id"] == workers[worker_id]["instance_id"]
                for index in (2, 3)), "Trace worker process does not match registration")
    print(f"{title}: {worker_id} @ {workers[worker_id]['worker_url']} | request={request_id} | trace={trace_id}", flush=True)
    return worker_id


def verify(url, ids, failed_worker=None, allow_same_host=False, timeout=20):
    print("Waiting for registrations and accepted heartbeats (status snapshots only)...", flush=True)
    workers = wait_for_state(url, ids, failed_worker, timeout)
    validate_placement(workers, urlsplit(url).hostname, allow_same_host)
    active = set(ids) - {failed_worker}
    for worker in workers.values():
        print(f"Registered: {worker['worker_id']} @ {worker['worker_url']} | process={worker['instance_id']}", flush=True)
    if allow_same_host:
        print("LOCAL REHEARSAL: same-host checks are allowed; this does not verify two physical PCs.", flush=True)
    else:
        print("PASS: two distinct LAN worker addresses. Confirm each terminal is on its intended physical PC.", flush=True)
    if failed_worker:
        print(f"PASS: {failed_worker} has expired through heartbeat silence; only {next(iter(active))} is eligible.", flush=True)
    verify_heartbeats(url, workers, active, timeout)
    for name in active:
        worker = workers[name]
        require(set(worker["models"]) == MODELS, f"{name} lacks prepared models; copy models/demo and restart it")
        health, _ = request_json(worker["worker_url"].rstrip("/") + "/health")
        require(health["service"] == "worker" and health["ready"] is True and health["worker_id"] == name,
                f"Wrong/unready worker at {worker['worker_url']}")
    models, _ = request_json(url + "/models")
    require({entry["model"] for entry in models} == MODELS
            and all(entry["available_workers"] == len(active) for entry in models), "Gateway model availability mismatch")
    cases = [("Iris", "/predict", {"features": [5.1, 3.5, 1.4, .2]}, IRIS, "setosa"),
             ("House", "/predict/house", dict(area_sqft=1800, bedrooms=3, full_bathrooms=2,
                 garage_spaces=2, year_built=2000, overall_quality=7), HOUSE, None)]
    for label in ("female", "male"):
        image, _ = fetch(url + "/samples/face/" + label)
        cases.append((f"Face ({label})", "/predict/face", {"image_base64": base64.b64encode(image).decode()}, FACE, label))
    for title, path, payload, model, expected in cases:
        assignments = [prediction(url, workers, active, title, path, payload, model, expected) for _ in range(6)]
        require(set(assignments) == active, f"{title}: not every eligible worker handled requests")
        if len(active) == 2:
            require(assignments == [assignments[0], assignments[1]] * 3 and assignments[0] != assignments[1],
                    "Round robin did not alternate; keep other prediction clients idle during this check")
        print(f"PASS: {title}, counts={dict(Counter(assignments))}; six complete Lamport traces.", flush=True)
    final = cluster_state(url)
    require(all(name in final and final[name]["instance_id"] == workers[name]["instance_id"] for name in ids),
            "A worker restarted during prediction checks; rerun verification")
    if failed_worker:
        require(final[failed_worker]["heartbeat_expired"] and final[failed_worker]["available"] is False,
                "Failed worker recovered during the survivor check; rerun the normal check")
    print("PASS: all three models and traces verified. Services were left running.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-ip", required=True, type=lan_ip)
    parser.add_argument("--gateway-port", default=8000, type=port_number)
    parser.add_argument("--worker-ids", nargs=2, default=["cpu-worker-1", "cpu-worker-2"], type=worker_identifier)
    parser.add_argument("--failed-worker", type=worker_identifier, help="After stopping this worker, verify expiry and survivor routing")
    parser.add_argument("--allow-same-host", action="store_true", help="Local rehearsal only; does not establish a two-PC deployment")
    parser.add_argument("--wait-seconds", type=port_number, default=20)
    args = parser.parse_args(argv)
    if len(set(args.worker_ids)) != 2 or (args.failed_worker and args.failed_worker not in args.worker_ids):
        parser.error("Use two distinct worker IDs; --failed-worker must name one of them")
    try:
        verify(f"http://{args.gateway_ip}:{args.gateway_port}", args.worker_ids, args.failed_worker,
               args.allow_same_host, args.wait_seconds)
        return 0
    except (RuntimeError, ValueError, KeyError, TypeError, argparse.ArgumentTypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
