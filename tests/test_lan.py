"""LAN configuration mistakes and dishonest topology/failure proofs must fail."""

import argparse
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_lan, check_lan


@pytest.mark.parametrize("ip", ["127.0.0.1", "0.0.0.0", "localhost", "8.8.8.8", "169.254.1.1", "::1", "192.168.1.999", "224.0.0.1"])
def test_lan_rejects_non_lan_addresses(ip):
    with pytest.raises(argparse.ArgumentTypeError):
        run_lan.lan_ip(ip)


@pytest.mark.parametrize("ip", ["10.1.2.3", "172.16.66.93", "192.168.1.10"])
def test_lan_accepts_private_ipv4(ip):
    assert run_lan.lan_ip(ip) == ip


@pytest.mark.parametrize("port", ["0", "65536", "-1", "bad"])
def test_lan_rejects_invalid_ports(port):
    with pytest.raises(argparse.ArgumentTypeError):
        run_lan.port_number(port)


def worker_args():
    return ["worker", "--ip", "192.168.1.11", "--gateway-ip", "192.168.1.10", "--worker-id", "cpu-worker-2"]


def test_worker_command_requires_explicit_identity_and_gateway():
    with pytest.raises(SystemExit):
        run_lan.parse_args(["worker", "--ip", "192.168.1.11"])
    with pytest.raises(SystemExit):
        run_lan.parse_args(worker_args() + ["--worker-id", "invalid worker"])
    with pytest.raises(SystemExit):
        run_lan.parse_args(worker_args() + ["--ip", "192.168.1.10", "--port", "8000"])


def test_worker_preflight_does_not_start_server_or_mutate_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(run_lan, "check_interface", lambda ip, port: calls.append((ip, port)))
    monkeypatch.setattr(run_lan, "check_gateway", lambda url: calls.append(url))
    monkeypatch.setattr(run_lan, "start_service", lambda args: pytest.fail("check-only started a service"))
    assert run_lan.main(worker_args() + ["--check-only"]) == 0
    assert calls == [("192.168.1.11", 8001), "http://192.168.1.10:8000"]


def test_worker_does_not_start_when_gateway_is_unreachable(monkeypatch, capsys):
    monkeypatch.setattr(run_lan, "check_interface", lambda *args: None)
    def unreachable(url):
        raise RuntimeError("gateway unreachable")
    monkeypatch.setattr(run_lan, "check_gateway", unreachable)
    monkeypatch.setattr(run_lan, "start_service", lambda args: pytest.fail("started without gateway"))
    assert run_lan.main(worker_args()) == 1
    assert "gateway unreachable" in capsys.readouterr().err


@pytest.mark.parametrize("role", ["gateway", "worker"])
def test_lan_overrides_stale_addresses_and_disables_inherited_controls(role, monkeypatch):
    import uvicorn
    monkeypatch.setenv("DEMO_CONTROL_TOKEN", "x" * 32)
    monkeypatch.setenv("WORKER_ID", "old-worker")
    monkeypatch.setenv("WORKER_URL", "http://127.0.0.1:9991")
    monkeypatch.setenv("GATEWAY_URL", "http://127.0.0.1:9990")
    captured = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.append((app.state.settings, kwargs)))
    args = run_lan.parse_args(worker_args() if role == "worker" else ["gateway", "--ip", "192.168.1.10"])
    run_lan.start_service(args)
    settings, options = captured[0]
    assert settings.demo_control_token is None
    assert options["host"] == args.ip and options["port"] == args.port
    if role == "worker":
        assert settings.worker_id == "cpu-worker-2"
        assert str(settings.worker_url) == "http://192.168.1.11:8001/"
        assert str(settings.gateway_url) == "http://192.168.1.10:8000/"


def workers():
    return {f"cpu-worker-{index}": dict(worker_id=f"cpu-worker-{index}", worker_url=f"http://192.168.1.{9 + index}:8001/",
             instance_id=f"instance-{index}", models=sorted(check_lan.MODELS), heartbeat_sequence=3,
             heartbeat_expired=False, available=True) for index in (1, 2)}


def test_placement_requires_distinct_worker_ips_unless_explicit_local_rehearsal():
    state = workers()
    check_lan.validate_placement(state, "192.168.1.10")
    state["cpu-worker-2"]["worker_url"] = "http://192.168.1.10:8002"
    with pytest.raises(RuntimeError, match="same IP"):
        check_lan.validate_placement(state, "192.168.1.10")
    check_lan.validate_placement(state, "192.168.1.10", allow_same_host=True)
    state["cpu-worker-2"]["worker_url"] = state["cpu-worker-1"]["worker_url"]
    with pytest.raises(RuntimeError, match="distinct endpoints"):
        check_lan.validate_placement(state, "192.168.1.10", allow_same_host=True)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001", "http://0.0.0.0:8001", "http://user:pass@192.168.1.11:8001",
                                "http://192.168.1.11:8001/path", "http://192.168.1.11:8001?x=1", "http://192.168.1.11:0"])
def test_checker_rejects_bad_advertisements(url):
    with pytest.raises((RuntimeError, ValueError, argparse.ArgumentTypeError)):
        check_lan.worker_origin(url)


def fake_time(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(check_lan.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(check_lan.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))


def test_failure_check_requires_heartbeat_expiry_not_just_unavailable(monkeypatch):
    fake_time(monkeypatch)
    state = workers()
    state["cpu-worker-2"]["available"] = False
    monkeypatch.setattr(check_lan, "cluster_state", lambda url: state)
    with pytest.raises(RuntimeError, match="Timed out"):
        check_lan.wait_for_state("http://gateway", list(state), "cpu-worker-2", 1)
    state["cpu-worker-2"]["heartbeat_expired"] = True
    assert check_lan.wait_for_state("http://gateway", list(state), "cpu-worker-2", 1) == state


def test_checker_rejects_unexpected_extra_workers(monkeypatch):
    state = workers()
    monkeypatch.setattr(check_lan, "cluster_state", lambda url: {**state, "unexpected": {}})
    with pytest.raises(RuntimeError, match="extra workers"):
        check_lan.wait_for_state("http://gateway", list(state), None, 1)


def test_heartbeat_check_requires_new_sequence_and_same_process(monkeypatch):
    fake_time(monkeypatch)
    before = workers()
    after = deepcopy(before)
    monkeypatch.setattr(check_lan, "cluster_state", lambda url: after)
    with pytest.raises(RuntimeError, match="advance"):
        check_lan.verify_heartbeats("http://gateway", before, set(before), 1)
    for worker in after.values():
        worker["heartbeat_sequence"] += 1
    check_lan.verify_heartbeats("http://gateway", before, set(before), 1)
    after["cpu-worker-2"]["instance_id"] = "new-process"
    with pytest.raises(RuntimeError, match="process changed"):
        check_lan.verify_heartbeats("http://gateway", before, set(before), 1)


@pytest.mark.parametrize("fault", ["wrong-worker", "wrong-instance", "bad-order", "missing-event", "wrong-request"])
def test_checker_rejects_misleading_traces(fault, monkeypatch):
    state = workers()
    request_id = "lan-fixed"
    monkeypatch.setattr(check_lan, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    result = dict(worker_id="cpu-worker-1", model=check_lan.IRIS, request_id=request_id, prediction="setosa")
    events = [dict(event=name, timestamp=index + 1, node_id="cpu-worker-1", instance_id="instance-1")
              for index, name in enumerate(check_lan.EVENTS)]
    trace = dict(complete=True, status_code=200, trace_id="trace", request_id=request_id, events=events)
    if fault == "wrong-worker":
        events[2]["node_id"] = "cpu-worker-2"
    elif fault == "wrong-instance":
        events[2]["instance_id"] = "old-instance"
    elif fault == "bad-order":
        events[2]["timestamp"] = 1
    elif fault == "missing-event":
        events.pop()
    else:
        trace["request_id"] = "other-request"
    def respond(url, *args):
        return (trace, {}) if "/traces/" in url else (result, {"X-Trace-ID": "trace", "X-Request-ID": request_id})
    monkeypatch.setattr(check_lan, "request_json", respond)
    with pytest.raises(RuntimeError):
        check_lan.prediction("http://gateway", state, set(state), "Iris", "/predict", {}, check_lan.IRIS, "setosa")
