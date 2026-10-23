import socket

import pytest

from scripts.run_local import check_ports, monitor_children


@pytest.mark.parametrize("ports", [[8000, 8001, 8001], [0], [65536], [-1]])
def test_launcher_rejects_invalid_or_duplicate_ports(ports):
    with pytest.raises(ValueError):
        check_ports(ports)


def test_launcher_rejects_occupied_port():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        with pytest.raises(RuntimeError, match="unavailable"):
            check_ports([occupied.getsockname()[1]])


def test_launcher_keeps_survivors_running_and_reports_worker_exit_once(monkeypatch, capsys):
    class Child:
        def __init__(self, result):
            self.result = result

        def poll(self):
            return self.result

    polls = 0

    def interrupt_after_two_polls(_):
        nonlocal polls
        polls += 1
        if polls == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("scripts.run_local.time.sleep", interrupt_after_two_polls)
    with pytest.raises(KeyboardInterrupt):
        monitor_children([("gateway", Child(None)), ("worker-1", Child(1)), ("worker-2", Child(None))])
    assert polls == 2
    assert capsys.readouterr().out.count("worker-1 stopped") == 1


def test_launcher_reports_gateway_exit_as_fatal():
    class StoppedGateway:
        def poll(self):
            return 1

    with pytest.raises(RuntimeError, match="Gateway stopped"):
        monitor_children([("gateway", StoppedGateway())])
