from pathlib import Path

import pytest


def test_existing_gateway_checker_accepts_reversed_rejoin_order(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import smoke_round_robin

    turn = 0

    def response(url, payload=None, request_id="unused"):
        nonlocal turn
        if url.endswith("/workers"):
            return 200, [{"worker_id": "cpu-worker-2"}, {"worker_id": "cpu-worker-1"}], None
        worker_id = ["cpu-worker-2", "cpu-worker-1"][turn % 2]
        turn += 1
        return 200, {"prediction": "setosa", "worker_id": worker_id, "request_id": request_id}, request_id

    monkeypatch.setattr(smoke_round_robin, "request_json", response)
    smoke_round_robin.verify("http://gateway.test", ["cpu-worker-1", "cpu-worker-2"], 6, fresh_gateway=False)
    assert "PASS: alternating" in capsys.readouterr().out


def test_existing_gateway_checker_rejects_missing_worker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import smoke_round_robin

    monkeypatch.setattr(smoke_round_robin, "request_json", lambda url: (200, [{"worker_id": "cpu-worker-1"}], None))
    with pytest.raises(AssertionError):
        smoke_round_robin.verify("http://gateway.test", ["cpu-worker-1", "cpu-worker-2"], 6, fresh_gateway=False)
