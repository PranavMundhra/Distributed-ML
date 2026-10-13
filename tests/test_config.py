import pytest
from pydantic import ValidationError

from app.core.config import GatewaySettings, WorkerSettings


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_gateway_requires_positive_finite_timeout(timeout):
    with pytest.raises(ValidationError):
        GatewaySettings(request_timeout_seconds=timeout)


@pytest.mark.parametrize("url", ["", "worker.test", "ftp://worker.test", "http://user:password@worker.test"])
def test_gateway_rejects_invalid_worker_urls(url):
    with pytest.raises(ValidationError):
        GatewaySettings(worker_url=url)


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("WORKER_ID", "environment-worker")
    monkeypatch.setenv("WORKER_URL", "http://127.0.0.1:9001")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "1.25")
    assert WorkerSettings().worker_id == "environment-worker"
    settings = GatewaySettings()
    assert str(settings.worker_url).rstrip("/") == "http://127.0.0.1:9001"
    assert settings.request_timeout_seconds == 1.25
