import pytest
from pydantic import ValidationError

from app.core.config import GatewaySettings, WorkerSettings


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_gateway_requires_positive_finite_timeout(timeout):
    with pytest.raises(ValidationError):
        GatewaySettings(request_timeout_seconds=timeout)


@pytest.mark.parametrize("url", [
    "", "worker.test", "ftp://worker.test", "http://user:password@worker.test",
    "http://worker.test/predict", "http://worker.test?x=1", "http://worker.test#fragment",
])
def test_worker_rejects_invalid_advertised_urls(url):
    with pytest.raises(ValidationError):
        WorkerSettings(worker_url=url)


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("WORKER_ID", "environment-worker")
    monkeypatch.setenv("WORKER_URL", "http://127.0.0.1:9001")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "1.25")
    assert WorkerSettings().worker_id == "environment-worker"
    settings = GatewaySettings()
    assert str(WorkerSettings().worker_url).rstrip("/") == "http://127.0.0.1:9001"
    assert settings.request_timeout_seconds == 1.25


@pytest.mark.parametrize("url", ["http://user:pass@gateway.test", "http://gateway.test/path", "ftp://gateway.test"])
def test_worker_rejects_invalid_gateway_url(url):
    with pytest.raises(ValidationError):
        WorkerSettings(gateway_url=url)


def test_empty_gateway_url_allows_standalone_worker():
    assert WorkerSettings(gateway_url="").gateway_url is None
