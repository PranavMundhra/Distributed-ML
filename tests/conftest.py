"""CPU-only API fixtures; no server process or network is required."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import WorkerSettings
from app.worker.main import create_app


@pytest.fixture(scope="module")
def worker_client():
    application = create_app(WorkerSettings(worker_id="test-cpu-worker"))
    with TestClient(application) as client:
        yield client
