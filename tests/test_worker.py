import json
import math
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.config import WorkerSettings
from app.worker.main import create_app


@pytest.mark.parametrize(
    ("features", "species", "class_id"),
    [
        ([5.1, 3.5, 1.4, 0.2], "setosa", 0),
        ([6.0, 2.9, 4.5, 1.5], "versicolor", 1),
        ([6.7, 3.1, 5.6, 2.4], "virginica", 2),
    ],
)
def test_predict_known_iris_samples(worker_client, features, species, class_id):
    first = worker_client.post("/predict", json={"features": features})
    second = worker_client.post("/predict", json={"features": features})
    assert first.status_code == second.status_code == 200
    result = first.json()
    assert set(result) == {
        "prediction", "class_id", "probabilities", "model", "worker_id", "request_id"
    }
    assert result["prediction"] == species
    assert result["class_id"] == class_id
    assert result["worker_id"] == "test-cpu-worker"
    assert result["model"] == "iris-logistic-regression-v1"
    probabilities = result["probabilities"]
    assert set(probabilities) == {"setosa", "versicolor", "virginica"}
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert max(probabilities, key=probabilities.get) == species
    assert probabilities == second.json()["probabilities"]
    assert first.headers["x-request-id"] == result["request_id"]
    UUID(result["request_id"])
    assert result["request_id"] != second.json()["request_id"]


def test_worker_health_reports_loaded_model(worker_client):
    response = worker_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "service": "worker", "ready": True,
        "worker_id": "test-cpu-worker", "model": "iris-logistic-regression-v1",
    }


INVALID_PAYLOADS = [
    {}, {"features": []}, {"features": [1, 2, 3]},
    {"features": [1, 2, 3, 4, 5]}, {"features": None},
    {"features": [0, 2, 3, 4]}, {"features": [-1, 2, 3, 4]},
    {"features": [True, 2, 3, 4]}, {"features": ["5.1", 2, 3, 4]},
    {"features": [None, 2, 3, 4]}, {"features": [[1], 2, 3, 4]},
    {"features": [float("nan"), 2, 3, 4]},
    {"features": [float("inf"), 2, 3, 4]},
    {"features": [float("-inf"), 2, 3, 4]},
    {"features": [100.01, 2, 3, 4]},
    {"features": [1e308, 2, 3, 4]},
    {"features": [1, 2, 3, 4], "unexpected": "value"},
    [1, 2, 3, 4],
]


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_worker_rejects_invalid_features(worker_client, payload):
    # Raw encoding intentionally exercises nonstandard NaN/Infinity tokens too.
    response = worker_client.post(
        "/predict", content=json.dumps(payload),
        headers={"Content-Type": "application/json", "X-Request-ID": "invalid-sample"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    assert response.headers["x-request-id"] == "invalid-sample"


def test_worker_rejects_malformed_json(worker_client):
    assert worker_client.post(
        "/predict", content='{"features":', headers={"Content-Type": "application/json"}
    ).status_code == 422


def test_worker_accepts_measurement_upper_bound(worker_client):
    response = worker_client.post("/predict", json={"features": [100, 2, 3, 4]})
    assert response.status_code == 200
    assert sum(response.json()["probabilities"].values()) == pytest.approx(1.0)


def test_worker_reuses_model_and_cleans_up_lifespan():
    application = create_app(WorkerSettings(worker_id="lifecycle-worker", gateway_url=None))
    with TestClient(application) as client:
        model = application.state.model
        assert model is not None
        for _ in range(2):
            assert client.post("/predict", json={"features": [5, 3, 1, 1]}).status_code == 200
            assert application.state.model is model
    assert application.state.model is None


@pytest.mark.parametrize("request_id", ["trace-123", "bad id", "x" * 200])
def test_worker_propagates_sane_request_ids(worker_client, request_id):
    response = worker_client.post(
        "/predict", json={"features": [5.1, 3.5, 1.4, 0.2]},
        headers={"X-Request-ID": request_id},
    )
    returned_id = response.json()["request_id"]
    assert response.headers["x-request-id"] == returned_id
    if request_id == "trace-123":
        assert returned_id == request_id
    else:
        assert returned_id != request_id
        UUID(returned_id)
