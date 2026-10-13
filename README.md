# Distributed ML Inference Platform

A Python 3.11 learning project for distributed ML inference across Windows machines. The current prototype runs one FastAPI gateway and one CPU worker as separate processes on one machine. The gateway forwards validated Iris measurements over HTTP and serves a minimal matte dark and purple inference page.

The worker trains a small scikit-learn logistic regression classifier from the bundled Iris dataset at startup. No GPU, downloaded model, database, container, or frontend package manager is needed. This demonstrates the request path and failure handling; held-out ML accuracy and performance benchmarks are not yet measured.

## Quick start

Run from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-prototype.txt
```

Start the worker in terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.worker.main:app --host 127.0.0.1 --port 8001
```

Start the gateway in terminal 2:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.coordinator.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), choose a sample, and select **Run inference**. API documentation is at [/docs](http://127.0.0.1:8000/docs). Press Ctrl+C in each terminal to stop.

Alternatively, start both services with `powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1` and press Ctrl+C in that terminal to stop both. The script supports `-Python`, `-GatewayPort`, and `-WorkerPort`; see the guide for details.

## API and checks

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
$payload = @{ features = @(5.1, 3.5, 1.4, 0.2) } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $payload
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_local.py
```

Feature order is sepal length, sepal width, petal length, petal width in centimeters, each greater than zero and at most 100. The response includes species, class ID, probabilities, model name, worker ID, and request ID. Gateway health reports its readiness separately from worker availability. Connection failures return 503, timeouts 504, and invalid worker responses 502.

## Documentation and next steps

- [User guide](docs/USER_GUIDE.md): setup, UI/API usage, configuration, stopping and restarting, troubleshooting, and the five study references.
- [Project plan](docs/PROJECT_PLAN.md): verified progress and remaining work.
- [Architecture guidance](CLAUDE.md): intended minimal distributed demonstration.

The current implementation has one configured worker, without registration, heartbeats, load balancing, or rerouting. Add and verify those incrementally before a second-machine demonstration. GPU execution, model replication, Lamport clocks, leader election, and benchmarks remain future work.

`requirements-prototype.txt` contains the dependencies needed for this version; the original broader `requirements-lock.txt` remains available. Configuration uses safe defaults and optional root `.env` values documented in `.env.example`. Keep environments, logs, and model binaries out of Git.
