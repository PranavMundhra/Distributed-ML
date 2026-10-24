# Distributed ML Inference Platform

A Python 3.11+ learning project for distributed ML inference across Windows machines. One FastAPI gateway and two registered CPU workers run as separate local processes. The interface serves three fixed models: Iris classification, house-price regression, and male/female face-image classification. The gateway uses round-robin among workers advertising the requested model. Heartbeats detect silent workers; Lamport clocks show the actual request path and its logical timestamps.

Each worker trains the small Iris classifier at startup and loads local replicas of the two prepared ONNX models once. Iris works without downloaded artifacts; prepare the additional models below for the full demonstration. No GPU, database, container, or frontend package manager is needed. The face classifier reuses pretrained weights; house-price holdout metrics and model limitations are in the model guide. Distributed performance benchmarks remain pending.

## Quick start

Run from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-prototype.txt
```

For all three models, prepare the artifacts once (this downloads the attributed source data/model):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe scripts/prepare_demo_models.py
```

Prepared artifacts already exist on this PC. For another PC, copy `models/demo` or prepare it there. Each worker loads its own local copy. Missing or invalid artifacts leave that model unavailable while Iris keeps working; restart workers after preparing files.

Start the gateway and two workers together:

```powershell
.\.venv\Scripts\python.exe scripts/run_local.py
```

The gateway uses port 8000. Workers `cpu-worker-1` and `cpu-worker-2` use ports 8001 and 8002. They register automatically. Press Ctrl+C to stop all three processes.

Verify six alternating requests in another terminal:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_round_robin.py --gateway-url http://127.0.0.1:8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), choose a model, load a sample or select a cropped face image, and select **Run inference**. Repeated predictions for each model display alternating worker IDs and a six-event Lamport trace. The readable [worker dashboard](http://127.0.0.1:8000/workers) shows addresses, models, heartbeat deadlines, completed prediction counts and recent traces. Model availability is at [/models](http://127.0.0.1:8000/models), and API documentation at [/docs](http://127.0.0.1:8000/docs).

For a presentation, use the dashboard's **Send 6 Iris requests** button, pause one worker, wait for heartbeat expiry, send another batch, then resume it. The local launcher enables these controls using a fresh internal secret shared only with its children. Pause stops that worker's heartbeats and rejects new inference while leaving the process running for resume. Gateway scheduling is still controlled by real heartbeat expiry or observed request failures. Actual process crashes can be demonstrated using manual worker terminals or `scripts/smoke_workers.py`.

Alternatively, use `powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1`. The launcher supports `--gateway-port`, `--worker-port` (first worker port), and `--worker-count` (default 2); the PowerShell wrapper supports `-Python`, `-GatewayPort`, `-WorkerPort`, and `-WorkerCount`.

## Two-PC LAN setup

Use separate terminals so either worker can be stopped independently. The following addresses are examples: replace them with each PC's connected-adapter IPv4 address from `ipconfig`.

PC 1, gateway terminal:

```bat
.venv\Scripts\python.exe scripts\run_lan.py gateway --ip 192.168.1.10
```

PC 1, Worker 1 terminal:

```bat
.venv\Scripts\python.exe scripts\run_lan.py worker --ip 192.168.1.10 --gateway-ip 192.168.1.10 --worker-id cpu-worker-1
```

PC 2, Worker 2 terminal:

```bat
.venv\Scripts\python.exe scripts\run_lan.py worker --ip 192.168.1.11 --gateway-ip 192.168.1.10 --worker-id cpu-worker-2
```

Prepare/copy the fixed model artifacts and install a fresh environment on PC 2 first. On the existing development PC, use `..\.verify-venv\Scripts\python.exe` if needed. Open `http://192.168.1.10:8000/workers`. The LAN launcher disables local pause/resume controls; stop Worker 2 with Ctrl+C in its own terminal for a real failure demonstration.

From another PC 1 terminal, check normal routing, then run the second command after stopping Worker 2:

```bat
.venv\Scripts\python.exe scripts\check_lan.py --gateway-ip 192.168.1.10
.venv\Scripts\python.exe scripts\check_lan.py --gateway-ip 192.168.1.10 --failed-worker cpu-worker-2
```

The checker requires distinct LAN worker IPs, advancing heartbeats, all three models and valid Lamport traces. Each check sends 24 predictions. Restart Worker 2 and rerun the normal check to demonstrate recovery. See the [complete LAN guide](docs/LAN_DEMO.md) for model transfer, firewall access, port options and troubleshooting. The tooling passed a one-PC private-interface rehearsal; physical two-PC verification remains pending.

## API and checks

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/workers
$payload = @{ features = @(5.1, 3.5, 1.4, 0.2) } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $payload
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_local.py
.\.venv\Scripts\python.exe scripts/smoke_round_robin.py
.\.venv\Scripts\python.exe scripts/smoke_heartbeats.py
.\.venv\Scripts\python.exe scripts/smoke_lamport.py
.\.venv\Scripts\python.exe scripts/smoke_models.py
.\.venv\Scripts\python.exe scripts/smoke_workers.py
```

Feature order is sepal length, sepal width, petal length, petal width in centimeters, each greater than zero and at most 100. The response includes species, class ID, probabilities, model name, worker ID, and request ID. Gateway health reports its readiness separately from worker availability. Connection failures return 503, timeouts 504, and invalid worker responses 502. Prediction JSON is unchanged; `X-Trace-ID` identifies a trace at `GET /traces/{trace_id}`. The UI displays six real message events for an updated worker. Logical timestamps describe ordering, not milliseconds or physical clock synchronization.

## Documentation and next steps

- [Prepared demo models](docs/DEMO_MODELS.md): locally trained house-price regression and a pretrained male/female face classifier, both in ONNX, with examples, provenance and validation. All three fixed models are connected to the interface and compatible workers. Public model uploads remain deferred.
- [User guide](docs/USER_GUIDE.md): setup, UI/API usage, configuration, stopping and restarting, troubleshooting, and the five study references.
- [Two-PC LAN demonstration](docs/LAN_DEMO.md): separate gateway/worker launchers, distinct-address checks and actual stop/restart demonstration.
- [Project plan](docs/PROJECT_PLAN.md): verified progress and remaining work.
- [Architecture guidance](CLAUDE.md): intended minimal distributed demonstration.

Workers register at `POST /workers/register` with their ID, service origin, loaded `models`, and a per-process instance ID. The original `model` field remains Iris for compatibility; registrations without `models` advertise Iris only. Duplicate registration does not add scheduling slots; conflicting IDs or addresses return 409. Changing capabilities requires a new worker instance. `POST /workers/heartbeat` accepts increasing heartbeat sequence numbers for the current instance. `GET /workers` includes availability, loaded models, last heartbeat receipt time (UTC), sequence, and expiry status. `WORKER_URL` is the worker's advertised address; `GATEWAY_URL` enables registration and heartbeats. Start the gateway before workers. Manual commands and configuration are in the user guide.

By default, workers send heartbeats every 2 seconds. After 8 seconds without registration or an accepted heartbeat, the gateway marks a worker unavailable; its expiry task checks every 0.5 seconds. Health probes and predictions cannot renew an expired heartbeat lease. A new valid heartbeat or worker registration restores eligibility. The launcher keeps the gateway and surviving workers running if one worker exits.

Run one gateway Uvicorn process: its registry and per-model round-robin cursors are in memory. Workers automatically re-register when a restarted gateway reports them unknown. A failed prediction is still returned without retry; new requests skip unavailable workers. Bounded retry/failover for an in-flight prediction, a verified two-PC run, optional GPU execution, and benchmarks remain future work. Bully election and coordinator takeover are outside the agreed project scope; the gateway remains a single point of failure. Heartbeat expiry indicates suspected unavailability, not proof of permanent machine failure.

The latest checks passed 361 tests and `scripts/smoke_lan.py --ip <this-PC-private-IPv4>`: LAN-interface launchers, same-IP rejection by the default checker, disabled inherited demo controls, all-model routing/traces, actual worker stop/restart and gateway restart recovery. This was a one-PC rehearsal. Earlier `scripts/smoke_workers.py` and browser verification covered dashboard counters/traces, demo pause, heartbeat expiry, six survivor requests and recovery to 3/3 without browser warnings/errors. Previous model and heartbeat checks are recorded in the project plan. Windows Application Control blocked native libraries in the original Python 3.11 environment on this PC. Verification used a separate Python 3.12.14 environment at `D:\WORK\CODING\PYTHON\DC\.verify-venv`, with scikit-learn 1.7.2 (within the supported range); the original environment and system security settings were preserved. See the user guide for the working launch command on this PC.

`requirements-prototype.txt` contains the dependencies needed for this version; the original broader `requirements-lock.txt` remains available. Configuration uses safe defaults and optional root `.env` values documented in `.env.example`. Keep environments, logs, and model binaries out of Git.
