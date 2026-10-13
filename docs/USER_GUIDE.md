# Local inference user guide

This prototype demonstrates a complete ML request across two Python processes on one Windows machine. A FastAPI gateway receives the browser or API request, validates it, and forwards it over HTTP to a CPU worker. The worker predicts an Iris flower species and returns model probabilities. The gateway also serves a small matte dark and purple interface.

The classifier uses scikit-learn's bundled Iris dataset and logistic regression. The worker trains the small model once during startup, using all bundled samples, with `max_iter=200` and `random_state=42`. There is no model download or saved model file. This is an inference plumbing demonstration: it does not report held-out accuracy, prove generalization, or identify arbitrary plants. Model confidence means the probability assigned to the predicted class, not measured accuracy.

## Study basis

All five local references informed the implementation. Their examples are learning material; the API contract and commands below describe this repository's actual design.

| Reference | Practice adopted in this prototype |
| --- | --- |
| [Volume 1: Foundations](Base/FastAPI_Prep_Vol1_Foundations.md) | Separate client, gateway, and worker roles; use explicit ports, JSON bodies, GET for health, POST for prediction, and HTTP status codes for failures. Browser checks supplement server validation. |
| [Volume 2: FastAPI Core](Base/FastAPI_Prep_Vol2_FastAPI_Core.md) | Declare separate Pydantic request and response contracts, constrain inputs, and expose generated OpenAPI documentation. |
| [Volume 3: The Engine Room](Base/FastAPI_Prep_Vol3_Engine_Room.md) | Use Uvicorn as the ASGI server, initialize resources in lifespan, await HTTP forwarding, and keep synchronous model work off the event loop. Serve the UI from the gateway so browser calls use the same origin. |
| [Volume 4: Real Applications](Base/FastAPI_Prep_Vol4_Real_Applications.md) | Keep the architecture small, read validated environment configuration, distinguish gateway readiness from worker availability, and exercise API validation and upstream failure behavior in tests. |
| [Volume 5: Mastery and Interview](Base/FastAPI_Prep_Vol5_Mastery_Interview.md) | Build the health → schema → model → forwarding flow incrementally; initialize the model once per process and correlate requests using request IDs and logs. |

The references also cover databases, JWT authentication, containers, queues, and larger deployment systems. Those features are deferred for this local prototype. The course project architecture in [CLAUDE.md](../CLAUDE.md) takes precedence over those larger tutorial examples. The explicitly requested UI is a small extension to the otherwise deferred frontend.

## Set up on Windows

Install Python 3.11, open PowerShell at the repository root, and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-prototype.txt
```

The prototype dependency file includes FastAPI, Uvicorn, HTTPX, Pydantic settings, scikit-learn, pytest, and pytest-asyncio. It avoids the larger unused dependencies in the original lock file. An existing Python 3.11 environment with these dependencies also works; substitute its Python executable in the commands below. Calling the executable directly avoids requiring environment activation.

Optionally copy the safe configuration template:

```powershell
Copy-Item .env.example .env
```

Skip this copy if you already have a `.env`; edit the existing file as needed. Defaults run both services locally without a `.env`.

## Start in two terminals

Both terminals must use the repository root as their working directory. Start the worker first in terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.worker.main:app --host 127.0.0.1 --port 8001
```

Wait for `Application startup complete`. Start the gateway in terminal 2:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.coordinator.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. Keep both terminals running. These commands bind to the local machine only; another computer cannot access this setup through its own `localhost`.

For a single-terminal launch, use the convenience script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1
```

The script starts the worker and gateway in hidden child processes, waits for readiness, prints the UI URL, and monitors the processes. Press Ctrl+C in the launch terminal to stop both children. Startup diagnostics are in `worker.log` and `gateway.log` in the repository root; the launcher overwrites these ignored logs on each run. It uses `.venv\Scripts\python.exe` when available, otherwise `python` on PATH. You can select another environment or ports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1 -Python "D:\PythonEnvs\distributed-ml-platform\Scripts\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1 -GatewayPort 8010 -WorkerPort 8011
```

The launcher sets `WORKER_URL` to the selected worker port for the child processes. Open the printed URL if using a different gateway port. The two-terminal method exposes service logs directly and is useful when investigating a problem.

## Use the interface

1. Check that the gateway is **Online** and the CPU worker is **Available**. Status refreshes every 15 seconds while the tab is visible; **Refresh status** checks immediately.
2. Choose a Setosa, Versicolor, or Virginica sample, or enter your own four measurements in centimeters. Every value must be finite, greater than zero, and at most 100 cm.
3. Select **Run inference**, or submit the form with Enter. The page shows a loading message while the request runs.
4. Read the predicted species, confidence, and probability for each class. Model, worker, and request identifiers appear below the result for traceability.

A preset only fills the inputs; it does not supply a prediction. Editing an input clears the previous result so it cannot be mistaken for a prediction of the new values. The page uses native labels, keyboard focus indicators, and a live status region. It loads no external fonts, scripts, or CSS.

## Use the API

Gateway interactive documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), with its machine-readable schema at `/openapi.json`. Worker documentation is at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs). FastAPI's default interactive documentation may require internet access to load its Swagger assets; the local inference UI and PowerShell requests work without those assets.

Run these from a third PowerShell terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8001/health

$payload = @{ features = @(5.1, 3.5, 1.4, 0.2) } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $payload
```

`POST /predict` expects exactly this JSON shape:

```json
{"features": [5.1, 3.5, 1.4, 0.2]}
```

Feature order is **sepal length, sepal width, petal length, petal width**, all in centimeters. The API rejects missing or extra fields, any other list length, zero or negative values, values above 100 cm, non-finite values, strings, and booleans. The response contains:

| Field | Meaning |
| --- | --- |
| `prediction` | `setosa`, `versicolor`, or `virginica` |
| `class_id` | Corresponding class index: 0, 1, or 2 |
| `probabilities` | Object mapping all three species to probabilities between 0 and 1 |
| `model` | `iris-logistic-regression-v1` |
| `worker_id` | Worker identity; default `cpu-worker-1` |
| `request_id` | Request correlation identifier, also returned in `X-Request-ID` |

Worker `/health` reports its readiness and loaded model. Gateway `/health` reports its own `ready` state and separately probes `worker_available`; it returns HTTP 200 with `status: "degraded"` when the gateway is ready but the worker is unavailable. Check `worker_available`, not HTTP 200 alone, before presenting inference availability.

| HTTP status from prediction | Meaning |
| --- | --- |
| 200 | Successful prediction |
| 422 | Invalid payload; `detail` lists validation errors |
| 503 | Worker unavailable or connection failed |
| 504 | Worker response exceeded the gateway timeout |
| 502 | Worker returned an unexpected status or invalid response |

The page displays errors without inventing a result. Health is a point-in-time check: a worker can become unavailable between a successful probe and a prediction.

## Configuration

Settings load from environment variables or the root `.env`, with environment variables taking precedence. Restart the relevant service after changes.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WORKER_URL` | `http://127.0.0.1:8001` | Gateway's single worker origin; no path, credentials, query, or fragment |
| `REQUEST_TIMEOUT_SECONDS` | `5` | Positive finite total forwarding deadline and HTTP I/O timeout, at most 120 seconds |
| `WORKER_ID` | `cpu-worker-1` | Worker label, 1–128 characters using letters, digits, `_`, `.`, or `-` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |

For example, if manually starting the worker on port 8011, set this in the gateway terminal before starting it:

```powershell
$env:WORKER_URL = "http://127.0.0.1:8011"
.\.venv\Scripts\python.exe -m uvicorn app.coordinator.main:app --host 127.0.0.1 --port 8000
```

Ports and bind addresses are Uvicorn command options. Environment values do not change a running process. The browser has a 30-second request limit, so setting the gateway timeout above 30 seconds can make the browser report a timeout first; API clients can choose their own timeout.

## Stop, restart, and observe failure

For two-terminal startup, press Ctrl+C in each service terminal. Restart with the same commands, worker first. Restarting the worker trains its small in-memory model again; restarting the gateway recreates its HTTP client. There is no persisted prediction history.

To demonstrate a connection failure, stop only the worker while leaving the gateway running. Refresh status, then run inference: the worker becomes unavailable and the gateway returns a clear service error. Start the worker again, refresh status, and retry. This demonstrates error handling and recovery of a single connection; there is no alternate worker to reroute to.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `py -3.11` is not found | Install Python 3.11 with the Windows launcher, or use an installed Python 3.11 executable explicitly. |
| `No module named uvicorn` or another import error | Install `requirements-prototype.txt` with the same Python executable used to launch the services. |
| `Could not import module` | Start from the repository root and use the exact module names above. |
| Port 8000 or 8001 is already in use | Stop the previous process, or select different ports; update gateway `WORKER_URL` for a manual worker port change. |
| Gateway is online, worker unavailable | Check the worker terminal or `worker.log`, its startup completion, and gateway `WORKER_URL`. |
| Worker does not start after changing configuration | Read the startup validation message; check URL format, worker ID, timeout, and log level. |
| 422 response | Send four JSON numbers in the required order, each greater than zero and at most 100; omit extra keys. |
| 504 response | Inspect the worker logs and CPU load. Restart the worker if needed; adjust the configured timeout only after identifying the cause. |
| Page cannot reach the gateway | Keep the gateway running and open its HTTP URL. Do not open `app/static/index.html` directly as a file. |
| `/docs` is blank while the UI works | Default Swagger assets may be blocked or offline. Use PowerShell or `/openapi.json` to inspect the API. |
| Launcher exits or services fail | Read `worker.log` and `gateway.log`; try the two-terminal commands to see startup errors directly. |

## Verification and limitations

Run the automated suite with the same environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_local.py
```

The smoke script starts its own two services on temporary local ports, checks real HTTP inference, stops and restarts the worker to verify failure/recovery, and cleans up its processes. It can run alongside the normal demo. Consult [PROJECT_PLAN.md](PROJECT_PLAN.md) for verified progress and remaining work. Automated API checks and a live HTTP smoke check do not replace checking the page visually in your browser.

The current topology is one local gateway and one local CPU worker. It has no worker registration, heartbeat registry, round-robin routing, retries, rerouting, GPU path, second-machine verification, Lamport clocks, leader election, model replication, authentication, database, prediction history, or performance benchmark. It uses plain HTTP on loopback for this learning demo. Model probabilities and request timing are not a distributed reliability or ML evaluation result. The next step is to add a worker registry and a second compatible CPU worker, then verify health-based routing before extending to a second machine.
