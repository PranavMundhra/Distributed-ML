# Local inference user guide

The prototype runs one FastAPI gateway and two CPU workers as separate local processes. Workers register their ID, address, loaded models, and per-process instance ID. They send periodic heartbeats so the gateway can detect silence without prediction traffic. The gateway validates predictions, selects compatible workers using a separate round-robin cursor per model, and serves the three-model inference interface.

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

The prototype dependency file includes FastAPI, Uvicorn, HTTPX, Pydantic settings, scikit-learn, ONNX Runtime, Pillow, pytest, and pytest-asyncio. ONNX Runtime serves the two prepared models on CPU; Pillow validates and resizes face images. An existing Python 3.11+ environment with these dependencies also works; substitute its Python executable in the commands below. Calling the executable directly avoids requiring environment activation.

For the full three-model demonstration, prepare the two ONNX artifacts once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.\.venv\Scripts\python.exe scripts/prepare_demo_models.py
```

The artifacts already exist on this PC. For a teammate's machine, copy `models/demo` or prepare them there. Every worker loads a complete local replica. Missing or invalid artifacts leave that model unavailable; Iris remains usable. Restart workers after adding files. No automatic model distribution is implemented.

Optionally copy the safe configuration template:

```powershell
Copy-Item .env.example .env
```

Skip this copy if you already have a `.env`; edit the existing file as needed. Defaults run both services locally without a `.env`.

## Start the local demonstration

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/run_local.py
```

The launcher starts the gateway on 8000 first, then `cpu-worker-1` on 8001 and `cpu-worker-2` on 8002. Workers register during startup. It checks readiness and monitors its child processes. Press Ctrl+C to stop all three. Logs are `gateway.log`, `worker-1.log`, and `worker-2.log`; they are ignored and overwritten on each launch.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Repeat predictions and observe the Worker field alternate. View the registry at [http://127.0.0.1:8000/workers](http://127.0.0.1:8000/workers). From a second terminal, print six requests and verify their worker IDs:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_round_robin.py --gateway-url http://127.0.0.1:8000
```

With a fresh gateway, the sequence is `cpu-worker-1, cpu-worker-2` repeated three times. An existing gateway may start with either worker; the checker accepts either alternating sequence. Avoid sending other predictions during this check. Health and listing requests do not consume scheduling turns.

The PowerShell wrapper also works:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1
```

It uses `.venv\Scripts\python.exe` when available, otherwise `python` on PATH. Select another environment or ports as needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1 -Python "D:\PythonEnvs\distributed-ml-platform\Scripts\python.exe"
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1 -GatewayPort 8010 -WorkerPort 8011
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1 -WorkerCount 1
```

`WorkerPort` is the first worker port; later workers use consecutive ports. The Python equivalent accepts `--gateway-port`, `--worker-port`, and `--worker-count` (1–8, default 2). The launcher sets unique worker IDs, advertised `WORKER_URL` values, and `GATEWAY_URL` for each worker.

## Manual startup in separate terminals

All terminals must use the repository root. Start the gateway first, with one Uvicorn process:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.coordinator.main:app --host 127.0.0.1 --port 8000
```

In terminal 2:

```powershell
$env:GATEWAY_URL = "http://127.0.0.1:8000"
$env:WORKER_ID = "cpu-worker-1"
$env:WORKER_URL = "http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m uvicorn app.worker.main:app --host 127.0.0.1 --port 8001
```

In terminal 3:

```powershell
$env:GATEWAY_URL = "http://127.0.0.1:8000"
$env:WORKER_ID = "cpu-worker-2"
$env:WORKER_URL = "http://127.0.0.1:8002"
.\.venv\Scripts\python.exe -m uvicorn app.worker.main:app --host 127.0.0.1 --port 8002
```

`WORKER_URL` must match the address reachable by the gateway. These commands use loopback, so they do not verify communication between PCs. Initial registration is attempted after training. An unreachable gateway, rejected/invalid registration, or unsafe heartbeat settings prevent startup; inspect the terminal and correct settings. An empty or unset `GATEWAY_URL` runs a standalone worker without registration or heartbeats. Manually registered workers must send heartbeats to remain eligible beyond the initial grace period.

## Use the interface

1. Check that the gateway is **Online** and Workers shows **2/2 available**. Status refreshes every 15 seconds while the tab is visible; **Refresh status** checks immediately.
2. Select Iris, House price, or Male/female image classification. Iris accepts four flower measurements and species samples; House shows six property inputs and an example; Image accepts a PNG/JPEG face crop or either prepared portrait example. Only the selected model's inputs are enabled.
3. Select **Run inference**, or submit the form with Enter. The page shows a loading message while the request runs.
4. Read the class probabilities for Iris/Image, or historical USD estimate for House. Model, worker, and request identifiers appear below the result. Scroll down to see the six-event Lamport trace.

A preset only fills the inputs; it does not supply a prediction. Editing an input clears the previous result so it cannot be mistaken for a prediction of the new values. The page uses native labels, keyboard focus indicators, and a live status region. It loads no external fonts, scripts, or CSS.

## Worker dashboard and failure demonstration

Restart the local launcher after updating the code, then open [http://127.0.0.1:8000/workers](http://127.0.0.1:8000/workers), or select **Worker nodes** from the inference page. The browser receives a readable page. API clients still receive JSON from `/workers`; `/workers?format=json` forces JSON in the browser, and `/workers/view` explicitly opens the page.

Each worker card shows its advertised address, loaded models, availability, heartbeat sequence/receipt time, remaining heartbeat deadline, process identity and validated gateway prediction count. The last 20 completed gateway requests show worker identity and trace links. Counts are not heartbeat counts and do not include direct worker requests. A new worker process starts at zero; gateway restart clears counters and history. Refreshing the page does not reset them.

Present the following sequence with the normal defaults:

1. Start with both workers available. Select **Send 6 Iris requests**. With no other predictions, each worker handles three requests.
2. Select **Pause cpu-worker-1 (demo)**. It stops sending new heartbeats and returns 503 for new inference/health requests. Its process and control endpoint stay running.
3. Watch **Waiting for detection** and the heartbeat countdown. After the 8-second silence limit and next expiry check, the card becomes **Unavailable**. The countdown measures time since the gateway's last accepted registration/heartbeat, so it may begin below eight seconds.
4. Send another batch. All six new requests should use Worker 2; its prediction counter increases while Worker 1's stays unchanged.
5. Select **Resume cpu-worker-1**. The next genuine heartbeat (normally within two seconds) restores eligibility. A new batch alternates again. Resume keeps the same process identity.

The page polls `/workers/status` once per second without health probes, scheduling turns or heartbeat renewal. Pause/resume records an acknowledged worker state but never edits gateway availability/deadlines directly. Keep other inference tabs idle while demonstrating heartbeat-only detection: their gateway health checks or prediction attempts can detect a paused worker earlier. Requests already executing may finish; a failed prediction is not retried.

The local launcher generates a fresh shared `DEMO_CONTROL_TOKEN` for its child services; it is not sent to the browser, stored in Git or shown in status responses. Gateway controls are restricted to loopback clients and reject cross-origin browser requests. Worker controls require that internal secret and the current instance ID. Manual startup has controls disabled by default, but observation and request batches still work. If a control response is uncertain, retry Pause to confirm the worker's state, then Resume. A genuinely stopped process must be started again from its terminal; Resume cannot relaunch a process.

For a genuine crash demonstration, use the separate worker terminals described above: stop just one worker with Ctrl+C, leave the gateway and other worker running, wait for expiry, and send another batch. Restart that worker with the original command; the dashboard shows a new process identity and zero counter. Ctrl+C in the combined launcher stops all its children, so use the per-worker terminals for a manual process-stop demonstration.

The automated real-process demonstration starts temporary services, checks simulated pause and actual process stop/restart, then cleans up:

```bat
..\.verify-venv\Scripts\python.exe scripts\smoke_workers.py
```

It also checks all-model survivor routing and gateway restart recovery. These are still local process demonstrations; the two-PC LAN test remains pending.

## Use the API

Gateway interactive documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), with its machine-readable schema at `/openapi.json`. Worker documentation is at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs). FastAPI's default interactive documentation may require internet access to load its Swagger assets; the local inference UI and PowerShell requests work without those assets.

Run these from a third PowerShell terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/workers
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

Worker `/health` reports its readiness and loaded model. Gateway `/health` probes all registered workers concurrently and reports `registered_workers`, `available_workers`, and `worker_available` (at least one ready compatible worker). It returns HTTP 200 with `status: "degraded"` if no worker is ready. `worker_id` and `worker_url` describe the first available worker for compatibility; use `/workers` for the complete list. Check dependency fields, not HTTP 200 alone.

`POST /workers/register` accepts `{"worker_id":"cpu-worker-1","worker_url":"http://127.0.0.1:8001","model":"iris-logistic-regression-v1","instance_id":"manual-process-1"}`. Automatic workers generate a UUID instance ID for each process lifetime. A new registration returns 201; repeated registration returns 200 without adding a scheduling slot. A reused ID with a different address, or address with a different ID, returns 409. Invalid IDs, origins, extra fields, and incompatible models return 422. A new instance at the same worker ID/address replaces the old incarnation and resets its heartbeat sequence. Re-registering the same instance preserves sequence; omitting `instance_id` on a repeat preserves the registered instance.

`GET /workers` returns worker ID, URL, model, `instance_id`, and `available`, plus `last_heartbeat_at` (gateway receipt time in UTC), `heartbeat_sequence`, `heartbeat_expired`, and `heartbeat_timeout_seconds`. Availability is `null` after registration, `true` after a successful observation/heartbeat, and `false` after an observed failure or heartbeat expiry. Registration grants an initial grace period but does not prove reachability. Round-robin uses insertion order and skips unavailable workers. Health, heartbeat, and listing requests do not consume turns; concurrent prediction completion order may differ from selection order.

`POST /workers/heartbeat` accepts `{"worker_id":"cpu-worker-1","instance_id":"manual-process-1","sequence":1}` and returns the updated worker. Sequences must be strictly increasing positive integers. Unknown workers return 404; an old instance or repeated/out-of-order sequence returns 409; malformed payloads return 422. Client timestamps are not accepted. Rejected heartbeats do not extend the deadline.

## House and face inference

The model selector displays the compatible worker count from `GET /models`. The gateway routes each model only to workers advertising it in their `models` list. Omitted capability lists mean Iris only for compatibility with older registrations. Changing a loaded model list requires restarting the worker with a new process identity. Worker `/health` retains its original Iris contract; `/workers` and `/models` describe additional model availability. Registration grants a grace period, so initial eligibility is not proof of reachability.

`POST /predict/house` accepts named inputs:

```json
{"area_sqft":1800,"bedrooms":3,"full_bathrooms":2,"garage_spaces":2,"year_built":2000,"overall_quality":7}
```

Bounds match the individual training ranges: living area 334–5,642 sq ft; bedrooms 0–8; full bathrooms 0–4; garage spaces 0–4; year 1872–2010; quality 1–10. Counts, year and quality must be integers. The example predicts about **US$232,247.41**. The response includes `predicted_price_usd`, `currency`, `market`, `model`, `worker_id`, and `request_id`. This is historical Ames, Iowa housing data from 2006–2010, not current Indian prices. There is no classification confidence for a regression result. Feature bounds do not guarantee realistic combinations; linear regression can still give implausible or negative estimates.

`POST /predict/face` accepts `{"image_base64":"<raw base64 PNG/JPEG bytes>"}` without a data-URL prefix. The UI handles conversion; API clients can base64-encode an image file. Validation rejects invalid base64, unreadable images, animation, unsupported formats, images below 32×32, images above 20 megapixels, and files larger than 5 MiB. The streamed JSON body is capped at 7 MiB (HTTP 413). Invalid input returns 422 before worker selection. Images stay in memory; file contents are not logged or stored in traces.

The face response includes `prediction` (`female` or `male`), both `probabilities`, `model`, `worker_id`, and `request_id`. Use a single upright cropped face. There is no face detector or no-face rejection. This reuses the pretrained Cydral model's binary appearance labels and cannot establish identity, gender identity or biological sex. Two portrait examples are demonstration checks, not an accuracy benchmark. See [DEMO_MODELS.md](DEMO_MODELS.md) for provenance and measured limitations.

Both new endpoints preserve request IDs and the same six-event Lamport trace used by Iris. The UI clears old results on model/input changes. The image upload is inference input; public uploading of ML model files remains deferred.

Run a self-contained check with temporary gateway/worker ports:

```bat
..\.verify-venv\Scripts\python.exe scripts\smoke_models.py
```

It verifies all models on both workers, traces and invalid input, then stops Worker 1 and confirms every model still works after heartbeat expiry. `--gateway-url http://127.0.0.1:8000` checks an existing two-worker demonstration without stopping services. The normal project `.venv` can be used instead on machines with a working runtime.

## Heartbeats and failure detection

Configured workers reuse their HTTP client to send heartbeats every 2 seconds by default. After 8 seconds without registration or an accepted heartbeat, the gateway marks a worker expired and unavailable. A background task sweeps every 0.5 seconds; selection and listing also check expiry. Default detection is normally within 8–8.5 seconds of the last accepted contact, with additional scheduling/load delay possible. The gateway uses its own monotonic clock for elapsed time; UTC fields are display metadata.

Successful HTTP health probes and predictions do not renew heartbeat deadlines or revive an expired lease. A valid newer heartbeat from the current instance can restore availability. This lets the project detect a broken heartbeat path even when direct inference is still responding. Expiry indicates suspected unavailability, not proof that a machine has permanently failed.

Heartbeat attempts have finite deadlines and wait for the configured interval between attempts. Network failures, server errors, and malformed acknowledgements are logged and retried on later intervals. If a heartbeat receives 404, the live worker re-registers automatically, which supports gateway restart recovery. Other 4xx responses stop that worker's heartbeat task to avoid reclaiming a newer process's registration; inspect the error and restart the intended worker. The local model remains independent of gateway connectivity. Shutdown cancels the heartbeat task before closing its client.

Run the repeatable demonstration (it uses its own temporary ports and stops only its own processes):

```powershell
.\.venv\Scripts\python.exe scripts/smoke_heartbeats.py
```

It verifies both workers' heartbeats, stops Worker 1, observes background expiry without health/prediction traffic, sends new predictions through Worker 2, restarts Worker 1, rejects its old instance's heartbeat, and restarts the gateway to verify automatic re-registration. It uses a documented 0.15-second interval, 1.5-second timeout, and 0.05-second sweep to run faster than the normal demo.

| HTTP status from prediction | Meaning |
| --- | --- |
| 200 | Successful prediction |
| 422 | Invalid payload; `detail` lists validation errors |
| 503 | Worker unavailable or connection failed |
| 504 | Worker response exceeded the gateway timeout |
| 502 | Worker returned an unexpected status or invalid response |

The page displays errors without inventing a result. Health is a point-in-time check: a worker can become unavailable between a successful probe and a prediction.

## Lamport clocks and the request trace

After a prediction, the page displays **How this request travelled**. For an
updated worker, the six recorded events are gateway receive, gateway forward,
worker receive, worker response, gateway receive of that response, and gateway
response to the browser. Each event includes its actual logical timestamp.

Each service process has one shared counter, including registration, heartbeat,
health and prediction messages. Sending increments the counter. Receiving a
timestamp updates it to `max(local, received) + 1`. Background traffic can create
gaps. These numbers are not elapsed time, synchronized wall-clock time, or proof
that unrelated requests caused one another. Round-robin still schedules inference;
logical clocks do not choose workers or detect failures.

The algorithm follows Leslie Lamport's
[Time, Clocks, and the Ordering of Events in a Distributed System](https://lamport.azurewebsites.net/pubs/time-clocks.pdf).

The prediction JSON stays compatible. HTTP responses carry:

| Header | Meaning |
| --- | --- |
| `X-Lamport-Time` | Sender's logical timestamp at response start |
| `X-Lamport-Received` | Logical timestamp when this HTTP request was received |
| `X-Lamport-Instance` | UUID identifying this service process lifespan |
| `X-Trace-ID` | Unique inference attempt ID, separate from the reusable request ID |

Service-to-service requests carry their send timestamp and process UUID too.
Malformed or repeated incoming clock timestamps return HTTP 400. Updated peers
must return complete, correctly ordered clock headers; invalid peer clock
metadata makes a forwarded prediction return 502. Legacy peers with no clock
headers can still predict, but their missing events are not invented.

`GET /traces/{trace_id}` returns the recorded events, original request ID, HTTP
status and a `complete` flag. The gateway retains only the last 200 completed
attempts in memory. Repeated request IDs do not overwrite distinct traces. An
unavailable worker, invalid input or older worker produces an explicitly partial
trace. Old or unknown trace IDs return 404. The trace contains no input features.

Restarting a process resets its counter and changes its UUID. Worker registration
uses the same new UUID as the worker's clock. Gateway restart clears its traces;
surviving workers merge the new gateway's timestamps and automatically re-register.
Counters are not persisted across process incarnations. Heartbeat timeouts continue
to use monotonic seconds, independently of logical time.

Run the self-contained real-process demonstration:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_lamport.py
```

It verifies six alternating predictions, actual causal timestamps, eight concurrent
HTTP requests, invalid clock rejection, worker restart and gateway restart. With
`--gateway-url http://127.0.0.1:8000`, it checks an existing two-worker demo without
stopping any services. All processes created by the self-contained mode are cleaned up.

### Verified runtime on this PC

Windows Application Control currently blocks native libraries required by the
original `.venv` Python 3.11 environment, including after a sandbox escalation.
System security settings were not changed. A separate verification environment
uses the bundled Python 3.12.14 runtime and scikit-learn 1.7.2; all other packages
were installed from the project's requirements. The original `.venv` is preserved.

Use these commands in **Command Prompt** on this PC:

```bat
cd /d "D:\WORK\CODING\PYTHON\DC\Distributed-ML"
..\.verify-venv\Scripts\python.exe scripts\run_local.py
```

Then open `http://127.0.0.1:8000`, run a prediction and scroll to the trace panel.
To run verification instead of starting the normal demo:

```bat
..\.verify-venv\Scripts\python.exe -m pytest -q --tb=short
..\.verify-venv\Scripts\python.exe scripts\smoke_lamport.py
```

The Python 3.11 path elsewhere in this guide remains the normal setup for machines
with a working Python 3.11 environment; Python 3.12 is also supported. The current
update was verified on Python 3.12, not re-verified on the blocked 3.11 runtime.

## Configuration

Settings load from environment variables or the root `.env`, with environment variables taking precedence. Restart the relevant service after changes.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WORKER_URL` | `http://127.0.0.1:8001` | Worker's advertised origin; no credentials, path, query, or fragment. The gateway no longer reads this as a fixed destination. |
| `GATEWAY_URL` | unset | Worker registration target. Set to `http://127.0.0.1:8000` for the local gateway; empty means standalone. |
| `REQUEST_TIMEOUT_SECONDS` | `5` | Positive finite deadline for forwarding, health checks, and startup registration, at most 120 seconds |
| `HEARTBEAT_INTERVAL_SECONDS` | `2` | Worker interval, positive and at most 60 seconds; must be less than the gateway's timeout |
| `HEARTBEAT_TIMEOUT_SECONDS` | `8` | Gateway silence limit, positive and at most 300 seconds |
| `HEARTBEAT_CHECK_INTERVAL_SECONDS` | `0.5` | Gateway expiry sweep, positive, at most 30 seconds, and no greater than its timeout |
| `WORKER_ID` | `cpu-worker-1` | Worker label, 1–128 characters using letters, digits, `_`, `.`, or `-` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `DEMO_CONTROL_TOKEN` | unset | Internal opt-in control secret; the combined local launcher generates it per run. Manual services leave demo pause/resume disabled by default. |
| `DEMO_MODEL_DIR` | repository `models/demo` | Worker directory containing `house_price/house_price.onnx` and `male_female/male_female.onnx`; loaded once at startup |

For example, if manually starting a worker on port 8011, set its advertised URL in the worker terminal:

```powershell
$env:WORKER_URL = "http://127.0.0.1:8011"
.\.venv\Scripts\python.exe -m uvicorn app.worker.main:app --host 127.0.0.1 --port 8011
```

Ports and bind addresses are Uvicorn command options. Environment values do not change a running process. The browser has a 30-second request limit, so setting the gateway timeout above 30 seconds can make the browser report a timeout first; API clients can choose their own timeout.

## Stop, restart, and observe failure

For manual startup, press Ctrl+C in each service terminal. Start the gateway before registered workers. Restarting a worker trains its model and registers a new process instance at the same address. A gateway restart clears its registry and cursor; running workers encounter heartbeat 404 and automatically register again. Their arrival order may differ after recovery. Run one gateway process, because separate Uvicorn processes do not share this registry. There is no persisted prediction history.

After successful startup, the launcher keeps the gateway and surviving workers running when one worker exits; it reports the exit once. It stops its remaining services on gateway exit, startup failure, or Ctrl+C. Manual terminals and the smoke script are convenient for failure demonstrations. A failed prediction still returns its 503/504/502 response without a retry. Later requests skip workers marked unavailable by either an observed failure or heartbeat expiry. Automatic retry of an already-failed prediction remains future work.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `py -3.11` is not found | Install Python 3.11 with the Windows launcher, or use an installed Python 3.11 executable explicitly. |
| `No module named uvicorn` or another import error | Install `requirements-prototype.txt` with the same Python executable used to launch the services. |
| `Could not import module` | Start from the repository root and use the exact module names above. |
| A gateway or worker port is already in use | Choose different ports; update each manual worker's advertised `WORKER_URL`. |
| Gateway is online, no workers available | Inspect `/workers`, expiry flags, heartbeat receipt times, and worker logs. Check `GATEWAY_URL`, advertised addresses, and heartbeat settings. |
| Worker health responds but heartbeat is expired | Direct HTTP health does not renew a heartbeat lease. Inspect heartbeat errors and instance conflicts; restart the intended worker if its heartbeat task stopped after a 4xx response. |
| Worker does not start after changing configuration | Read the startup validation message; check URL format, worker ID, timeout, and log level. |
| 422 response | Check the selected model's documented input contract; omit extra keys. For face inference, use a valid still PNG/JPEG crop within size limits. |
| House/face has no compatible workers | Prepare/copy the matching artifacts on each worker, inspect model-load warnings, then restart workers. Iris availability alone does not prove other models loaded. |
| 504 response | Inspect the worker logs and CPU load. Restart the worker if needed; adjust the configured timeout only after identifying the cause. |
| Page cannot reach the gateway | Keep the gateway running and open its HTTP URL. Do not open `app/static/index.html` directly as a file. |
| `/docs` is blank while the UI works | Default Swagger assets may be blocked or offline. Use PowerShell or `/openapi.json` to inspect the API. |
| Launcher exits or services fail | Read `gateway.log` and `worker-*.log`; try manual terminals to see startup errors directly. |

## Verification and limitations

Run the automated suite with the same environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_local.py
.\.venv\Scripts\python.exe scripts/smoke_round_robin.py
.\.venv\Scripts\python.exe scripts/smoke_heartbeats.py
.\.venv\Scripts\python.exe scripts/smoke_lamport.py
```

`smoke_local.py` checks single-worker inference and stop/restart recovery. `smoke_round_robin.py` checks two-worker registration, alternating requests, validation, and duplicates. `smoke_heartbeats.py` adds silence detection, recovery, stale-instance rejection, and gateway restart recovery. All use temporary ports, clean up owned processes, and can run alongside the normal demo. Consult [PROJECT_PLAN.md](PROJECT_PLAN.md) for verified progress. Automated checks do not replace visual inspection of the page.

The verified local topology has one gateway, two registered CPU workers, three fixed inference models, compatible per-model round-robin scheduling, periodic heartbeats, timeout-based expiry, worker/gateway recovery, and Lamport timestamps with a visible request trace. Bounded retry of failed predictions, a verified two-PC run, optional GPU execution, Bully election, and benchmarks remain future work. Registry endpoints are unauthenticated and intended for a trusted learning setup. There is no database; traces are bounded in-memory observations. Model probabilities and request timing are not distributed reliability or ML evaluation results. The next course-focused milestone is coordinator election with real takeover, followed by LAN verification and performance measurements.
