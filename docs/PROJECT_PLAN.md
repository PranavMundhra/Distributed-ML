# Project plan

## Setup
- [x] Python 3.11 environment created on D:
- [x] Core dependencies installed using pip
- [x] GitHub repository created and pushed
- [x] Verify environment imports

## Single-machine prototype
- [x] FastAPI worker health endpoint
- [x] Inference endpoint and request schemas
- [x] Gateway-to-worker HTTP communication
- [x] Automated tests

The current version runs one gateway and two registered CPU workers as separate local
processes. Each worker serves Iris and any prepared fixed ONNX models it successfully loads. The gateway distributes compatible predictions using per-model round-robin and serves the inference interface. See
[USER_GUIDE.md](USER_GUIDE.md) for setup, configuration, UI/API usage, and troubleshooting.

Verification (2026-10-13, Python 3.11):
- `python -m pytest -q --tb=short`: 87 tests passed; one dependency deprecation warning.
- `python scripts/smoke_local.py`: real-process health, UI assets, OpenAPI,
  direct and forwarded inference, validation, stopped-worker HTTP 503, and
  recovery after restarting the worker passed. The smoke check stops its own services.
- Reference basis: all five Markdown volumes in `docs/Base` were studied before implementation.
- Headless Edge: live service status, sample/preset predictions, invalid-input
  feedback, and a 390px mobile layout passed without JavaScript exceptions.
  Desktop and mobile screenshots were inspected.

## Milestone 1: registered workers and round-robin

- [x] Explicit worker registration and listing APIs
- [x] In-memory worker registry with idempotent registration and conflict checks
- [x] Automatic startup registration when `GATEWAY_URL` is configured
- [x] Round-robin scheduling across registered eligible workers
- [x] Worker identity and model compatibility validation
- [x] Two-worker launcher and real-process demonstration script
- [x] Sequential and concurrent routing tests, invalid/duplicate registration tests
- [x] Updated Windows setup instructions and configuration template

Verification (2026-10-21, Python 3.11.16):
- `python -m pytest -q --tb=short`: 134 tests passed, one existing Starlette TestClient dependency deprecation warning.
- `python scripts/smoke_local.py`: real single-worker inference, validation, worker stop/restart, and recovery passed.
- `python scripts/smoke_round_robin.py`: real gateway plus two workers registered automatically; six predictions alternated `cpu-worker-1, cpu-worker-2` three times, with valid request IDs and three requests per worker. Empty-registry, duplicate, and invalid-registration checks passed.
- `python scripts/run_local.py`: launched on 8000/8001/8002; the external checker passed both from a fresh cursor and after an extra prediction. Ctrl+C stopped its owned services.
- `node --check app/static/app.js` and `git diff --check`: passed.

At the end of milestone 1, health checks were on demand and heartbeats were not yet implemented. The registry remains local to one gateway process. Two-PC networking and performance benchmarks remain unverified.

## Milestone 2: heartbeats and failure detection

- [x] Periodic worker heartbeats with bounded HTTP deadlines
- [x] Background gateway expiry using monotonic receive time
- [x] Configurable interval, silence timeout, and expiry sweep
- [x] Last heartbeat receipt time, sequence, and expiry metadata in `/workers`
- [x] Reject stale sequences, old process instances, and malformed messages
- [x] Health probes/predictions cannot renew an expired heartbeat lease
- [x] Restore worker eligibility on a valid current heartbeat or registration
- [x] Automatically re-register live workers after a gateway restart
- [x] Cancel background tasks and close owned clients on shutdown
- [x] Keep surviving demo services running when a worker exits
- [x] Real-process heartbeat expiry and recovery demonstration

Verification (2026-10-21, Python 3.11.16):
- `python -m pytest -q --tb=short`: 187 passed, one existing Starlette TestClient dependency deprecation warning.
- `python scripts/smoke_heartbeats.py`: both workers sent periodic heartbeats; background expiry detected a stopped worker without gateway health/prediction traffic; new requests used the survivor; restarting the worker restored round-robin; an old-instance heartbeat returned 409; live workers automatically re-registered after a gateway restart. Owned processes were cleaned up.
- `python scripts/smoke_local.py` and `python scripts/smoke_round_robin.py`: existing inference/recovery and six-request alternation checks passed.
- Tests also cover exact timeout boundaries using an injected clock, stale/out-of-order messages, malformed payloads, invalid settings, HTTP probes not renewing leases, task cancellation/client cleanup, and launcher behavior when a worker exits.
- `node --check app/static/app.js` and `git diff --check`: passed.

The heartbeat smoke demo uses temporary ports and faster test-only settings (0.15-second interval, 1.5-second timeout, 0.05-second sweep). Normal defaults remain a 2-second interval, 8-second timeout, and 0.5-second sweep. No failed prediction is retried yet; suspecting a worker from missed heartbeats is not proof of permanent machine failure. No new dependencies were added.

## Models prepared for the upload feature

- [x] Train a six-feature house-price linear regression using real Ames sales data
- [x] Evaluate on a separate 586-row test set and compare the exported ONNX model
- [x] Prepare the CC0 pretrained Cydral male/female image classifier in ONNX
- [x] Embed input contracts, preprocessing, labels, and provenance in both models
- [x] Include sample inputs, measured outputs, reproducible preparation, and local inference commands
- [x] Connect the three fixed models to the interface and both workers (public uploads deferred)

Verification (2026-10-22, Python 3.11.16):
- `python scripts/prepare_demo_models.py --offline`: both self-contained ONNX artifacts passed validation and CPU inference. The house model's ONNX predictions differ from scikit-learn by at most US$0.03125; image normalization preserves upstream probabilities.
- House holdout: 2,344 training rows / 586 test rows; MAE US$27,385.01, RMSE US$42,175.11, R² 0.7781. These are historical Ames, Iowa prices in USD.
- Image model: reused pretrained weights, not trained locally; two public-domain portrait smoke checks passed. No independent image accuracy benchmark was performed.
- Model checks: 27 passed as part of the full test suite.
- `python -m pytest -q --tb=short`: 214 passed, one existing Starlette TestClient deprecation warning.

At preparation time, serving was Iris-only. Milestone 4 connects the prepared models. ONNX Runtime and Pillow are now serving dependencies; `requirements-models.txt` adds preparation tools. See [DEMO_MODELS.md](DEMO_MODELS.md). Generated artifacts and cached data remain ignored by Git. No commits were made.

## Milestone 3: Lamport clocks and request tracing

- [x] One thread-safe logical counter per gateway/worker process lifespan
- [x] Timestamp prediction, registration, heartbeat and health messages
- [x] Increment on send; merge on receive with `max(local, remote) + 1`
- [x] Capture actual HTTP response-start times without buffering response bodies
- [x] Preserve prediction JSON and carry clock metadata in HTTP headers
- [x] Show a six-event inference trace in the interface
- [x] Unique trace IDs for concurrent attempts, even with reused request IDs
- [x] Bounded 200-trace retention and explicit partial traces for failures/legacy peers
- [x] New process UUID on restart; clear old gateway traces and merge surviving clocks
- [x] Keep heartbeat lease expiry independent of logical counters

Verification:
- Full suite: 250 passed, including 36 new Lamport checks; one existing Starlette TestClient deprecation warning.
- `scripts/smoke_lamport.py`: real gateway and two workers, six alternating predictions, eight concurrent HTTP traces, malformed-clock rejection, worker restart and gateway restart passed. Owned processes were cleaned up.
- `scripts/smoke_heartbeats.py`: background expiry without health/prediction traffic, survivor routing, worker recovery, old-instance rejection and gateway restart recovery still passed.
- Browser verification: successful predictions alternated workers and displayed the six actual timestamps and process identities; the trace panel was visually inspected at the browser's narrow viewport.
- JavaScript syntax and Git whitespace checks passed.

Windows Application Control blocked native libraries in the original Python 3.11 runtime, including outside the sandbox. Tests and live services used a separate `.verify-venv` alongside the repository, with bundled Python 3.12.14 and compatible scikit-learn 1.7.2. The original environment and security settings were preserved. Python 3.11 was not re-verified for this update. See [USER_GUIDE.md](USER_GUIDE.md) for the working command on this PC. No commits were made.

## Milestone 4: three fixed models in the interface

- [x] Model selector with separate Iris, house and face-image inputs
- [x] House result in historical USD; image class probabilities and preview
- [x] Prepared portrait examples and PNG/JPEG image upload for inference
- [x] Explicit model-specific request and response validation at both boundaries
- [x] Load trusted local ONNX sessions once per worker; preserve Iris if artifacts are absent/broken
- [x] Register loaded capabilities and round-robin independently per requested model
- [x] Six-event Lamport traces for every model; no feature/image data in traces
- [x] Bound image size, dimensions, still-image formats and streamed JSON request size
- [x] Tests for mixed concurrent inference, incompatible workers, malformed responses and missing artifacts
- [x] Real-process demo across both workers and after heartbeat expiry

Verification (2026-10-24, Python 3.12.14 verification environment):
- `python -m pytest -x -q --tb=short`: 301 passed (51 new fixed-model checks), one existing Starlette TestClient deprecation warning. All prepared-artifact tests ran on this PC; no tests skipped. Includes chunked-body limits and serving/preparation parity for EXIF-oriented PNG/JPEG.
- `python scripts/smoke_models.py`: six alternating requests each for Iris, house, female sample and male sample; valid causal traces; invalid input rejection; all models continued on Worker 2 after stopping Worker 1 and waiting for background heartbeat expiry.
- `python scripts/smoke_heartbeats.py`: background expiry, survivor routing, worker restart, stale-instance rejection, gateway restart and automatic re-registration passed after integration.
- Browser: model selection, Iris prediction, historical USD house prediction, face preview/example, actual file-picker upload and image prediction verified; repeated image inference used the second worker and retained a complete trace. Narrow layout visually inspected; no browser warnings/errors. Screenshot saved outside the repository in the workspace verification folder.
- JavaScript syntax and Git whitespace checks passed. Temporary verification services were stopped after checks.

The new image upload accepts inference input only. Public ML-model uploading, authentication and generic model schemas remain deferred. Artifacts are manually prepared/copied to each worker; this is model replication, not an automatic model-distribution protocol. Bully coordinator takeover, two-PC verification and performance benchmarks remain pending. No commits were made.

## Milestone 5: worker dashboard and failure demonstration

- [x] Readable `/workers` page for browsers while preserving JSON for API clients
- [x] Worker addresses, models, heartbeat deadlines, process identities and availability
- [x] Validated per-process gateway prediction counters and bounded 20-request history
- [x] Six-request batch to visibly compare normal routing and survivor routing
- [x] Worker-side pause/resume of heartbeats and new inference; clear simulated-failure labels
- [x] Normal heartbeat expiry/recovery controls eligibility; no direct scheduling toggle
- [x] Controls disabled by default; local launcher uses a fresh internal shared secret
- [x] Loopback/origin checks, stale-instance checks and acknowledgement validation
- [x] Real process stop/restart demonstration and gateway restart recovery

Verification (Python 3.12.14 verification environment):
- `python -m pytest -x -q --tb=short`: 326 passed, including 25 new dashboard/control checks; one existing Starlette TestClient deprecation warning.
- `python scripts/smoke_workers.py`: counters and traces matched six alternating predictions; paused worker stopped heartbeats/inference; background expiry routed every model to the survivor; resume restored scheduling through a heartbeat; actual worker stop/restart changed instance identity and reset its counter; gateway restart cleared history/counters and live workers re-registered. Owned processes were cleaned up.
- Browser: readable page, 3/3 request split, paused-worker countdown and expiry, six survivor requests, resume and 3/3 split verified. Desktop layout visually inspected; no browser warnings/errors. Screenshot saved in the workspace verification folder outside the repository.
- JavaScript syntax and Git whitespace checks passed. Temporary browser-verification services were stopped after checks.

Pause simulates service failure while keeping the worker process alive for resume; the automated and manual terminal demonstrations support genuine process stops separately. Failed in-flight predictions are not retried. Demo pause does not implement coordinator election, permanent failure proof or two-machine operation. No commits were made.

## Two-machine demonstration
- [ ] Optionally confirm GPU model and driver if GPU execution is needed (CPU is sufficient)
- [ ] Run one worker per physical machine
- [ ] Configure LAN communication and firewall
- [ ] Verify registration and round-robin across physical machines
- [x] Add background heartbeats and timeout-based expiry (verified locally)
- [ ] Add least-loaded scheduling if needed by the course rubric

## Distributed features
- [x] Failure detection; new requests route to a surviving compatible worker (no in-flight retry)
- [x] Local complete model replicas on each worker
- [ ] Automatic model distribution between physical machines
- [x] Lamport clocks and a request trace (verified locally)
- [ ] Bully leader election with working coordinator takeover
- [ ] Benchmark latency, throughput, and recovery time

Update this checklist only when tasks are actually verified.
