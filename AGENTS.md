# Repository Guidelines

## Project Structure & Module Organization

This Python 3.11+ project runs one local gateway and two registered CPU workers. `app/coordinator/` implements the HTTP gateway, in-memory registry, compatible per-model round-robin selection, and heartbeat expiry task. `app/worker/` registers at startup, sends periodic heartbeats when configured, and serves Iris plus prepared house-price and face-image ONNX replicas. `app/core/` holds configuration, logging, process-local Lamport clocks and bounded inference traces. `app/schemas/` defines explicit fixed-model API contracts, and `app/static/` contains the minimal three-model interface.

Use `tests/` for automated tests, `scripts/` for startup and smoke checks, and `docs/PROJECT_PLAN.md` for verified progress. `docs/USER_GUIDE.md` explains operation. `models/` holds local artifacts; only its `.gitkeep` is tracked. `CLAUDE.md` contains architecture guidance.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-prototype.txt
python -m pytest
python -m pytest tests/test_worker.py -q
python scripts/run_local.py
python scripts/smoke_local.py
python scripts/smoke_round_robin.py
python scripts/smoke_heartbeats.py
python scripts/smoke_lamport.py
python scripts/smoke_models.py
python scripts/smoke_workers.py
```

These install dependencies, run tests, launch the gateway and two workers, and verify inference, registration, round-robin, heartbeat expiry, worker recovery, and gateway restart recovery. Smoke scripts use temporary ports and stop their owned processes. `requirements-lock.txt` retains the original broader environment snapshot. Manual entry points are `app.coordinator.main:app` and `app.worker.main:app`; start the gateway first. See the user guide for settings. No build step is required.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep functions focused and add useful type hints. Define explicit Pydantic request and response schemas; use standard Python logging. No formatter or linter is currently configured.

## Testing Guidelines

Use pytest, pytest-asyncio, and HTTPX. Cover health, inference, registration, duplicate/conflicting workers, round-robin, concurrency, heartbeat expiry/recovery, stale sequences and process instances, lifecycle cleanup, and invalid payloads. Test lease boundaries with an injected monotonic clock; keep tests CPU-only and reproducible. No numeric coverage threshold is configured; report exact test commands and outcomes.

## Commit & Pull Request Guidelines

History uses short imperative descriptions such as `Revise CLAUDE.md` and occasional `docs:` prefixes; no strict convention is established. Keep commits focused. PRs should explain behavior changes, link relevant issues, include validation results, and document configuration or dependency changes. Update setup instructions when commands change.

## Architecture & Configuration

Use one gateway process with an in-memory registry. Workers use `GATEWAY_URL`, `WORKER_URL`, and `WORKER_ID`. Heartbeat defaults are a 2-second interval, 8-second timeout, and 0.5-second sweep. Use the gateway's monotonic receive time for expiry; UTC receipt timestamps are for display. Reject stale sequences and old instance IDs. Health probes/predictions must not renew heartbeat leases. Cancel background tasks before closing clients. Gateway restarts clear registration; workers re-register on heartbeat HTTP 404. Bounded retry of failed predictions remains future work. Keep CPU execution the default, GPU optional, and avoid extra infrastructure. Use portable paths and safe `.env.example` values; never commit secrets or generated environments/logs.

Keep one thread-safe Lamport counter per service lifespan. Share it across inference, registration, heartbeat and health messages. Increment on send; receive with `max(local, remote) + 1`. Capture actual response-start timestamps in pure ASGI middleware. Logical time must never drive heartbeat deadlines. A restarted worker uses its new clock instance UUID for registration too. Preserve the Iris prediction JSON contract; house and face have separate endpoints. Clock metadata travels in headers and `GET /traces/{trace_id}`. Trace IDs distinguish attempts even when request IDs repeat. Retain only 200 completed traces, clear on gateway restart, and never invent timestamps for legacy peers or failed messages. Load only the two trusted fixed local ONNX paths once per worker. Advertise only loaded models and keep per-model scheduling cursors. Image inputs are memory-only, still PNG/JPEG, at most 5 MiB, 32x32 minimum, 20 MP maximum; bound their streamed JSON body to 7 MiB. Bully election remains future work; anonymous model uploads are deferred.

The worker dashboard uses read-only `/workers/status` snapshots without health probes or lease renewal. `/workers` serves HTML when browsers request it and JSON to API clients; `?format=json` forces JSON. Preserve current-process gateway success counters and only 20 recent completions, without input data. Demo pause/resume is opt-in via a per-launch internal secret, disabled for manual services by default. Keep gateway controls local with origin checks; validate the worker instance and acknowledgement. A demo pause really stops future heartbeats and rejects new inference, but leaves the process alive for resume. Never fake gateway eligibility, heartbeat expiry, process death or process identity. Resume depends on a genuine heartbeat; actual crash/restart checks remain separate. Do not expose the shared control secret to browser code, logs or status responses.
