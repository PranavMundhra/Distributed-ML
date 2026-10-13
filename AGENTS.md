# Repository Guidelines

## Project Structure & Module Organization

This Python 3.11 project runs a single-machine distributed inference prototype. `app/coordinator/` implements the HTTP gateway; `app/worker/` trains and serves the CPU Iris classifier. `app/core/` holds configuration and logging, `app/schemas/` defines the API contract, and `app/static/` contains the minimal interface. Unused module directories are placeholders.

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
```

These create and activate an environment, install prototype dependencies, run tests, launch both services, and check real HTTP inference plus worker failure/recovery. `requirements-lock.txt` retains the original broader environment snapshot. Manual entry points are `app.coordinator.main:app` and `app.worker.main:app`; run them with Uvicorn on ports 8000 and 8001. No build step is required.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep functions focused and add useful type hints. Define explicit Pydantic request and response schemas; use standard Python logging. No formatter or linter is currently configured.

## Testing Guidelines

Use pytest, pytest-asyncio for asynchronous behavior, and HTTPX for HTTP tests. Name files `test_*.py` and functions `test_*`. Cover health, inference, forwarding, lifecycle cleanup, connection failures, and invalid payloads. Keep tests reproducible on CPU-only machines. No numeric coverage threshold is configured; report exact test commands and outcomes. Add registry and heartbeat tests when those features are implemented.

## Commit & Pull Request Guidelines

History uses short imperative descriptions such as `Revise CLAUDE.md` and occasional `docs:` prefixes; no strict convention is established. Keep commits focused. PRs should explain behavior changes, link relevant issues, include validation results, and document configuration or dependency changes. Update setup instructions when commands change.

## Architecture & Configuration

The prototype uses one gateway and one configured HTTP worker. Registration, round-robin routing, and heartbeats remain future work. Keep CPU execution the default and GPU support optional. Defer external infrastructure until required. Use environment variables and portable paths; commit safe `.env.example` placeholders, never secrets, environments, logs, or model binaries.
