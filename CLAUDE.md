# Claude Code Project Instructions

## 1. Project Overview

Build a Python-first Distributed ML Inference Platform demonstrated across two physical machines:

- **CPU machine:** runs CPU-based model inference.
- **GPU machine:** runs GPU-accelerated inference when compatible hardware and drivers are available.

The platform should demonstrate distributed inference, worker registration, health monitoring, load balancing, model management, fault tolerance, and distributed algorithms.

## 2. Mandatory First Steps

Before making changes to the project:

1. Read `README.md` and the relevant files under `docs/`.
2. Inspect the repository structure and current Git status.
3. Determine the current operating system, Python version, and available environment.
4. Verify installed dependencies using the active Python interpreter.
5. Identify the existing project root and virtual environment location.
6. Report inconsistencies between the documentation and the actual setup.

Do not assume that another developer has the same drive letters, usernames, directory structure, Python installation, or hardware.

## 3. Cross-Machine Compatibility

The project must remain portable across Windows computers and should avoid machine-specific assumptions.

- Never hardcode absolute paths tied to a particular user's computer.
- Use project-relative paths where appropriate, such as `Path(__file__).resolve()`.
- Use `pathlib.Path` for filesystem operations.
- Read configurable paths, host addresses, ports, and service URLs from environment variables when appropriate.
- Provide an `.env.example` containing placeholders, never real credentials.
- Keep machine-specific settings in local environment configuration that is excluded from Git.
- Do not assume that every developer has a D: or E: drive.
- Do not assume that both computers have the same Python version, GPU hardware, or installed packages.
- Document setup commands for the relevant operating systems and shells.
- Keep CPU inference functional without CUDA or GPU-specific dependencies.

## 4. Python Environment and Dependencies

Use Python 3.11 as the current development baseline unless project requirements justify changing it.

The original development machine uses pip. Continue using the existing environment unless the user explicitly approves a package-management change.

Do not hardcode a virtual environment's absolute interpreter path in application code, tests, or shared scripts.

Prefer commands that work with an activated environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

When appropriate, document activation instructions separately for Windows PowerShell and other supported shells.

Do not install dependencies or upgrade major package versions without explaining why the change is necessary.

CPU and GPU machines may require different hardware-specific installation steps. Keep common application dependencies consistent wherever possible.

## 5. Architecture and Implementation Order

Implement the platform incrementally:

1. FastAPI worker with health and inference endpoints.
2. Gateway that communicates with a worker over HTTP.
3. Worker registration and an in-memory registry.
4. Communication between the two physical machines.
5. Heartbeats and failure detection.
6. Round-robin and least-loaded scheduling.
7. CPU/GPU-aware model routing.
8. Model registration, artifact distribution, and replication.
9. Failure recovery and request rerouting.
10. Required distributed algorithms, including leader election and Lamport logical clocks.
11. Automated tests, benchmarks, and a reproducible demonstration procedure.

Keep the gateway, coordinator, worker, inference logic, and distributed algorithms separated into understandable modules.

## 6. Infrastructure Constraints

Docker and the Redis server are intentionally deferred for the initial prototype.

- Do not require Docker or Redis to run the first working prototype.
- Start with native Python processes and an in-memory registry.
- Add Redis only when shared state or coordination requirements justify it.
- Add PostgreSQL when persistent metadata or inference records are needed.
- Introduce Docker when reproducible deployment or service isolation provides a clear benefit.

The Python Redis client being installed does not imply that a Redis server is running.

## 7. Configuration and Security

- Never commit passwords, API tokens, private keys, or real `.env` files.
- Keep `.env`, virtual environments, caches, datasets, logs, and large model artifacts out of Git.
- Do not expose development services publicly without an explicit security design.
- Use configurable worker URLs rather than hardcoded LAN IP addresses.
- Apply request timeouts, input validation, and appropriate error handling to network calls.
- Do not assume a trusted network removes the need for access controls.

## 8. Coding and Testing Standards

- Inspect existing files before editing them.
- Explain the intended changes before making broad architectural modifications.
- Prefer small, testable implementations over large rewrites.
- Test distributed algorithms independently before integration.
- Add or update tests when changing behavior.
- Run relevant tests using the active project environment.
- Report actual test results; never claim that unexecuted tests passed.
- Avoid adding dependencies when the standard library or existing packages are sufficient.

## 9. Documentation and Handoffs

Keep these files current:

- `README.md`: project purpose, setup, and quick start.
- `docs/PROJECT_PLAN.md`: implementation checklist and progress.
- `docs/DECISIONS.md`: architectural decisions and rationale.
- `docs/SETUP_WINDOWS.md`: Windows setup and networking instructions.
- `CLAUDE.md`: durable instructions for future Claude Code sessions.

At the end of each meaningful task, report:

1. What changed.
2. Which files changed.
3. Which commands and tests were executed.
4. Any outstanding issues.
5. The recommended next step.

## 10. Definition of Done

A task is complete only when the requested behavior is implemented, relevant tests have been run, and important configuration or architectural decisions have been documented.

Before beginning a new task, inspect the current project state rather than assuming previous work was completed.
