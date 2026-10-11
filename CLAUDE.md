import pypandoc

content = r"""# CLAUDE.md — Distributed ML Inference Platform

## 1. Project purpose

Build a small, demonstrable **Fault-Tolerant Distributed ML Inference System** for a Distributed Computing course.

The system distributes ML inference requests across multiple worker processes/machines. The main academic value is demonstrating distributed-system concepts—communication, naming/registration, failure detection, fault tolerance, and measurable evaluation—not building a production-grade ML platform.

**Keep the project intentionally minimal.** Prefer a working, testable end-to-end demonstration over extra infrastructure or features.

## 2. Intended minimal architecture

Initial topology:

- **Coordinator**: accepts inference requests, tracks available workers, chooses a worker, and forwards requests.
- **CPU worker**: runs on the coordinator machine as a separate process.
- **GPU worker**: runs on a second Windows computer, reachable over the local network.
- **Communication**: HTTP request/response using FastAPI and Uvicorn.
- **Worker registry**: in-memory registry maintained by the coordinator.
- **Load distribution**: simple round-robin routing among healthy, compatible workers.
- **Failure detection**: workers send heartbeats; the coordinator marks a worker unavailable after a configurable timeout.
- **Fault-tolerance demonstration**: stop or disconnect a worker and show requests being routed to another healthy compatible worker.
- **Logical time**: add Lamport logical-clock values to relevant events or request logs as a focused distributed-computing concept.
- **Evaluation**: measure latency, throughput, and recovery time under clearly described test conditions.

Start with one coordinator and two workers. Do not introduce leader election, distributed consensus, or multiple coordinators unless the course rubric or instructor specifically requires them. If an election algorithm is required, implement one small, clearly isolated demonstration such as the Bully algorithm rather than building a full consensus system.

## 3. Technology choices

Use the existing project stack where appropriate:

- Python 3.11
- FastAPI
- Uvicorn
- HTTPX for HTTP requests and tests
- Pydantic / pydantic-settings for configuration and validation
- scikit-learn and Joblib for a small CPU-friendly demonstration model
- pytest and pytest-asyncio for tests
- Standard Python logging

The initial model should be lightweight and reproducible. Prefer a small scikit-learn model that can run on CPU on both computers. Treat GPU execution as an optional capability of the second worker until its GPU model, drivers, and compatible ML framework have been verified. Do not install PyTorch or other large GPU frameworks without a specific, justified need.

The Python Redis client may be installed, but a Redis **server** is not part of the initial architecture. Do not add dependencies merely because they are already installed.

## 4. Explicitly deferred infrastructure

Do **not** add these to the initial version unless there is a demonstrated requirement:

- Docker or Docker Compose
- Redis server
- PostgreSQL or another external database
- Kafka, RabbitMQ, or another message broker
- Kubernetes
- gRPC
- Authentication systems or complex authorization
- A web dashboard or frontend
- Multiple coordinators, consensus protocols, or advanced autoscaling
- Model registries, elaborate model versioning, or production deployment infrastructure

Use native Python processes on Windows first. Add infrastructure only when it solves a specific problem and its value can be demonstrated.

## 5. Portability across collaborators' computers

The repository must work on Windows computers with different usernames, drive letters, and hardware.

- Never hardcode a developer's absolute path, username, drive letter, or machine-specific directory in application code.
- Use paths relative to the project root, such as `pathlib.Path(__file__).resolve()`.
- Read machine-specific settings from environment variables or a local `.env` file.
- Commit `.env.example`, not real `.env` files or secrets.
- Do not commit virtual environments, caches, generated datasets, model binaries, logs, or machine-specific configuration.
- Keep setup instructions valid for a fresh Windows checkout.
- Use the same Python major/minor version where practical; Python 3.11 is the current development target.
- Make CPU the safe default. GPU support must be optional and must not prevent a CPU-only machine from starting or running tests.
- Bind a network-accessible worker to an appropriate host interface only when needed; document firewall and LAN requirements. Do not assume that `localhost` on one computer is reachable from another.
- Avoid assuming that all machines have the same installed packages or GPU capabilities.

## 6. Repository conventions

Keep the codebase simple and organized. Adapt to the existing folders rather than moving files unnecessarily.

Suggested structure:

```text
distributed-ml-platform/
├── app/
│   ├── coordinator/
│   ├── worker/
│   ├── common/
│   └── main.py
├── models/
├── scripts/
├── tests/
├── docs/
├── .env.example
├── .gitignore
├── README.md
├── CLAUDE.md
└── requirements-lock.txt
```

This is a guide, not a requirement to create every directory immediately. Before adding or moving files, inspect the repository and explain why the change is necessary.

## 7. Implementation sequence

Work incrementally in this order:

1. Inspect the existing files and summarize the current state before changing anything.
2. Establish configuration, logging, and a health endpoint.
3. Implement one worker that loads a small model and serves an inference endpoint.
4. Implement the coordinator's worker registration and in-memory registry.
5. Add coordinator-to-worker HTTP forwarding and round-robin selection.
6. Add heartbeat reporting and timeout-based failure detection.
7. Demonstrate rerouting when a worker becomes unavailable.
8. Add simple Lamport logical-clock metadata to selected events and logs.
9. Add automated tests for routing, worker health, failure handling, and invalid requests.
10. Add scripts and documentation to run the coordinator and workers in separate terminals.
11. Measure latency, throughput, and recovery time; document the setup and limitations.
12. Add an election algorithm only if required by the rubric or instructor.

At each stage, keep the system runnable and tests focused. Avoid implementing all stages in one large change.

## 8. API and reliability guidelines

- Use explicit request and response schemas.
- Return useful HTTP status codes and concise error messages.
- Set finite timeouts on coordinator-to-worker requests.
- Handle a worker timeout or connection error without crashing the coordinator.
- Never route a request to a worker known to be unhealthy.
- If no compatible worker is available, return a clear service-unavailable response.
- Avoid infinite retries and retry storms.
- Make worker registration and heartbeat updates safe against malformed or stale data.
- Log request IDs, selected worker IDs, status, elapsed time, and relevant logical-clock values.
- Do not log secrets or unnecessary personal data.
- Clearly distinguish a worker being marked unhealthy from proof that the machine has permanently failed; heartbeat-based detection is imperfect.

## 9. Testing and evaluation

Tests should cover at least:

- Worker health endpoint.
- Successful inference on a worker.
- Worker registration and registry updates.
- Round-robin distribution across healthy workers.
- Skipping or removing an unhealthy worker.
- Timeout and connection-error handling.
- Clear behavior when no workers are available.
- Invalid request payloads.
- Basic logical-clock ordering or monotonicity, if logical clocks are implemented.

Evaluation should include:
- Test environment and hardware description.
- Number of requests and concurrency/load settings.
- Average and preferably median/p95 latency where feasible.
- Throughput over a stated interval.
- Time from worker failure to detection and recovery.
- Results with both workers available versus one worker unavailable.
- Limitations, including LAN variability and the difference between a controlled demo and production fault tolerance.

Do not fabricate measurements. Provide scripts or repeatable steps so the results can be reproduced.

## 10. Coding rules for Claude Code

Before making changes:
- Inspect the relevant files and existing tests.
- State the smallest sensible plan.
- Preserve existing working code unless a change is justified.
- Ask before adding a large dependency, infrastructure service, or broad refactor.

While implementing:
- Prefer straightforward, readable Python over clever abstractions.
- Keep functions and modules focused.
- Add type hints where useful.
- Use configuration rather than hardcoded machine-specific values.
- Avoid speculative features and premature optimization.
- Do not silently change the architecture or introduce an external service.
- Do not overwrite user work or delete files without explaining the impact and obtaining confirmation when appropriate.

After changes:
- Run relevant tests and report the exact commands and outcomes.
- If a command cannot run, state why instead of claiming success.
- Update README or setup documentation when behavior or commands change.
- Explain the design in terms the student can present and defend.
- List known limitations and the next smallest step.

## 11. Git and collaboration

- Keep commits small and focused.
- Do not commit `.env`, secrets, virtual environments, cache directories, local logs, or large generated artifacts.
- Use `.env.example` for documented configuration keys with safe placeholder values.
- Do not rewrite shared Git history or force-push unless explicitly requested.
- Ensure setup commands and documentation do not depend on one contributor's drive letter or username.
- When changing dependencies, explain why each new dependency is needed and update the dependency file consistently.

## 12. Definition of done for the first demonstrable version

The first version is complete when:

1. The coordinator starts on the CPU machine.
2. A CPU worker starts as a separate process.
3. A second worker can start on another computer and communicate over the LAN.
4. An inference request is routed to a healthy worker and returns a valid result.
5. Requests are distributed across available workers using the selected simple policy.
6. When one worker is stopped or made unreachable, the coordinator detects the failure after the configured timeout and routes requests to another compatible worker.
7. Logs and repeatable evaluation steps demonstrate the behavior.
8. Tests and Windows setup instructions are available.
9. The student can explain the architecture, failure-detection limitations, and measured results.

**Default decision rule:** If a feature is not needed for this demonstration, the course rubric, or an explicitly approved extension, leave it out.
"""

output_path = "/mnt/data/CLAUDE.md"
pypandoc.convert_text(
    content,
    to="md",
    format="md",
    outputfile=output_path,
    extra_args=["--standalone"],
)
print(f"Created {output_path}")
