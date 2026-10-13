# Project plan

## Setup
- [x] Python 3.11 environment created on D:
- [x] Core dependencies installed using pip
- [ ] GitHub repository created and pushed
- [x] Verify environment imports

## Single-machine prototype
- [x] FastAPI worker health endpoint
- [x] Inference endpoint and request schemas
- [x] Gateway-to-worker HTTP communication
- [x] Automated tests

The current version runs one gateway and one CPU Iris worker as separate local
processes. The gateway serves a minimal matte dark and purple interface. See
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

## Two-machine demonstration
- [ ] Confirm GPU model and driver on second machine
- [ ] Run one worker per physical machine
- [ ] Configure LAN communication and firewall
- [ ] Register workers and track heartbeats
- [ ] Implement round-robin and least-loaded scheduling

## Distributed features
- [ ] Failure detection and rerouting
- [ ] Model distribution and replication
- [ ] Leader election and Lamport clocks
- [ ] Benchmark latency, throughput, and recovery time

Update this checklist only when tasks are actually verified.
