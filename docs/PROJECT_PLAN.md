# Project plan

## Setup
- [x] Python 3.11 environment created on D:
- [x] Core dependencies installed using pip
- [ ] GitHub repository created and pushed
- [ ] Verify environment imports

## Single-machine prototype
- [ ] FastAPI worker health endpoint
- [ ] Inference endpoint and request schemas
- [ ] Gateway-to-worker HTTP communication
- [ ] Automated tests

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
