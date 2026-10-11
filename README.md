# Distributed ML Inference Platform

A Python-first distributed inference system running across two physical
Windows machines: one CPU-only and one GPU-capable.

## Planned features
- FastAPI inference gateway and workers
- Worker registration and health monitoring
- CPU/GPU-aware load balancing
- Model discovery and replication
- Failure detection and recovery
- Leader election and Lamport logical clocks
- Latency, throughput, and recovery benchmarks

## Environment
- Python 3.11
- Project: D:\Dev\Projects\distributed-ml-platform
- Virtual environment: D:\PythonEnvs\distributed-ml-platform
- Dependencies installed with pip
- Docker and Redis server deferred for the initial prototype

## Development
Start with a minimal worker and gateway, then introduce the second machine
and distributed algorithms incrementally.
