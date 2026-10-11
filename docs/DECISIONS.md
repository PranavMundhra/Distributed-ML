# Technical decisions

1. Use Python and FastAPI for the initial gateway and workers.
2. Demonstrate communication between two physical Windows machines.
3. Keep the current pip environment; do not switch to uv without approval.
4. Defer Docker and Redis server installation until needed.
5. Start with an in-memory registry and add persistence when justified.
6. Keep source code and the Python environment on D:.
7. Keep secrets, datasets, and large model artifacts out of Git.

Open questions:
- What GPU and driver are available on the second machine?
- Which ML model and dataset will be used?
- Where should the coordinator run?
