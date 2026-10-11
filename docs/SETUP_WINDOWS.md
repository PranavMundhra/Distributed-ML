# Windows setup

Project root:
D:\Dev\Projects\distributed-ml-platform

Python environment:
D:\PythonEnvs\distributed-ml-platform

Run Python commands using:

& "D:\PythonEnvs\distributed-ml-platform\Scripts\python.exe" -m pytest

The current machine is CPU-only. Install GPU-specific packages only on the
GPU machine after checking its hardware and drivers.

For LAN communication, use the worker machine's LAN IP, configure required
Windows Firewall rules for a trusted private network, and do not expose
development endpoints publicly.
