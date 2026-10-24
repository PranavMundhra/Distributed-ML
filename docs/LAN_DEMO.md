# Two-PC LAN demonstration

The gateway/coordinator and Worker 1 run in separate terminals on PC 1. Worker 2 runs on PC 2. Each worker has complete local copies of the same three models. One inference request runs entirely on its selected worker; separate requests are distributed using per-model round robin. Bully election and coordinator takeover are outside the project scope.

The LAN launchers and checker have a one-PC rehearsal, but a second physical PC is still needed to verify the actual LAN, firewall and failure demonstration. Two distinct advertised IP addresses are a useful check, not proof that two physical computers were used.

## Prepare both computers

Connect both PCs to the same reachable LAN or hotspot. Use the same updated project code on both computers. On PC 2, get the project from GitHub:

```bat
git clone https://github.com/PranavMundhra/Distributed-ML.git
cd Distributed-ML
```

If PC 2 already has a checkout, open its project folder and run `git pull --ff-only` instead. The updated checkout includes the LAN scripts and this guide. Do not copy a virtual environment between computers. Generated model files are not in GitHub; prepare them using the commands below or copy the prepared model folders from PC 1.

From the project folder on PC 2, in Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-prototype.txt
```

Copy the prepared `models/demo/house_price` and `models/demo/male_female` folders from PC 1 into the same locations on PC 2, including manifests and sample images. Generated model files are excluded from Git. The `.sources` download cache is not required for inference. Alternatively, prepare them on PC 2:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-models.txt
.venv\Scripts\python.exe scripts\prepare_demo_models.py
```

The remaining commands use `.venv\Scripts\python.exe`. On the existing development PC, the verified interpreter is `..\.verify-venv\Scripts\python.exe`; substitute that path for PC 1 commands if its original environment is still blocked by Windows Application Control. Keep the usual security settings enabled.

Run `ipconfig` on both PCs. Use the IPv4 address of the connected Wi-Fi/Ethernet adapter. The examples below assume **PC 1 = 192.168.1.10** and **PC 2 = 192.168.1.11**; replace these with your actual addresses. The launcher requires a private IPv4 address, binds only that interface, and checks that the address belongs to this PC and the port is free. It rejects `localhost`, `127.0.0.1`, `0.0.0.0` and public IPs.

Stop the old combined local demo before this demonstration. The LAN setup starts only one worker on each PC. No `.env` editing is needed: command-line addresses and worker IDs override existing `.env`/environment values. Existing heartbeat and model-directory settings still apply. The normal defaults are two-second heartbeats and an eight-second silence limit.

## PC 1: gateway, then Worker 1

Open two Command Prompt terminals in the updated project folder. Start the gateway first in terminal 1:

```bat
.venv\Scripts\python.exe scripts\run_lan.py gateway --ip 192.168.1.10
```

Leave it running. In terminal 2:

```bat
.venv\Scripts\python.exe scripts\run_lan.py worker --ip 192.168.1.10 --gateway-ip 192.168.1.10 --worker-id cpu-worker-1
```

The gateway uses port 8000; Worker 1 uses port 8001. Ctrl+C in terminal 2 stops only Worker 1. The services print their addresses and keep startup, registration and request logs visible in their own terminals.

## PC 2: Worker 2

In Command Prompt, from the updated project folder:

```bat
.venv\Scripts\python.exe scripts\run_lan.py worker --ip 192.168.1.11 --gateway-ip 192.168.1.10 --worker-id cpu-worker-2
```

Worker 2 also uses port 8001; this is valid because it runs on a different computer. It advertises `http://192.168.1.11:8001`, registers with PC 1, and sends heartbeats to PC 1. Its startup preflight must reach the gateway, and the actual registration validates the response before inference becomes ready. Workers train Iris locally and advertise only additional models they successfully load.

To check an unused local port and PC 2-to-gateway connectivity before starting the worker, append `--check-only` to that command. This does not prove the reverse gateway-to-worker path; the inference checker does that after startup. A preflight on a port already occupied by a running service correctly fails.

All LAN services explicitly disable pause/resume demo controls, even if a token exists in `.env` or the environment. Use Ctrl+C in a worker's terminal for an actual process stop. The dashboard, request batches, addresses, counters and traces remain available.

## Network access

Open `http://192.168.1.10:8000/` and `http://192.168.1.10:8000/workers` from either computer. Relative frontend requests continue to go through PC 1. The browser does not need to call a remote inference endpoint directly.

Allow Python on the trusted **Private** network if Windows asks. PC 2 must reach PC 1 TCP 8000; PC 1 must reach PC 2 TCP 8001. Browsing PC 1 from PC 2 additionally uses PC 1 TCP 8000. No router port forwarding is needed.

If Windows Firewall blocks the ports, the owner can add narrowly scoped rules in an Administrator PowerShell terminal. Replace both addresses first; these commands are documented here and are not executed by the launcher.

PC 1:

```powershell
New-NetFirewallRule -DisplayName "Distributed-ML PC1 LAN" -Direction Inbound -Action Allow -Protocol TCP -LocalAddress 192.168.1.10 -LocalPort 8000,8001 -RemoteAddress 192.168.1.11 -Profile Private
```

PC 2:

```powershell
New-NetFirewallRule -DisplayName "Distributed-ML PC2 LAN" -Direction Inbound -Action Allow -Protocol TCP -LocalAddress 192.168.1.11 -LocalPort 8001 -RemoteAddress 192.168.1.10 -Profile Private
```

Use these rules only for a network you trust and have designated Private in Windows settings. Test the required direction with PowerShell `Test-NetConnection 192.168.1.10 -Port 8000` on PC 2, and `Test-NetConnection 192.168.1.11 -Port 8001` on PC 1. Both services must be running. If both computers share an SSID but cannot communicate, the Wi-Fi may isolate clients; use a suitable hotspot/LAN. The scripts do not change firewall rules or network profiles.

## Verify normal inference

Keep other prediction clients idle for the round-robin check. From a third terminal on PC 1:

```bat
.venv\Scripts\python.exe scripts\check_lan.py --gateway-ip 192.168.1.10
```

The checker waits for both configured worker IDs and accepted heartbeats, rejects extra workers and loopback advertisements, and requires two distinct worker IPs with one on PC 1. It checks increasing heartbeat sequences with unchanged process identities, directly reaches each active worker, and checks that both advertise all three models.

It sends six requests for each of Iris, house price, female sample and male sample: **24 real predictions**, each with a unique request ID. Each group must alternate workers 3/3. Output includes worker ID, advertised IP/port, request ID and trace ID. The checker validates results and the six-event Lamport trace against the worker's current registered process identity. Example images are fetched from the gateway; the checker need not have a separate local copy of them.

Open the dashboard and use **Send 6 Iris requests** for the visual presentation. Show the two addresses, changing counters and both workers' terminal logs. Submit a house prediction and an image inference through the frontend and open the request trace.

## Stop Worker 2, then recover it

1. Press Ctrl+C in **PC 2's worker terminal**. Leave PC 1's gateway and Worker 1 running.
2. Watch Worker 2's heartbeat stop and its deadline expire. Normally detection happens about 8-8.5 seconds after the last accepted heartbeat, with additional scheduling delay possible. A health/prediction connection failure may mark it unavailable earlier; the checker below specifically waits for heartbeat expiry.
3. From PC 1, run:

```bat
.venv\Scripts\python.exe scripts\check_lan.py --gateway-ip 192.168.1.10 --failed-worker cpu-worker-2
```

This polls status snapshots while waiting for expiry, then verifies that all 24 new predictions use Worker 1, with valid traces. It does not stop a service, renew a heartbeat deadline or automatically retry a failed prediction.

Restart PC 2's worker with its original launch command, then run the normal checker again. It must regain eligibility through registration/heartbeats and return to 3/3 routing. Show the new process identity on the dashboard; the restarted worker's prediction counter starts again at zero. To demonstrate Worker 1 failure instead, stop its terminal and use `--failed-worker cpu-worker-1`.

A failed request already in progress can return an error. This demonstration proves survivor routing for new requests after detection. Stopping the gateway interrupts inference access; automatic coordinator takeover is outside scope.

## Other ports and local rehearsal

The launchers accept `--port`. Workers also accept `--gateway-port`; the checker accepts `--gateway-port`, `--worker-ids ID1 ID2` and `--wait-seconds`. If the gateway port changes, use it consistently in all commands. If a worker port changes, its advertised address changes automatically. Two services on the same PC must use different ports.

The self-contained rehearsal starts a gateway and two worker processes on this PC's actual private interface using temporary ports:

```bat
.venv\Scripts\python.exe scripts\smoke_lan.py --ip 192.168.1.10
```

It verifies address binding, disabled inherited demo controls, all models, round robin, traces, actual worker process stop/restart, survivor routing and worker re-registration after gateway restart. It cleans up only its own processes. This is a **one-PC rehearsal**, not a two-PC LAN result. The checker requires explicit `--allow-same-host` in this situation and labels the result accordingly; the rehearsal supplies that flag internally. Run the default checker on the actual two-PC setup to complete physical verification.

## Presentation evidence still to collect

Record both PCs and their addresses, the 3/3 request split, Worker 2's terminal stop, heartbeat expiry, survivor-only requests, and recovery with a new process identity. Record Python versions and hardware. Latency, throughput and recovery-time benchmarks remain a separate task; these smoke checks do not establish a speed improvement.
