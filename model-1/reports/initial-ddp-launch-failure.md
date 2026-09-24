# Initial local distributed launch failure — 2026-09-24

Command: `.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m aim.distributed --steps 5`

The launcher did not reach the rank scripts. Its TCP rendezvous repeatedly failed to resolve/connect to `localhost:0` in the restricted execution environment. The observed error included:

```
The IPv6 network addresses of (localhost, 0) cannot be retrieved
The client socket has timed out after 60000ms while trying to connect to (localhost, 0).
```

The launcher timed out and exited with code 1 after its retry. A subsequent attempt to interrupt the owned process found it had already exited. No throughput or distributed correctness result is claimed from this attempt. Since failure happened before rank script initialization, no rank manifest exists. This incident record preserves the unsuccessful attempt; the original tool transcript contains its stack trace.

Correction: add `aim.distributed_launch`, a bounded launcher with explicit IPv4 loopback, a retained launcher log, and process-group cleanup. The subsequent attempt and outcome are separate run artifacts. Loopback permission is specific to the local benchmark; it does not grant access to the VIT machines.
