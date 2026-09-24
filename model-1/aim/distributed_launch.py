"""Bounded local DDP launcher, with a retained log even if rendezvous fails."""
import argparse
import os
import platform
import signal
import socket
import subprocess
import sys
from pathlib import Path

from .tracking import ROOT, Run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks",type=int,choices=(1,2,4),default=2)
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    args = parser.parse_args()
    with Run(args.runs,"local-ddp-launch",{"ranks":args.ranks,"deadline_seconds":90,"scope":"loopback only"}) as run:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0))
            port = sock.getsockname()[1]
        # Small release/bind race is possible; launch failure is retained, never hidden.
        command = [sys.executable,"-m","torch.distributed.run","--nnodes=1",f"--nproc-per-node={args.ranks}",
                   "--master-addr=127.0.0.1",f"--master-port={port}","--rdzv-conf=timeout=30",
                   "-m","aim.distributed","--steps","5","--runs",str((run.path/"ranks").resolve())]
        env = dict(os.environ, OMP_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo0" if platform.system()=="Darwin" else "lo")
        write_json(run.path/"launch.json", {"command":command,"environment_overrides":{k:env[k] for k in ("OMP_NUM_THREADS","GLOO_SOCKET_IFNAME")}})
        with (run.path/"launcher.log").open("x") as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = process.wait(timeout=90)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"Distributed launch exited {code}; inspect retained launcher.log")
    print(run.path)


if __name__ == "__main__":
    main()
