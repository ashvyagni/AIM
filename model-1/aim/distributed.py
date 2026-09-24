"""Gloo/DDP benchmark entry point for torchrun; run on authorized hosts only."""
import argparse
import datetime
import os
import platform
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .benchmark import describe
from .neural import CausalLM,ModelConfig
from .tracking import Run,write_json
from .training import seed_all


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    parser.add_argument("--steps",type=int,default=5)
    parser.add_argument("--elements",type=int,default=65536)
    args=parser.parse_args()
    rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE","1"))
    with Run(args.runs,"distributed-rank"+str(rank),{"rank":rank,"world_size":world,"steps":args.steps,"elements":args.elements}) as run:
        if not 3<=args.steps<=100 or not 1<=args.elements<=4_194_304:
            raise ValueError("Bounded benchmark: 3..100 steps, <=16MiB float32 payload")
        seed_all(17)
        dist.init_process_group("gloo",timeout=datetime.timedelta(seconds=30))
        try:
            model=DistributedDataParallel(CausalLM(ModelConfig()))
            opt=torch.optim.AdamW(model.parameters(),lr=0.001)
            generator=torch.Generator().manual_seed(17+rank)
            batch=torch.randint(0,259,(2,32),generator=generator)
            def step():
                opt.zero_grad()
                loss=torch.nn.functional.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten())
                loss.backward();opt.step()
            step();step()
            times=[];collectives=[]
            for _ in range(args.steps):
                dist.barrier();start=time.perf_counter();step();elapsed=time.perf_counter()-start
                maximum=torch.tensor(elapsed,dtype=torch.float64);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
                times.append(maximum.item())
                payload=torch.full((args.elements,),float(rank+1));dist.barrier()
                start=time.perf_counter();dist.all_reduce(payload);duration=time.perf_counter()-start
                if not torch.all(payload==world*(world+1)/2):
                    raise RuntimeError("Incorrect collective result")
                maximum=torch.tensor(duration,dtype=torch.float64);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
                collectives.append(maximum.item())
            # Exact parameter agreement after replicated optimizer updates.
            from .tracking import digest
            parameter_hash=digest(b"".join(p.detach().numpy().tobytes() for p in model.parameters()))
            records=[None]*world
            dist.all_gather_object(records,{"hostname":platform.node(),"parameter_hash":parameter_hash})
            consistent=len({r["parameter_hash"] for r in records})==1
            if not consistent:
                raise RuntimeError("DDP weights differ between ranks")
            result={"rank":rank,"world_size":world,"hosts":records,"replicas_equal":consistent,
                    "training_step":describe(times),"all_reduce":describe(collectives),
                    "payload_bytes":args.elements*4,"aggregate_tokens_per_second":world*2*31/describe(times)["median_seconds"],
                    "scope":"local multi-process" if len({r["hostname"] for r in records})==1 else "multi-host",
                    "scaling_efficiency":None,"note":"Need a matched one-rank run and multiple physical hosts before estimating VIT efficiency"}
            write_json(run.path/"metrics.json",result)
            print(run.path)
        finally:
            dist.destroy_process_group()


if __name__=="__main__": main()
