"""Local measurements never imply VIT measurements or linear cluster scaling."""
from __future__ import annotations

import os
import platform
import resource
import statistics
import time
from pathlib import Path

from .tracking import Run, command, write_json


def describe(values):
    values=sorted(values)
    return {"count":len(values),"median_seconds":statistics.median(values),
            "p95_seconds":values[min(len(values)-1,int(0.95*len(values)))],"samples_seconds":values}


def measure(fn,repeats,warmup=2):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();values.append(time.perf_counter()-start)
    return describe(values)


def benchmark(config,runs):
    import torch
    from .neural import CausalLM,ModelConfig,ByteTokenizer
    from .training import seed_all
    with Run(Path(runs),"benchmark",config) as run:
        repeats=config.get("repeats",5)
        if type(repeats) is not int or not 3 <= repeats <= 100:
            raise ValueError("Benchmark repeats must be 3..100")
        seed_all(config.get("seed",17))
        cfg=ModelConfig(**config.get("model",{}))
        if cfg.parameter_estimate()>2_000_000:
            raise ValueError("Profile tiny reference before commissioning a large allocation")
        model=CausalLM(cfg)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        length=min(64,cfg.context)
        batch=torch.randint(0,cfg.vocab_size,(2,length))
        def step():
            optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(batch[:,:-1]).flatten(0,1),batch[:,1:].flatten())
            loss.backward();optimizer.step()
        train=measure(step,repeats)
        train["tokens_per_second"]=2*(length-1)/train["median_seconds"]
        with torch.no_grad(): prefill=measure(lambda:model(batch),repeats)
        tokenizer=ByteTokenizer();sample="AIM evidence α + β = γ.\n"*1000
        tokenization=measure(lambda:tokenizer.encode(sample),repeats)
        tokenization["bytes_per_second"]=len(sample.encode())/tokenization["median_seconds"]
        matrix=torch.randn(256,256)
        matmul=measure(lambda:matrix@matrix,repeats)
        checkpoint_path=run.path/"benchmark-checkpoint.pt"
        start=time.perf_counter()
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict()},checkpoint_path)
        checkpoint_seconds=time.perf_counter()-start
        reopened=torch.load(checkpoint_path,weights_only=True)
        checkpoint_ok=all(torch.equal(v,reopened["model"][k]) for k,v in model.state_dict().items())
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report={"scope":"this local host only; VIT cluster not accessed", "hostname":platform.node(),
                "host_cpu_report":command(["/usr/sbin/sysctl","-n","machdep.cpu.brand_string"]) if platform.system()=="Darwin" else command(["lscpu"]),
                "logical_cpus":os.cpu_count(),"process_threads":torch.get_num_threads(),"precision":"float32",
                "parameter_count":sum(p.numel() for p in model.parameters()),"batch_size":2,"sequence_length":length,
                "training_step":train,"prefill":prefill,"tokenizer":tokenization,"matrix_256":matmul,
                "checkpoint":{"seconds":checkpoint_seconds,"bytes":checkpoint_path.stat().st_size,"round_trip_equal":checkpoint_ok},
                "peak_rss_bytes":rss if platform.system()=="Darwin" else rss*1024,
                "network":None,"cluster_scaling":None,"accelerator_throughput":None,
                "note":"Short CPU FP32 fixture including optimizer step; no long-run/thermal/energy claims"}
        write_json(run.path/"metrics.json",report)
    return run.path,report
