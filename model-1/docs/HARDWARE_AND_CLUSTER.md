# Hardware and cluster engineering protocol

## Known versus reported

**Reported by project owner:** 70–84 lab desktops, each with 32 GB RAM, a 13th-generation Intel i9 and Intel UHD Graphics 770. Exact CPU SKU, core counts, usable RAM, OS, disk, network fabric, scheduling rights, simultaneous availability and sustained throughput have not been audited here. The delivered audit JSON is an unfilled template, not a measurement.

70×32 to 84×32 gives **2,240–2,688 nominal GB across the fleet**. That aggregate is not one address space. Standard DDP replicates the model and optimizer on each rank; it does not combine node memory to fit a larger model. Intel integrated graphics are not treated as validated training accelerators by this implementation. No supported precision or sustained GPU rate has been measured for them.

**Measured in this phase:** a macOS arm64 host reporting 8 logical CPUs, using FP32 CPU kernels and one torch thread per process. CPU marketing-name lookup was unavailable in the restricted benchmark, and is recorded null. The host is not claimed to match any VIT desktop. Timings cannot be extrapolated to the lab.

## Implemented benchmark commands

From `model-1/`:

```sh
.venv/bin/python -m aim benchmark
.venv/bin/python -m aim.distributed_launch --ranks 1
.venv/bin/python -m aim.distributed_launch --ranks 2
```

The CPU benchmark includes two warmup iterations and five measured samples for a 90,624-parameter model, batch 2, sequence length 64. It measures a complete training update, no-gradient prefill, byte tokenization, 256×256 matrix multiplication, checkpoint write/load equality and peak RSS. The checkpoint write time is not a durable-disk/fsync or filesystem bandwidth guarantee.

The local DDP benchmark uses the same model, batch 2 per rank, sequence length 32, two warmups and five samples. It measures max-rank update time, verifies all-reduce values, and checks exact replica parameter hashes after updates. Its all-reduce payload is 65,536 FP32 elements (262,144 bytes). The local launcher uses IPv4 loopback, explicit timeout, log retention and process-group cleanup; it cannot contact lab nodes.

Use `aim.distributed` directly under an authorized multi-host launcher for actual lab measurements. PyTorch documents Gloo/distributed configuration and backend capabilities in its [version 2.8 distributed documentation](https://docs.pytorch.org/docs/2.8/distributed.html). No accelerator backend is selected here.

Example **template**, not executed on lab machines:

```sh
# Supply actual authorized interface, coordinator IP, rank and port on EACH host.
GLOO_SOCKET_IFNAME=<lab-interface> OMP_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=1 --node-rank=<0-or-1> \
  --master-addr=<authorized-coordinator-ip> --master-port=<approved-port> \
  --rdzv-conf=timeout=30 -m aim.distributed --steps 20 --elements 4194304
```

Run with a supervising job deadline and identical source/environment on both hosts. Do not use the local launcher's loopback interface override for a multi-host run. This prototype has no cluster discovery, SSH deployment, scheduler integration, authentication service, elastic recovery or multi-host checkpoint sharding.

## Lab audit sequence

1. Record actual node inventory and job/network permissions in a copied audit file. Preserve reported versus measured fields. Use anonymized node identifiers for shared reports.
2. Validate one CPU host's environment, memory ceiling, disk availability, numerical checks and complete training update. Preserve failed installations and unsupported operations.
3. Sweep thread affinity/count and realistic sequence/batch sizes in new bounded configs; measure warm and sustained throughput, memory, power/thermal throttling and foreground interference. Current hardcoded microbenchmark settings are not those full sweeps.
4. Measure pairwise and cross-switch network latency and useful throughput. Measure collective payloads spanning anticipated gradient bucket sizes, not just the included small fixture.
5. Repeat DDP at 1, 2, 4 and 8 physical nodes before increasing fleet size. Match per-rank workload and source/configuration; report global batch changes explicitly. Use maximum rank time and include stragglers and failed runs.
6. Measure checkpoint size, write/read times, restart behavior and storage saturation on the intended filesystem. Run authorized node-loss/recovery tests in a separate experiment.
7. Produce an owner-reviewed scale recommendation using the measured compute/storage/network budget and corpus plan.

Weak-scaling efficiency at fixed per-rank work may be summarized as E(N)=throughput(N)/(N×throughput(1)), with matching model/context/precision/thread settings. Local multiple-process results share the same physical CPU and are not physical-node scaling results. Report strong-scaling tests separately at fixed global batch/workload.

## Parameter accounting and memory floors

For this tied-embedding dense design, let V=vocabulary, d=width, L=layers, f=FFN width and k=KV projection width. Exact trainable parameter count is:

\[
N=Vd+d+L(2d^2+2dk+3df+2d).
\]

The following are **analytical planning candidates**, not allocated or trained. All parameters are active; none is MoE. The actual count is more useful than its rounded family label.

| Planning family | Exact parameters | FP32 Adam parameter/gradient/moment floor, decimal GB |
|---|---:|---:|
| 100M class | 100,682,496 | 1.61 |
| 300M class | 304,137,216 | 4.87 |
| ~1B class | 864,626,688 | 13.83 |
| 7B class | 7,113,805,824 | 113.82 |

The floor assumes weights 4N + gradients 4N + two optimizer moments 8N = 16N bytes. It excludes activations, attention buffers, allocator/workspace, reference models, runtime, data, checkpoints and OS needs. An FP32 frozen DPO/RLVR reference adds approximately 4N bytes. Therefore a 32 GB desktop's ability to hold weights does not establish the feasibility or speed of a full training stage.

The smoke training harness caps model allocation at 2M parameters. A reviewed scale-trial runner must replace that guard deliberately, with memory estimation and an explicit time/token budget; do not merely edit the integer and launch 1B.

Sharding or pipeline/model parallelism would be necessary for some candidates on 32 GB hosts, but introduce substantial communication and failure-recovery work. FSDP/ZeRO, CPU offload, mixed precision and MoE are controlled future experiments, not implemented capabilities. Current GQA repeats KV tensors, preserving the small model's architecture but not optimizing the CPU memory footprint of attention.

## Scale gates

- **100M–300M proxy:** licensed corpus and tokenizer, validated streaming/checkpoint-resume, memory and sustained tokens/sec evidence, approved short training budget.
- **~1B distributed experiment:** demonstrated physical-node efficiency, acceptable checkpoint recovery and total wall-time estimate. It remains a systems experiment unless meaningful data/training quality is available.
- **7B+:** validated infrastructure plus sufficient tokens, human/data evaluation capacity, storage and wall-clock budget. Fleet size alone is insufficient evidence.
- **14B/32B/MoE:** separate architecture records and measured advantage over dense baselines, including active/total parameters and communication costs.

For a target of D processed tokens and measured sustained aggregate throughput R, the ideal active processing time is D/R. Add measured downtime, evaluation, checkpoint and startup overhead separately. Do not estimate R from desktop core counts, nominal FLOPS, or this Mac's microbenchmark.
