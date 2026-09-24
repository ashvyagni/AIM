# VIT ML lab hardware audit

**Status:** project-reported fleet inventory, not an independent physical audit or performance benchmark. **Date recorded:** 24 September 2026.

## Inventory reported by AIM

The lab has approximately **70-84 machines** described as having the following per-machine specifications:

| Component | Per machine | Fleet-level interpretation | Audit state |
|---|---|---|---|
| CPU | 13th-generation Intel Core i9 | 70-84 CPUs; exact aggregate core/thread count cannot be established until exact SKUs are known | Generation/family stated; exact SKU, desktop/mobile variant, core topology, instruction support, clocks/power not confirmed |
| System memory | 32 GB RAM | **2.24-2.688 TB raw installed RAM** across 70-84 nodes | Capacity stated; whether each has 32 GB usable, speed/channels and reservation remain unverified |
| Graphics | Intel UHD Graphics 770 | Integrated graphics in each reported system; shares host memory, not 70-84 discrete GPUs with independent VRAM | Model stated; drivers, framework compute support and useful ML throughput untested |
| Count | 70-84 machines | Potential CPU cluster if simultaneous access and cluster policies permit | Range supplied by user; exact count and availability schedule unknown |
| Storage | Not stated | Aggregate storage cannot be calculated | Device, free capacity, local/shared filesystem, quotas and throughput unknown |
| Network | Not stated | Cluster scaling cannot be predicted | Topology, NIC/switch rates, oversubscription, latency, RDMA and routing unknown |
| Scheduler / cluster software | Not stated | A connected fleet is not necessarily a schedulable distributed cluster | OS, admin permission, scheduler, MPI/runtime and job limits unknown |

The 2.24-2.688 TB figure is a simple aggregate (`node count × 32 GB`) and **not** a unified memory pool. RAM is physically distributed. A distributed sharded training system can use aggregate memory only if the software, process layout, access rights and network support it; model states, activations, caches and communication buffers still have to fit per node. Data-parallel copies can replicate rather than pool model state.

The exact CPU SKU matters: 13th-generation Core i9 includes desktop and mobile variants with different core counts, power envelopes and memory behavior. Intel's [13th Gen desktop product brief](https://cdrdv2-public.intel.com/743780/13th-gen-desktop-processor-post-launch-product-brief.pdf) describes multiple variants; it does not identify the VIT SKU. Intel identifies UHD 770 as integrated Xe-based graphics, not a discrete datacenter accelerator with dedicated high-bandwidth VRAM ([Intel brief](https://cdrdv2-public.intel.com/743780/13th-gen-desktop-processor-post-launch-product-brief.pdf), [graphics API support](https://www.intel.com/content/www/us/en/support/articles/000005524/graphics.html)). Do not count each integrated GPU as a CUDA GPU or assume that it provides an efficient transformer training backend.

CPU distributed processes are technically available in standard software: PyTorch documents Gloo for CPU collectives, while Intel provides MPI/oneCCL libraries for compatible cluster communications ([PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed), [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html), [Intel oneCCL guide](https://cdrdv2-public.intel.com/833523/oneccl_developer-guide-reference_2021.14-772609-833523.pdf)). This establishes software options, **not** that the lab hosts have those packages installed, are interconnected, or can scale efficiently.

## Updated feasibility conclusion

This is potentially a substantial CPU fleet, not one workstation. It can make a meaningful AIM contribution through highly parallel data preparation, retrieval/indexing, simulation, symbolic and statistical verification, code tests, synthetic environment generation, benchmark execution, and independently batched CPU inference/rollout jobs. Those workloads often parallelize across independent tasks with less cross-node communication than model training.

For full model training the conclusion is more cautious. A 7B model's BF16 weights are about 14 GB per copy; an Adam-style training-state planning estimate is about 112 GB before activations/workspaces. This exceeds a node's 32 GB. Aggregate raw fleet RAM could hold sharded model states on paper, but CPU FSDP/ZeRO must repeatedly communicate shards; a high-latency or oversubscribed LAN can dominate step time. The fleet might run small distributed CPU training if interconnect and permission allow, but **the user’s node count and RAM alone do not establish a practical 7B training budget**.

### Illustrative 7B compute scenarios

At 7B parameters and 140B tokens, first-order dense pretraining compute is about `5.88e21 FLOPs` (`6ND`). To show sensitivity, the table assumes perfect linear scaling, 100% availability and an effective *end-to-end transformer training throughput per node* of 10 GFLOP/s, 100 GFLOP/s or 1 TFLOP/s. These are scenario inputs, **not benchmark results or claims about i9 performance**. Real wall time is longer after communication, thermal limits, scheduler downtime, checkpoints, validation and non-scaling work.

| Nodes | Assumed sustained useful throughput per node | Aggregate ideal throughput | Idealized 7B/140B-token duration |
|---:|---:|---:|---:|
| 70 | 10 GFLOP/s | 0.70 TFLOP/s | ~266 years |
| 84 | 10 GFLOP/s | 0.84 TFLOP/s | ~222 years |
| 70 | 100 GFLOP/s | 7.0 TFLOP/s | ~26.6 years |
| 84 | 100 GFLOP/s | 8.4 TFLOP/s | ~22.2 years |
| 70 | 1 TFLOP/s | 70 TFLOP/s | ~2.66 years |
| 84 | 1 TFLOP/s | 84 TFLOP/s | ~2.22 years |

Even the 1 TFLOP/s-per-node row is only a hypothetical sensitivity point until a representative full training benchmark demonstrates it. The system is likely valuable for AIM, but the first scientific scale target should be selected from measured tokens/second, not this sensitivity table.

## Likely cluster topology and work allocation

If administrators permit, use an initial CPU cluster as a **parallel research/evaluation platform**. Keep the evidence database, object storage and scheduler centralized or appropriately replicated; run independent worker jobs across nodes. First compare: (A) embarrassingly parallel jobs such as parsers/verifiers; (B) multi-process CPU inference; (C) all-reduce for a tiny training proxy; and (D) FSDP/ZeRO-sharded training only if C is healthy. Start with PyTorch Gloo or supported MPI/oneCCL only after confirming the installed operating system/runtime and fabric. Do not select MoE: all-to-all expert routing is especially sensitive to bandwidth and latency, while there are no dedicated GPU accelerators stated.

Recommended usage order:

1. Data ingestion, rights manifests, deduplication, paper parsing, tokenizer experiments and retrieval-index construction.
2. CPU verifiers: SymPy, compilers/tests, formal methods, statistics and simulations, with sandbox and resource limits.
3. Synthetic research worlds, hidden-test generation, evaluation, adversarial red-teaming and human-label triage.
4. Batch inference/rollouts and small 100M-300M training proxies, measuring throughput and useful concurrency.
5. Distributed 1B experiments only if a 24-hour measured run is stable, within memory and useful budget.
6. Treat a 7B CPU cluster run as a research systems experiment only after measured cost/speed indicates meaningful progress; seek accelerator allocation for serious 7B+ base pretraining.

## Required audit and benchmark

1. Inventory every node class: count actually schedulable together, exact CPU SKU, P/E cores, threads, SIMD/AMX flags, NUMA, sustained clock, memory speed/channels, storage and OS.
2. Record admin permissions, scheduler, queue/usage policies, user quotas, container support, job preemption/uptime, simultaneous node availability and security boundaries.
3. Measure network topology and per-node NICs; test point-to-point bandwidth/latency, all-reduce and all-gather for realistic payload sizes, first within rack and then across racks. Record p50/p95 under competing load. Verify whether MPI/oneCCL/Gloo can use the fabric.
4. Test CPU tokenization/decompression, preprocessing, indexing and verifier throughput. These may deliver high immediate value independent of network.
5. Run full forward/backward/optimizer benchmark at 100M/300M/1B, 1K/2K/4K sequences, conservative microbatches and supported precision. Capture tokens/s, step-time distribution, resident memory, energy/thermal throttling, data starvation and checkpoint/restart. Repeat at least three times and a representative 24-hour job.
6. Test inference at batch 1 and parallel batch, prompt/decode split, quantization, context and failure recovery. Measure each node and fleet throughput.
7. Test model sharding only after collectives pass. Report scaling efficiency `S_n/(n S_1)`, memory per rank and end-to-end tokens/s; compare with a single-node baseline.
8. Capture storage read/write, checkpoint duration, shared-filesystem contention, scratch quotas and artifact backup.

## Scale gate and recommendation

Current recommendation: **treat 70-84 systems as a potentially valuable CPU cluster whose first proven value will likely be data, verifiers, simulation, search, batch evaluation and small proxy work.** Use 100M-300M as the first training proxies; profile before 1B. Do not claim 7B training is ruled out by aggregate RAM alone, but do not promise it either: if nodes can be joined, FSDP can shard state in principle; the compute scenarios show that useful training throughput, fabric and allocation determine whether it is practical. For substantial base training, accelerator access remains the recommended route. The cluster may still support a novel, useful research-loop/RL evaluation program without full-scale pretraining.

Unknown after the user's update: exact model/SKU count, whether the 70-84 hosts share 32 GB/i9/UHD 770 specs, OS homogeneity, availability at the same time, network, scheduler, storage and benchmark throughput. Confirm each before final model-scale commitment.

## Local Codex host distinction

The Codex host separately reported Apple M3 integrated GPU (10 cores) via `system_profiler`; CPU/RAM `sysctl` queries were restricted. It is not one of the reported VIT machines and was not included in node or throughput estimates.
