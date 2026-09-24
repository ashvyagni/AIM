# Scaling and compute analysis

## Scope and assumptions

The user reports approximately 70–84 VIT machine-learning-lab systems, each described as having a 13th-generation Intel Core i9, 32 GB RAM and Intel UHD Graphics 770. If all are simultaneously accessible and consistent, this corresponds to 2.24–2.688 TB of aggregate installed RAM across separate hosts, not one shared memory pool. Exact count/SKUs, per-node consistency, network, scheduler, storage, access policy, uptime and measured cluster throughput are unverified. The system may be joinable as a CPU cluster; this is a hypothesis to test, not a confirmed capability. All throughput values below are transparent planning scenarios, not measurements. Dense training first-order compute is `F ≈ 6 N D` (parameters × training tokens × 6 FLOPs), excluding attention overhead nuances, optimizer updates, failed runs, post-training and data/tool costs. It is an order-of-magnitude comparison, not a performance model. Chinchilla’s reported optimum is not a universal ratio; modern repeated-token training, quality, inference demand and data limits shift it.

## Training compute scenarios

| Model | Illustrative tokens | `6ND` FLOPs | Approx. wall time at 1 PFLOP/s sustained | 100 TFLOP/s | 10 TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| Dense 7B | 140B (20 tok/param) | 5.88×10²¹ | 68 days | 1.87 years | 18.7 years |
| Dense 14B | 280B | 2.35×10²² | 272 days | 7.46 years | 74.6 years |
| Dense 32B | 640B | 1.23×10²³ | 3.90 years | 39.0 years | 390 years |
| Dense 70B | 1.4T | 5.88×10²³ | 18.7 years | 187 years | 1,870 years |
| MoE 100B total / 20B active | 400B | 4.80×10²² approximate active FLOPs | 1.52 years | 15.2 years | 152 years |
| MoE 300B total / 32B active | 1T | 1.92×10²³ approximate active FLOPs | 6.09 years | 60.9 years | 609 years |

At fixed sustained throughput, time scales linearly with FLOPs. 1 PFLOP/s sustained is a powerful accelerator-cluster scenario, not an i9 estimate. At 1 TFLOP/s per node, 1 PFLOP/s requires 1,000 nodes *at sustained useful training throughput*; at 100 GFLOP/s/node it requires 10,000. A CPU node's actual transformer throughput depends on SIMD/precision support, memory bandwidth, kernel, sequence/batch and thermal/power scheduling. Measure it; theoretical peak is not sustained useful training throughput.

Token sensitivity is linear: halve/double tokens, halve/double estimated FLOPs. If quality corpora support only 50B unique tokens, a 7B model at 50B tokens costs 2.1×10²¹ FLOPs (about 24 days at 1 PF/s) but is 7 tokens/parameter. Repeating data may be useful but has diminishing returns and overfit risk; do not pretend it equals fresh tokens.

## Memory estimates

BF16 weights: `2N` bytes (7B ≈14 GB, 14B ≈28 GB, 32B ≈64 GB, 70B ≈140 GB; 100B–200B MoE ≈200–400 GB, 300B–400B ≈600–800 GB). This excludes buffers, fragmentation and quantization metadata.

Adam-style mixed-precision training commonly requires roughly 12–18 bytes/parameter for weights, gradients, FP32 master copy and two FP32 moments depending on implementation/state dtypes. Use a conservative 16 B/parameter planning point: 7B≈112 GB; 14B≈224 GB; 32B≈512 GB; 70B≈1.12 TB; 300B MoE≈4.8 TB. Sharding distributes state but does not remove total bytes; activations, attention workspace, communication buffers and checkpoint staging are additional. MoE full weights/states follow total parameters; compute approximation follows active parameters plus routing/attention/shared layers.

KV cache per sequence for standard K/V is roughly `2 × layers × seq_len × n_kv_heads × head_dim × bytes`, excluding allocator overhead. For example, 32 layers, 8 KV heads, head dimension 128, BF16, 32K tokens gives about 4 GiB per sequence. GQA reduces this proportional to KV heads; MLA changes formula, so profile implementation. Longer context can dominate serving memory even where weights fit.

Checkpoint size: BF16 model weights 2N bytes; full optimizer checkpoint can be ~16N bytes before compression and metadata. Keep at least one immutable base, resumable latest, and milestone checkpoints; asynchronous save and quota must be measured. A 300B model’s BF16 weights alone are 600GB; full optimizer state is about 4.8TB. Replicas/retention multiply requirements.

## Hardware decision thresholds

Measure sustained end-to-end tokens/s and FLOPs/s at intended context, not GEMM peak only. Include dataloading, checkpointing, validation, outages and utilization. Budget project time with effective throughput `P_eff = P_peak × utilization × kernel efficiency × availability`; each factor should be measured or explicitly scenario-bounded. Estimate compute with 0.2/0.4/0.6 utilization cases until profiling supplies data. Inter-node collectives must be benchmarked for all-reduce and MoE all-to-all; count bytes/rank and topology oversubscription.

## Recommendation

No final scale is authorized yet. Treat the reported 70–84-node fleet as a potentially useful CPU cluster, subject to access and benchmark gates. Start with 100M–300M single-node CPU proxies and parallelize data, verifiers, indexing, simulation, evaluation and rollouts across available nodes. Profile single-node and multi-node 1B runs; then decide whether a sharded 7B pilot has meaningful wall-clock value. Aggregate RAM could accommodate sharded state in principle (not as one pool), but does not establish practical training throughput. A 7B full run is conditional on measured end-to-end performance, network, schedule, data and budget; accelerators remain the preferred route for substantial pretraining. Consider 14B/32B and MoE only after scaling curves and full lifecycle resources justify them. Gate decisions on the audit in `../hardware/vit-ml-lab-audit.md`.
