# State-of-the-art architecture review

## Decision context

The objective is an auditable scientific workflow, not an ungrounded “scientist model.” The literature supports decoder-only Transformers as the lowest-risk base family; MoE is an option when accelerator memory and network can support sparse routing. Alternatives such as state-space/hybrid models are worth benchmarked small-scale experiments but not a first-system dependency because tooling and transfer evidence for research workflows are less mature than the Transformer ecosystem.

## Alternatives

| Option | Advantages | Costs / failure modes | AIM decision |
|---|---|---|---|
| Dense decoder, shared researcher and decision representations | Simplest training, serving and debugging; one model can emit text and structured JSON; mature kernels/checkpoints. | All weights active each token; multiobjective gradient conflict; verbal confidence may be poorly calibrated. | Required 7B-class reference when compute exists; small proxy now. Add a trained decision head only as controlled variant. |
| Single backbone + dedicated typed decision head | Shared knowledge; low extra inference latency; schema-constrained outputs. | Head inherits representation errors; calibration can be domain-shifted; generation objective may conflict with decision scores. | Ablation after baseline; score head separately with proper loss. |
| Researcher + separate calibrated judge | Independent objective, data and update cadence; can score claims/route actions without rewriting prose. | Extra serving cost; judge can share blind spots; disagreement and correlated errors; stale-model coordination. | Best first integrated architecture, using small judge and verifier ensemble; require independent held-out correctness. |
| Researcher MoE + judge + deterministic verifier/controller | Sparse active compute can expand capacity; clear role boundaries; objective tools can certify narrow outputs. | Full expert memory and optimizer state; expert all-to-all; routing collapse/imbalance; operational complexity. | Future GPU research path only after 7B dense and a measured MoE proxy. No domain-pinned experts initially. |
| Retrieval-augmented existing model, no new pretraining | Fastest useful system, fresh evidence and lower compute; tests whether loop/value comes from tooling. | Capability ceiling and license/availability dependence; no base-model contribution. | Must be a systems baseline. Research novelty can be workflow/training even if initial researcher is a licensed open model. |

## Proposed model family, conditional on accelerator access

Use a conventional pre-norm decoder Transformer, RMSNorm, SwiGLU MLP, RoPE, GQA and BF16 (or hardware-supported FP8 after validated numerics). These are evidence-backed engineering defaults, not a claim they are uniquely optimal. Begin with 32 layers, hidden dimension 4096, 32 query heads, 8 KV heads, head dimension 128, and SwiGLU intermediate width chosen to land near 7B parameters; parameter count must be computed from the actual tokenizer vocab, embedding tying, bias policy, and exact layer spec before run configuration. A 7B-equivalent spec is an experiment target, not selected VIT scale. 14B/32B/70B require separate budget and scaling evidence.

Context plan: train robustly at 4K–8K first, extend to 16K–32K only with long-sequence data and memory profiling, and serve longer evidence through retrieval/chunking rather than forcing every request into a maximum context. Use RoPE base/position settings with length curriculum; test native training versus YaRN-style extension on retrieval, positional extrapolation, and long-document synthesis. GQA reduces KV heads/cache but is not free: memory scales approximately with layers × sequence × KV heads × head dimension × bytes × 2 for K and V per request.

Tokenizer: compare a modern byte-fallback BPE/unigram tokenizer trained on licensed, domain-balanced text. Include scientific notation, Unicode math, LaTeX, code, tables, biomedical terms and multilingual samples. Evaluate compression by domain, round-trip loss for symbols, segmentation of rare equations, and tokenization fairness; freeze tokenizer before expensive pretraining. Avoid claiming a tokenizer is universally optimal.

MoE decision: only consider after dense scaling proxy says capacity is limiting rather than data, data quality or post-training. Prototype modest 8–16 experts/layer, top-2 routing with capacity and explicit load-balance monitoring; compare Switch-style coarse experts with fine-grained/shared-expert DeepSeekMoE design at equal active FLOPs. Keep experts data-driven; evaluate specialization and cross-domain transfer before domain labels are used in routing. MoE uses active parameters for approximate per-token dense FLOPs but *total* expert parameters for weight residency, optimizer and checkpoint storage; token dispatch requires all-to-all. Dense is preferred on weak/intermittent networks.

Position and attention: RoPE is a strong, portable baseline; GQA is a cache optimization with quality tradeoffs to measure. FlashAttention-2 is exact attention with IO-aware kernels, but only where supported by the device stack. MLA can reduce KV memory, as reported by DeepSeek-V2/V3, but adds architectural and kernel complexity; benchmark at matched quality before considering. Do not assume NoPE, linear attention, state-space or hybrid block designs beat a plain Transformer for AIM: benchmark one representative alternative only if the dense control is bottlenecked by sequence or memory.

Objective/hardware: next-token cross entropy on packed documents with document boundaries and loss masks; AdamW baseline; compare Muon on a small preregistered scaling slice. BF16 baseline where accelerator supports it. Use activation checkpointing, fused kernels, async data prefetch, and sharded optimizer/model state when GPU distributed training is selected. Choose TP/PP/DP/EP from topology measurements; high-latency Ethernet is not equivalent to NVLink/InfiniBand.

## Evidence and tradeoffs

DeepSeek-V3 is a concrete sparse reference (671B total, 37B activated, 14.8T training tokens) but trained on dedicated H800 infrastructure; it demonstrates feasibility for a well-resourced organization, not feasibility on the reported VIT CPU fleet absent measured throughput and interconnect benchmarks. Qwen3 spans dense and MoE and supports thinking/non-thinking modes. GLM-4.5 reports 355B total / 32B active. These reports support architectural plausibility, not an AIM scale choice. DeepSeekMoE's expert splitting/shared experts show one viable load/specialization technique. Muon reports improved compute efficiency in its own experiments; AdamW remains the conservative control.

## Parameter-scale gate

Measure tokens/s, sustained training FLOPs/s, memory headroom, all-reduce and all-to-all first. Then fit loss-vs-compute scaling proxies across at least three model sizes and several token budgets. If the funded compute cannot complete a credible 7B training run on licensed tokens, treat 7B as a target only if accelerator access is secured; do not train a nominal 32B underfit artifact for marketing. The immediate low-risk build is retrieval/system prototype plus 100M–300M single-node CPU proxies, while using the reported fleet for parallel data/verifier/evaluation workloads. A distributed 1B pilot is a benchmark gate, not a presumed outcome. For funding: solicit a GPU allocation or cloud grant after reproducible throughput data exists.

## Tests that could change this recommendation

1. Dense versus MoE proxy at matched active FLOPs and data: held-out loss, domain transfer, load balance, wall-clock, energy and serving cost.
2. GQA/MLA variants at equal training compute: long-context accuracy and cache bytes/token.
3. Native long-context training versus retrieval: evidence recall and supported synthesis at same GPU-hours.
4. Shared decision head versus separate judge: Brier/log score and selective risk at fixed cost.
5. AdamW versus Muon: tokens-to-target validation loss, sensitivity and restart reproducibility.
