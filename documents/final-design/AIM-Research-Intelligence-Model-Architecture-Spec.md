# AIM Research Intelligence Model — Architecture Specification

**Version:** 0.1 research proposal, 24 September 2026  
**Decision status:** conditional; preliminary workstation specification supplied, full VIT inventory and benchmarks still pending.  
**Source documents:** Astra Master Research Directive v0.1 and Research Intelligence Model Charter v0.1, treated as project vision and hypotheses.

## 1. Proposed system

AIM is a system composed of (1) an autoregressive Researcher that proposes claims, hypotheses, derivations and actions; (2) a typed decision/judge component for route/continue/retry/abstain and evidence-status decisions; (3) deterministic and learned verifier ensemble; (4) a controller that maintains append-only research state; (5) provenance/evidence storage and retrieval; and (6) human review for high-impact or unresolvable claims.

Recommendation is **Variant C at the system boundary, not necessarily a giant MoE neural network**: researcher + separate small judge + deterministic shell/verifiers. Compare shared-head Variant A and researcher+judge Variant B. Choose the simplest system that wins trajectory ablations.

## 2. Model specification and unresolved scale

### Current scale decision

The user reports approximately 70–84 VIT machine-learning-lab systems, each described as a 13th-generation Intel Core i9 with 32 GB RAM and Intel UHD Graphics 770. This remains preliminary inventory, not an independent audit or benchmark. The conditional aggregate is 2.24–2.688 TB RAM across separate hosts, not a shared pool. Exact node consistency/SKUs, network, scheduler, storage, access and throughput are unknown; whether this can be operated as a cluster must be demonstrated. Begin with 100M–300M single-node CPU proxies and parallel cluster workloads; benchmark multi-node scaling at 1B and evaluate a sharded 7B pilot only if measured throughput supports a useful schedule. Distributed sharding may address model-state capacity, but communication and CPU compute may dominate. No final parameter count exists today. Accelerator access is preferred for substantial pretraining. Keep 14B/32B and 100B–400B MoE as options, not commitments.

### Conditional 7B-family starting configuration

Use a decoder-only pre-norm Transformer. A provisional design point for profiling is 32 layers, width 4096, 32 query heads, 8 KV heads, head dimension 128, SwiGLU FFN, RMSNorm, RoPE, tied embeddings only if measured quality permits, no biases except where required. Derive exact parameter count from implemented config/vocabulary; this rough shape is near 7B but not guaranteed exactly. Vocabulary starting range 32K–64K after domain tokenizer study. Context 4K–8K for base; 16K/32K adaptation only after memory/retrieval comparisons.

Dense baseline first. MoE experiment only with accelerator fabric: 8–16 experts per sparse layer, top-2 routing as a controlled baseline; compare coarse versus fine-grained/shared-expert design at equal active parameter FLOPs. Explicit expert count/layer pattern and total/active params must be chosen from throughput profiling. No domain-pinned experts in first run.

### Alternatives and reasons

- 14B/32B dense: stronger capacity, but requires proportionally larger training compute/data and memory. Defer until throughput/scaling curves justify it.
- 70B dense: scale reference, not credible without substantial GPU allocation and >1T tokens for a Chinchilla-like regime.
- 100–400B MoE: high total storage/optimizer and communication cost despite active FLOPs; research value does not justify the network risk before a small MoE proxy.
- Existing open checkpoint + tools: fastest capability baseline, dependent on exact license and model disclosure; essential system baseline, not substitute for trainable proxies.
- State-space/hybrid: retain as a targeted long-sequence experiment only if Transformer attention is measured bottleneck.

## 3. Core model choices

| Component | Baseline | Alternatives to test |
|---|---|---|
| Attention | GQA + RoPE; exact attention kernels where supported | MHA/MQA; MLA; YaRN extension; one state-space/hybrid proxy if warranted |
| MLP | SwiGLU dense | MoE with measured expert routing |
| Norm | RMSNorm | LayerNorm if existing code/reference requires |
| Precision | BF16 on validated GPU | FP32 CPU proxy; FP8 only after numerical validation on compatible accelerators |
| Optimizer | AdamW reference | Muon in same-compute controlled trial |
| Token objective | Causal next-token cross-entropy, masks and document boundary handling | Auxiliary multi-token prediction only as ablation |
| Context | 4–8K base, retrieve evidence for larger corpus | 16–32K staged extension, longer only after measured need |

## 4. Model system boundaries

Researcher emits structured claim/action records in addition to human-facing prose. Decision model returns a schema-constrained categorical distribution and action confidence. It has no authority to assert fact absent evidence. Verifiers return versioned results and raw artifacts. Controller owns state transitions, budgets, permissions, event log and evidence graph. Final writer renders a report from verified/qualified claim objects and preserves citations.

## 5. Memory/data interfaces

Relational store for papers, versions, claims, evidence spans, hypotheses, tests, results, forecasts, source licenses and events; object store for originals and run artifacts; graph relations for supports/refutes/derives/cites/supersedes; vector + lexical indexes for retrieval candidates. Every source includes URL/DOI, version, retrieval time, hash, rights and parser. Every claim includes scope, epistemic state, source spans/derivation, verification method, confidence target and time. Never conflate model confidence, evidence confidence and verifier state.

## 6. Inference policy

Plan → retrieve primary evidence → extract claims → generate alternative hypotheses/predictions → calculate/test/prove → verify → revise state → write. Use bounded best-of-N only where verifier can rank candidates and total cost is capped. Self-consistency is a sampling heuristic, not evidence. Invoke human review for high consequence, contradictory sources, missing license/authority, suspected verifier exploit or confidence OOD. Apply stopping via expected information gain vs cost, hard resource limits and explicit unresolved list.

## 7. Distributed training and storage

On GPU cluster, begin with data parallel + FSDP/ZeRO sharding for dense; add tensor/pipeline parallel only when single-node memory or scaling requires. Context parallel for long sequences after baseline. Expert parallel only for MoE and benchmark all-to-all. Topology must guide mesh. Save sharded resumable checkpoints and periodic consolidated export; verify restart. CPU nodes handle corpus processing, indexing, simulations, verifier execution and evaluation where cost-effective.

Planning memory at 16 bytes/parameter for Adam-like state: 7B ≈112 GB aggregate before activation/workspace; 14B ≈224 GB; 32B ≈512 GB; 70B ≈1.12 TB. BF16 weights: 14/28/64/140 GB respectively. Full model/optimizer storage scales with total parameters for MoE. Measure actual backend.

## 8. Acceptance criteria

No scale commitment without measured compute and licensed-data budget. No model architecture claim without equal-compute comparison. No scientific reliability claim without held-out task-level evaluation and verifier audit. See training/evaluation specs and `research/scaling/scaling-and-compute-analysis.md`.
