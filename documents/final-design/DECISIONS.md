# AIM architecture decision record

**Date:** 24 September 2026. **Status:** provisional pending VIT hardware audit, rights review and ablations.

| ID | Decision | Alternatives considered | Rationale / evidence | Reversal condition |
|---|---|---|---|---|
| D-001 | No final model size selected. User reports ~70–84 VIT lab machines, each described as 13th-gen Intel Core i9, 32 GB RAM, UHD 770. Potential 2.24–2.688 TB aggregate RAM is distributed across hosts, not pooled; cluster connectivity and performance remain unverified. Begin with 100M–300M CPU proxies; benchmark distributed 1B and gate any 7B pilot on measured throughput/schedule. | 7B now; 14–32B; 70B; 100–400B MoE. | User-reported inventory and first-order state/compute arithmetic; exact SKUs, homogeneity, availability, network, scheduler and throughput unknown. Chinchilla is an empirical reference, not a universal rule. | Hardware/data/budget audit and scaling-law experiments select viable scale. |
| D-002 | Dense decoder Transformer is first architecture control. | MoE, SSM/hybrid, external checkpoint only. | Mature toolchain and simplest causal comparison; modern sparse papers establish possibility but not local feasibility. | Dense bottleneck evidence plus MoE end-to-end gain at matched quality/cost. |
| D-003 | Research system uses deterministic controller/provenance shell, independent verifiers, researcher model, small separate judge first. | Single prompt transcript; shared decision head; full MoE system. | Separation makes correctness/provenance auditable; RLCD public evidence is insufficient for adopting a proprietary method. | Shared-head ablation matches separate judge in calibration, shift robustness and cost. |
| D-004 | Use DPO as preference baseline, PPO/GRPO only in well-specified RL environments; no arbitrary sum of RLHF/RLVR/RLCD. | Weighted scalar reward; sequential/alternating; constrained objective. | Signals are noncommensurate; reward model overoptimization documented; verifiers offer objective islands. | Controlled experiments justify a scalarization or joint update with no guardrail regressions. |
| D-005 | Proper scoring (Brier/log) and selective risk/coverage measure decision calibration; ECE diagnostic only. | Accuracy/confidence instruction; ECE-only. | Proper scoring rules elicit probability forecasts; ECE depends on bins and is not proper. | Domain-specific score analysis suggests a better predeclared decision loss. |
| D-006 | Outcome verifier is primary correctness reward; process reward bounded/auxiliary and adversarially tested. | Pure PRM; no intermediate signal. | PRM evidence is promising in math but narrow; process can reward style and exploit. | Transfer studies show process signal robustly improves held-out verifiable outcomes. |
| D-007 | Retrieval-first evidence access with staged context length. | Long-context-only; all-source prompt. | Context length is not evidence quality; retrieval maintains provenance and reduces KV cost. | Matched task tests show native long context materially wins at acceptable cost. |
| D-008 | Treat user-reported 70–84 VIT machines as preliminary inventory for a potential CPU cluster; continue the audit before scale commitment. | Treat aggregate RAM as a shared pool or infer training throughput from node count. | Per-node specs/count reported by user; topology, concurrency, access, exact SKU, storage, scheduler and measurements absent. | Replace assumptions with measured documented inventory and scaling results. |

## Rejected assumptions

- “More parameters means more research intelligence.” Not established; quality, tokens, feedback, verifiers and inference budget interact.
- “RLHF + RLVR + RLCD should be three weighted rewards.” No evidence for universal weights and scalarization can hide unacceptable tradeoffs.
- “Typed outputs cannot hallucinate.” Only malformed/out-of-schema outputs can be constrained; wrong decisions remain possible.
- “Open-access papers can be used for training.” Rights vary per asset and license; metadata and full text differ.
- “MoE active parameter count describes memory.” Total weights/optimizer/checkpoint plus communication are critical.
- “The project contribution is Research-Loop RL.” Novelty is speculative until ablation and prior-art review demonstrate a gap.
