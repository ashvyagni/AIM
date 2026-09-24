# AIM research program

**Evidence cutoff:** 24 September 2026. **Status:** research architecture proposal; not a claim that the VIT cluster has been audited or that a final model size is authorized.

This repository separates the user’s request (produce a reproducible, sourced research package) from the attached Charter (scientific vision and hypotheses) and Astra Directive (operational protocol and requested deliverables). Those attachments are project requirements, not evidence that their architectural propositions are correct.

## Decision summary

Build and compare a dense decoder Transformer first. The user reports approximately 70–84 VIT lab systems, each described as 13th-gen Intel Core i9, 32 GB RAM and UHD Graphics 770. If consistent and concurrently available, that is 2.24–2.688 TB aggregate RAM across separate hosts and a potential CPU cluster, not shared memory. Node count/spec consistency, exact SKUs, network, scheduler, storage, policies and sustained throughput are unverified. Use 100M–300M single-node CPU proxies first; measure a distributed 1B pilot and cluster workloads. A sharded 7B run is neither ruled out by aggregate memory alone nor justified by it: select only after a measured end-to-end pilot and schedule/budget analysis. Use accelerators if accessible for serious pretraining. The supplied inventory is a report, not an independent audit. The local Codex host's Apple M3 observation is unrelated to VIT.

Build the initial research system around an existing openly available checkpoint as a systems baseline where its license permits, plus an independently trainable 100M–1B model for controlled learning experiments. The contribution should be a verified research loop and a falsifiable training study, not parameter count.

Use one autoregressive researcher with structured tool contracts, deterministic provenance storage, independent domain verifiers, and initially a small separate correctness/decision model. Treat a shared decision head as an ablation. RLHF, RLVR, and calibrated decisions serve different tasks and should not be naively summed: start with SFT, preference learning, isolated verifier-based RL, and a separately scored decision policy; integrate only after factorial ablations.

## Document map

- `literature/literature-matrix.md` — source index and evidence status.
- `architecture/state-of-the-art-architecture-review.md` — alternatives and proposed model family.
- `rl/rlhf-rlvr-rlcd-review.md` — objective mathematics, RLCD evidence limits, sequencing.
- `rl/reward-and-verifier-design.md` — verifier ensemble and anti-gaming controls.
- `data/data-strategy.md` — corpus tiers, licenses, provenance and synthetic data.
- `scaling/scaling-and-compute-analysis.md` — transparent FLOP and memory scenarios.
- `hardware/vit-ml-lab-audit.md` — what is known, what is not, and an executable audit protocol.
- `agentic-research/research-loop-design.md` — controller, state and evidence graph.
- `evaluation/evaluation-and-benchmark-plan.md` — metrics, splits and gates.
- `open-questions/open-research-questions.md` — explicit hypotheses and deciding experiments.
- `documents/final-design/` — architecture, training, evaluation and implementation specifications plus decision record.
- `documents/AIM_Master_Research_Architecture.md` — consolidated master artifact.

## Epistemic labels

- **Established evidence:** directly reported in a primary paper, official technical report, implementation, or documentation; does not mean universal truth.
- **Recommendation:** reasoned design choice supported by evidence and project constraints.
- **Hypothesis:** testable claim not established for AIM.
- **Speculation:** plausible idea with weak direct evidence.
- **Unknown:** necessary measurement or source disclosure is absent.

## Reproducibility rule

Record exact source URL, version/date, license, retrieval time, transformations, hashes, benchmark split, software versions, random seeds, hardware, tokens, optimizer, run duration, and checkpoints. Freeze a held-out evaluation set before training. Archive failed runs and negative results. Revisit any current model or service specification against its primary source before implementation.
