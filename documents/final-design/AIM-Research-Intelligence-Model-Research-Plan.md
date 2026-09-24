# AIM Research Intelligence Model — Research Plan

## Purpose

Determine whether a system that combines generative research, explicit evidence state, verification, and calibrated action selection improves measurable research outcomes under a realistic compute budget. The deliverable is a falsifiable study and working system, not a scale demonstration.

## Phases and milestones

### Phase 0 — Access and hardware (first)

Validate the user-reported 70–84 systems and per-node specs; establish simultaneous access, node homogeneity, cluster scheduler, permissions, network fabric, storage quota and job limits. Run single-node and multi-node 100M/300M/1B training, inference, all-reduce, checkpoint/recovery and 24-hour availability benchmarks. **Exit:** measured sustained throughput, scaling efficiency and usable allocation with resume path. Until then, treat cluster operation as possible rather than confirmed; use nodes for independent system/verifier/data workloads. If accelerator capacity is absent, evaluate CPU training from measured schedule and secure academic/cloud accelerator access for substantial pretraining.

### Phase 1 — Research infrastructure

Implement provenance schema, rights manifest, immutable event log, paper ingestion with span locators, hybrid retrieval, sandboxed verifiers and evaluation harness. Use a licensed public checkpoint as baseline subject to model license. **Exit:** a human can trace every final claim to source/tool artifact, and evaluation reproduces.

### Phase 2 — Small trainable proxies

Train 100M/300M dense controls first; advance to distributed 1B only if the hardware gate supports a useful reproducible schedule. Run tokenizer and data-mixture sweeps and basic SFT. Reproduce loss curves and compare a fixed RAG+tool baseline. **Exit:** scaling curve, cost report and no rights/contamination blockers.

### Phase 3 — RL signal study

At a fixed proxy, factorial tests for DPO/preference, outcome-only RLVR, calibrated decision head/model, PRM auxiliary and integrations. Use math/code/simulation tasks with independent held-out generators. **Exit:** preregistered results, judge/verifier audit, posterior/calibration and reward-hacking analysis.

### Phase 4 — Long-horizon research loop

Build controller and evidence graph; compare transcript-only, structured-memory, learned router, fixed router. Evaluate 10–50-step tasks, source conflicts, tool failures and scientific simulations. **Exit:** repeatable improvement in supported research progress per budget on sealed tasks.

### Phase 5 — Substantial model decision

Use scaling loss and task curves, licensed token volume, accelerator allocation, full memory/throughput and inference targets to choose 7B, 14B or 32B dense. Seek external compute for any serious base pretraining. Run modest MoE only if measured fabric and dense bottleneck justify it. **Exit:** board-approved scale with actual price/time range and contingency.

### Phase 6 — Replication and public contribution

External team reruns configs/benchmarks; publish method, limitations, data lineage, negative results and artifacts allowed by rights. Claim novelty only if ablations show causal benefit beyond baseline tools and the benefit survives held-out domains.

## Staffing / work allocation

One ML systems lead (audit/training/repro), data-rights/data-engineering lead, agent/provenance engineer, evaluation/statistics lead, domain experts for label protocols, and project PI for preregistration and decision gates. CPU cluster is prioritized for source processing, retrieval indexing, code/math verifiers, simulations, RL rollout orchestration and evaluation. GPU allocations are reserved for model training/inference. Exact staffing is an organizational assumption to confirm.

## Stop conditions

Pause scaling when: accelerator throughput/availability misses plan; no legal corpus supports token target; reward score rises while independent correctness falls; calibrated judge has unacceptable shifted risk; research loop fails to beat a simple RAG+tool baseline; provenance cannot be audited; or benchmark leakage invalidates primary result. Record failure and redirect to the bottleneck rather than increasing parameter count.
