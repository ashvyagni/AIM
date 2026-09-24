# AIM Research Intelligence Model — Training Specification

## Status

Research protocol; the user reports approximately 70–84 VIT lab systems, each described as 13th-generation Intel Core i9, 32 GB RAM and Intel UHD Graphics 770. Treat specs/count as preliminary inventory. Aggregate RAM would be 2.24–2.688 TB only if consistent, separately distributed across hosts; the nodes may be joinable as a CPU cluster, but network, scheduler, access, storage and throughput are unknown. Start with 100M–300M single-node CPU proxies and distributed systems/data/verifier workloads. Require measured multi-node scaling before 1B/7B decisions; 7B model-state sharding is technically distinct from practical end-to-end training. Accelerator access remains preferred for substantial pretraining. Values below are experiment designs or conditional defaults, never measurements.

## Stage plan

| Stage | Data/objective | Method and artifact | Gate |
|---|---|---|---|
| 0: reproducibility | Fixed public/licensed tiny corpus, synthetic benchmark | Deterministic data loader, tokenizer tests, 100M–1B reference run, checkpoint/resume, logs | Same config reproduces loss curve and evaluation within declared variance. |
| 1: rights/data/tokenizer | Layer A–E with source manifests | Filtering, dedup, contamination, license gates; tokenizer comparison by domain | Rights coverage and token/domain report reviewed before training. |
| 2: scaling proxies | Broad licensed text plus held-out domain data | Dense sizes such as 100M, 300M, 1B at multiple token budgets; fit loss-vs-compute; AdamW baseline | Select scale using measured tokens/s and target quality, not headline parameters. |
| 3: mid-training | Math/code/science/long context with quality weights | Continue pretraining, controlled context curriculum; compare no-long-context + RAG | Gains survive knowledge and retrieval evaluations without broad regression. |
| 4: Research SFT | Tool calls, claim/evidence records, research episodes | Supervised action/protocol and final writing; masks for tool results; no unvalidated synthetic trace as gold | Tool contract accuracy, citation support and task completion improve. |
| 5: RLVR | Math, code, formal proof, statistics/simulation environments | PPO or GRPO outcome reward; sandbox and hidden tests; separate task groups | Verifier false-accept audit; gains on held-out generators at fixed compute. |
| 6: human preferences | Expert/researcher pairwise outputs and rubric dimensions | DPO first; compare PPO/RM only if online exploration merits it. Keep separate dimensions. | Better blind usefulness without factuality/calibration degradation. |
| 7: calibrated decision learning | Claim evidence labels, route outcomes, abstain/usefulness costs | Typed head/model; proper score supervised first; RL only for clean trajectory decision outcomes | Better Brier/log/selective utility in domain/time-shift sets. |
| 8: process guidance | Correct/incorrect step annotations and outcome-conditioned traces | PRM as bounded search/ranking/shaping ablation; compare outcome-only | No gain that relies on familiar step style or verbosity. |
| 9: integration | Full research loop trajectories | Constrained objective and component ablations; freeze judge/verifier versions | Independent trajectory gains per cost; external red-team passes. |

## Objective details

Pretraining: `L_LM = -Σ mask_t log πθ(x_t|x_<t)`. SFT uses token NLL for target responses/actions; never backprop through untrusted tool observations. Preference: pairwise Bradley–Terry, DPO as primary simple baseline. RLVR: terminal reward from verifiers, conservative KL to reference and clipped PPO/GRPO comparison. Decision: multiclass log score/Brier on resolved labels plus explicit abstain action and task-cost utility; do not optimize ECE directly. Process model: step labels and outcome-conditioned validation; no universal process reward.

Integrated choice: keep signals separated at first. Treat correctness/rights/tool safety as constraints or release gates; optimize usefulness and resource cost within feasible region; calibrate decision distributions separately. A later shared policy update is acceptable only if scale normalization, conflict, reward uncertainty and Pareto outcomes are logged. No fixed `w_h,w_v,w_c,w_p` is specified because no measurement supports universal exchange rates.

## Data mixture and sequence strategy

Build A knowledge, B structured reasoning, C research process, D negative/corrective, E synthetic. Establish ratios via pilot sweep (e.g., 0/5/10/20% high-quality reasoning mixtures in continued pretraining as experimental levels, not proposed final proportions). General-language anchor prevents narrow overfit. Dedup by source/family. Hold out complete synthetic generators, source families, dates and task templates. Train with packed variable-length examples and loss masks. Start base sequence 4–8K; use bucketed batches and staged long context only if it wins against RAG on matched tasks.

## Optimizer, schedule, precision

AdamW reference; warmup + cosine or validated constant/decay schedule chosen from pilot and exact run length. Log LR, weight decay, gradient norms, clipping, update/weight ratio, loss by source/domain, throughput, utilization, scaler and skipped steps. Compare Muon on identical data/compute and tuning budget before selecting. BF16 where hardware/runtime numerics pass; FP32 CPU proxy; optional FP8 on supported accelerators. Activation checkpointing and sharding choices require benchmark. No claimed learning rate/batch size until scaling pilot.

## RL protocol

Collect rollouts from frozen snapshot; attach exact model/tokenizer/verifier/environment revisions; group prompts by difficulty and report reward variance; cap response/tool length; use hidden evaluator items. Measure policy KL, entropy, answer length, sample pass rate, verifier agreement, reward vs independent outcome, abstention, failures and compute. Stop on proxy/outcome divergence, reward exploitation, calibration regression or catastrophic held-out regressions. Retain best checkpoint by predeclared evaluation, not latest reward.

## Compute, distributed topology, reproducibility

FLOPs estimate `6ND`; no final token count before rights-cleared corpus and cluster budget. Conditional dense 7B at 140B tokens ≈5.88e21 FLOPs; 32B at 640B ≈1.23e23. Checkpoints at immutable initialization, milestones, best validation, final; keep sharded resume plus export. Hash every data manifest/config/code/container/checkpoint. Seed Python/NumPy/framework/sampler; document nondeterministic kernels. Publish run table including failures. Storage estimate and hardware gate in scaling/hardware docs.
