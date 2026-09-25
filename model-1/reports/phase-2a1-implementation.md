# Phase 2A.1 — resumable training and arithmetic-step supervision

**Date:** 2026-09-25 local time. **Completed. Both capability gates failed; worked-arm promotion gate failed.**

## Artifacts and implementation

- [Preregistered experiment](../docs/experiments/phase-2a1-protocol.md), published in `79e8843` before the new dataset or training was run.
- [Comparison reproduction guide](../docs/FINITE_DIFFERENCE_COMPARISON.md).
- [Researcher checkpoint continuation](../docs/STRUCTURED_RESEARCHER.md).
- Reproduce with `.venv/bin/python -m aim.comparison_reproduce` from `model-1/`.

This phase implemented tested continuation of research-specific SFT, a checkpoint-versioned arithmetic-step output format, independent process diagnostics, fresh data shared across two training arms, matched dense training shapes, frozen multi-seed selection, and actual Controller/verifier evaluation.

All learned weights originate locally from random initialization. The Researcher, separate Judge, deterministic Controller, independent verifiers and provenance memory retain their responsibilities. SFT, preference/RLHF, RLVR and Judge calibration remain separate mechanisms. This comparison adds no joint reward or process reward model.

## What the continuation tests establish

Four uninterrupted updates match two plus two resumed updates exactly for model tensors, optimizer state, torch RNG, sampler RNG and validation metrics. The original random baseline and earlier checkpoint-selection history survive continuation. Additional tests exercise original-format checkpoint history, deadline interruption, altered optimizer/seed/data, rewritten past evaluation schedules and tampered historical validation.

An additional retained audit resumed the **actual Phase 2A seed-17 checkpoint at step 1,200** and trained its remaining 600 updates. The result exactly matched the original step-1,800 model tensors, optimizer, torch RNG, sampler RNG, selected step and validation metrics; the original step-0 checkpoint hash was preserved. This audit used the old dataset and validation set, with no new holdout evaluation. Its directory is `runs/20260924T192233-legacy-research-resume-audit-3ec8993d`.

New runs preserve parent hashes, source snapshots, configuration, tokenizer, dataset identity, optimizer and RNG state, validation artifacts and checkpoint history. Original run directories remain intact. Missing or inconsistent history is an error. Legacy cost counters are explicitly scoped by `accounting_start_step`; absent historical token usage is not invented.

Limitations: this establishes deterministic continuation in the recorded CPU environment. Cross-device, cross-version and distributed continuation are not established. History references use absolute paths, so moving artifacts requires a separate relocation design. The full comparison driver begins a fresh paired experiment; it does not automatically assemble six resumed seed runs.

## Controlled comparison

| Item | Fixed choice |
|---|---|
| Architecture | Dense native decoder, 228,096 parameters; width 96, two layers, GQA 4/2, FFN 256, context 256 |
| Initialization | Seeds 17, 23, 41; identical initial tensors within each plain/worked pair |
| Data | 512 training, 64 validation, 64 in-family test, 64 cubic OOD worlds |
| Exclusion | All 704 coefficient vectors from Phase 2A's four splits excluded by actual vector |
| Training | 1,800 AdamW updates, batch 16, LR 0.001, weight decay 0.01, clip 1, CPU FP32, one thread |
| Dense compute proxy | Both arms padded to 256 training input positions: 7,372,800 positions per arm/seed |
| Validation | Step 0, 300, 600, 1200, 1800; agreement, validity, NLL, earliest tie |
| Generation | Greedy, 128 new tokens maximum, exactly one hypothesis, no repair or fallback |
| Loop | Rule Judge, eight-action ceiling, actual measurement and provenance checks |

The plain arm generates final coefficients and citations. The worked arm additionally generates d1=y1−y0 and d2=y2−2y1+y0. Independent diagnostics check those numbers and each coefficient against the supplied observations. No computed answer is inserted into model context or substituted for its output.

The budget matches dense tensor shapes and sampled-world exposure, not measured FLOPs, wall time, or supervised information quantity. Worked responses intentionally provide more target tokens. Report these costs separately. NLL is used only for within-arm checkpoint selection because response distributions differ.

The data generator exports training/validation separately from test/OOD. All six selected checkpoint hashes are frozen before holdout scoring. The final evaluation repeats the same 128 distinct holdout worlds across 13 backends, yielding 1,664 loop executions; these are not 1,664 independent research questions.

## Interpretation and scope

Contract validity, coefficient agreement, process correctness and independently VERIFIED final predictions are different measurements. A correct prediction can coexist with a wrong worked step. The latter remains a recorded process failure; this experiment does not silently extend the Controller's two-check verification scope.

For in-family tasks, observed-coefficient agreement equals recovery of the synthetic quadratic's coefficients. For cubic OOD, it measures only agreement with a quadratic interpolation of the three observed points; it is not hidden cubic coefficient recovery or proof of a law. Exact OOD/training prompt overlap remains a disclosed ambiguity rather than a removed case.

Wilson intervals are descriptive for this finite synthetic sample. Seeds share worlds, so report seed dispersion and paired gains/losses separately. These results cannot select a 1B/7B size, validate a 70–84-node VIT cluster, establish broad scientific reasoning, or substantiate a novel RLHF+RLVR+RLCD combination.

## Retained invalid attempt and correction

The first attempt, `runs/20260924T191101-comparison-reproduction-85b869f1`, was deliberately interrupted after a data-boundary bug was found. Its split manifest included new holdout coefficient vectors; the trainer loaded that file to obtain hashes. Those vectors were never selected as training rows, prompts or responses, but loading them violated the registered file-access boundary. The attempt is invalid for capability claims.

The process was stopped before any holdout evaluation, and its failed run, partial checkpoints, traceback and `protocol-invalidated.json` annotation remain on disk. No holdout outcomes were used to change the model, budget, thresholds or selection rule. The fix moved label-bearing membership into a separate audit file, allowlisted trainer metadata, and added tests for both separation and rejection. A full fresh-initialization rerun uses the same preregistered worlds and unchanged protocol. This is a disclosed implementation correction, not a new favorable seed search.

## Results and next decision

**77 tests passed; zero skips.** The corrected reproduction completed all six training runs and all 1,664 Controller investigations. The original invalid attempt is excluded from these capability results.

| Backend | Selected update | Contract-valid | Verified test predictions | Test rate | Wilson 95% | OOD verified |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic reference | — | 64/64 | 64/64 | 100.00% | 94.34–100.00% | 0/64 |
| Plain 17 | 1200 | 63/64 | 1/64 | 1.56% | 0.28–8.33% | 0/64 |
| Plain 23 | 1800 | 64/64 | 5/64 | 7.81% | 3.38–17.02% | 0/64 |
| Plain 41 | 1800 | 63/64 | 4/64 | 6.25% | 2.46–15.00% | 0/64 |
| Worked 17 | 1800 | 63/64 | 9/64 | 14.06% | 7.58–24.62% | 0/64 |
| Worked 23 | 1800 | 63/64 | 13/64 | 20.31% | 12.27–31.71% | 1/64 |
| Worked 41 | 1800 | 64/64 | 2/64 | 3.12% | 0.86–10.70% | 1/64 |

All six random-initialization backends scored zero verified predictions on both splits. Every invalid output remains in the denominator. Final prediction success is scoped to a requested measurement, not proof of a global polynomial law.

| Seed | Worked gained / lost versus plain | Net difference |
|---|---:|---:|
| 17 | 9 / 1 | +12.50 percentage points |
| 23 | 12 / 4 | +12.50 percentage points |
| 41 | 2 / 4 | -3.12 percentage points |

Mean worked-minus-plain test gain: **+7.29 percentage points**. The registered worked-advantage gate is **False**, and the stricter worked-promotion gate is **False**.

| Arm | Mean test success across seeds | Sample standard deviation |
|---|---:|---:|
| Plain | 5.21% | 3.25 percentage points |
| Worked | 12.50% | 8.70 percentage points |

Across all clean-run investigations, **0 VERIFIED claims lacked passing checks**. This is evidence for the implemented boundary on these cases, not a universal factuality guarantee.

### Numerical and process diagnostics

| Backend | c0 correct | c1 correct | c2 correct | d1 correct | d2 correct | All steps correct | Verified with wrong process |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain 17 | 63/64 | 8/64 | 12/64 | — | — | — | — |
| Plain 23 | 64/64 | 5/64 | 12/64 | — | — | — | — |
| Plain 41 | 63/64 | 5/64 | 17/64 | — | — | — | — |
| Worked 17 | 63/64 | 13/64 | 16/64 | 34/64 | 15/64 | 7/64 | 2 |
| Worked 23 | 63/64 | 15/64 | 25/64 | 23/64 | 25/64 | 8/64 | 5 |
| Worked 41 | 64/64 | 5/64 | 17/64 | 19/64 | 17/64 | 1/64 | 1 |

These diagnostics count all worlds, including invalid generations. Intermediate checks are independent arithmetic diagnostics and did not alter final Controller decisions.

Eight of the 24 familiar-world successes from the worked arm had an incorrect process trace (2, 5 and 1 by seed). These are repeated evaluations on shared worlds, not 24 independent tasks. The two cubic OOD passes also had incorrect intermediate steps and failed to recover the observation-consistent quadratic coefficients. They happened to match the requested target measurement; they do not establish cubic reasoning. This directly supports retaining explicit verification scope and separate process diagnostics.

### Measured cost and retained artifacts

| Arm / seed | Processed training positions | Supervised response tokens | Training + validation seconds |
|---|---:|---:|---:|
| Plain 17 | 7,372,800 | 1,334,983 | 111.36 |
| Plain 23 | 7,372,800 | 1,335,178 | 109.09 |
| Plain 41 | 7,372,800 | 1,334,674 | 109.20 |
| Worked 17 | 7,372,800 | 1,765,014 | 112.56 |
| Worked 23 | 7,372,800 | 1,765,589 | 112.12 |
| Worked 41 | 7,372,800 | 1,764,799 | 112.42 |

Holdout evaluation took 271.35 seconds. Total clean reproduction took 947.19 seconds. The separate legacy-resume audit overlapped the first plain seed; these elapsed times are operational observations, not isolated throughput benchmarks or a fair speed ranking. No VIT hardware was measured.

The dataset reports one exact OOD/training prompt overlap. Different hidden laws can share observations; no such case was filtered after scoring.

Artifacts:

- [Portable results, seeds, hashes, costs and environment](phase-2a1-evidence/results.json).
- [All 1,664 case records with raw generations and process diagnostics](phase-2a1-evidence/case-results.json).
- [Full 77-test log](phase-2a1-evidence/tests.log).
- [Invalidated attempt and retained failure](phase-2a1-evidence/invalidated-run.json).
- [Real-checkpoint continuation audit](research-resume-audit.md).
- [Portable export integrity manifest](phase-2a1-evidence/export-manifest.json).
- [Exact changed-file inventory](phase-2a1-files.json).

Complete clean run: `runs/20260924T192137-comparison-reproduction-35716f6b`. Its stage-pointer files locate the dataset, training and evaluation. Code/config snapshots and original checkpoints remain local. Later Git commits changed reporting documents; source hashes record the actual executed implementation.

### Decision and next experiment

**Decision:** keep the deterministic Researcher as the default. Keep both native arms opt-in. Worked supervision is a promising but unstable signal in this small synthetic setting: its mean improvement exceeds five percentage points, but seed 41 regressed and no seed reached the required 25% verified-success rate. All six models passed the syntax-validity threshold; that does not compensate for missed numerical gates. No architecture, final parameter count or joint reward is approved by this outcome.

**Recommended next experiment, not yet implemented:** preregister an arithmetic-curriculum ablation at the same model scale. Compare worked SFT alone with a mixture that explicitly teaches prerequisite signed subtraction, second differences and coefficient reconstruction, under matched dense-compute budgets. Record exact training versus validation arithmetic accuracy to distinguish inadequate fitting from failed generalization. Exclude coefficient worlds from both completed phases, freeze new test/OOD tasks before training, and keep this now-inspected suite as a regression diagnostic. The data mixture and stopping rule need a new protocol before execution.

**Verifier follow-up:** design a separately scoped observation-consistency check and counterexample cases. A future hypothesis-level claim should not inherit the status of a single successful target prediction. Any new gate needs a new evaluation version; do not silently relabel the results here. Incorrect process traces also argue against feeding unchecked generated steps into RLVR rewards.

**Still unresolved:** whether a curriculum or greater capacity fixes numerical learning; whether preference optimization or verifier-grounded RL improves these research tasks; how to calibrate the separate Judge under shift; cross-machine artifact relocation and distributed recovery; real corpus/tokenizer quality; and the physical VIT fleet's throughput, memory and network limits. Preserve separated SFT/preference/RLVR/Judge baselines while investigating these questions.

## Component commits and file accounting

- `d10eeb5`, `3ed9377` — research-trainer continuation and legacy history validation.
- `79e8843` — fixed protocol and configuration before experimental execution.
- `5ca03c0` — worked output contract and fresh failure traces.
- `2fc2178`, `d08e8c3` — fresh-world data, corrected audit isolation and regression tests.
- `4fb1af8` — matched training shapes and cost counters.
- `8f77261` — frozen paired evaluation and process diagnostics.
- `25aa63b` — complete bounded reproduction entry point.
- `c110fc7`, `9ee34f7` — reproduction and data-boundary documentation.
- `c6be643` — actual old-checkpoint continuation evidence.

The corrected six seed runs share source hash `502d6e4696469f27f0403115cfba6bc63e0a25f1feafc8e0dd59c57d606e9f80`. Individual manifests retain Git revisions and working-tree status. The [inventory](phase-2a1-files.json) lists every phase change relative to `caae60a`; it excludes local run stores and unrelated user files. The historical handbook PDF and canonical phase-1 evaluation files were not changed.
