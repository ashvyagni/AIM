# Phase 2A.2 — arithmetic curriculum and observation consistency

**Date:** 2026-09-25. Completed and audited. **The curriculum promotion gate failed.** Retain the deterministic Researcher as the default; both learned recipes remain opt-in experiments.

## Implemented components

1. **Fresh research worlds:** exclude all 1,408 complete coefficient vectors used by Phase 2A and Phase 2A.1. Expand the intercept/slope range to -19..19 because too few unused linear worlds remain in the former range. Preserve 512 training, 64 validation, 64 test and 64 cubic OOD worlds, with source-bound evidence and trainer-safe metadata.
2. **Staged arithmetic SFT:** compare the existing worked-response recipe with prerequisite signed subtraction, second differences and coefficient reconstruction. Teachers derive only from each training row's observations. All outputs remain model-generated at full-research inference.
3. **Fit and transfer diagnostics:** measure full research generation on a fixed training sample and exact auxiliary arithmetic on deduplicated training/validation prompts. Validation auxiliary prompts seen in the training auxiliary pool are excluded. These diagnostics do not select checkpoints.
4. **Resumable recipe accounting:** retain task presentation counts, sampler state and a rolling hash of sampled worlds alongside optimizer/model/RNG/selection history. Both arms must match initial weights, world draws and processed-position budget.
5. **Independent hypothesis observation checker:** bind results to hypothesis/evidence hashes, validate original spans and topic, evaluate every cited observation with rational arithmetic, and report PASS/FAIL/UNKNOWN with explicit finite-observation scope.
6. **Read-only memory audit:** reject mutating APIs before object writes, open SQLite read-only/query-only, and verify state/database hashes before and after each case audit. Existing Controller statuses remain unchanged.
7. **Reproduction and export:** run tests, data generation, all six seeds, frozen Controller evaluation and a separate observation audit; export validated metrics, case records, selected fitting diagnostics, test logs and integrity hashes.

The Researcher, separate Judge, deterministic Controller, independent Verifiers and provenance Memory architecture remains intact. Preference/RLHF, RLVR and Judge calibration stay separate from this SFT recipe comparison. No new pretrained weights, combined reward, tokenizer or final parameter-size decision is introduced.

## Protocol and reproducibility

- [Preregistered protocol](../docs/experiments/phase-2a2-protocol.md), committed as `046f96a` before dataset generation or training.
- [Reproduction guide and limitations](../docs/ARITHMETIC_CURRICULUM.md).
- Command from `model-1/`: `.venv/bin/python -m aim.curriculum_reproduce`.
- Registered config: `configs/arithmetic-curriculum.json`.
- Full retained run: `runs/20260925T073010-curriculum-reproduction-60d13ca4`.

The model has 228,096 parameters and starts from random weights for seeds 17, 23 and 41. Both arms use the worked output schema, CPU FP32, one thread, AdamW LR 0.001, weight decay 0.01, gradient clip 1, batch 16, 1,800 updates, context 256 and greedy generation capped at 128 new tokens. Both process 7,372,800 padded training positions per seed. This controls dense tensor shapes, not measured FLOPs, wall time or supervised information quantity.

The worked arm receives 28,800 full research presentations per seed. The curriculum receives 19,200 full research, 4,800 subtraction, 2,400 second-difference and 2,400 reconstruction presentations. The experiment compares these full recipes. It does not isolate ordering from task exposure or prove that an arithmetic example is more valuable than a research example. The selected checkpoints may occur at different update boundaries even though total training budgets match.

Selection uses only full-research validation agreement, validity, NLL and earliest tie. Every initial/selected checkpoint hash is frozen before test/OOD scoring. Previous holdouts are now historical diagnostics; new results are not directly comparable to their rates because the coefficient range and world selection changed.

## Tests and verification scope

**92 preflight tests passed; zero skips.** New coverage includes exact signed teacher labels, strict auxiliary schemas, curriculum boundaries and world attribution, validation-prompt overlap exclusion, exclusion of both historical world sets, metadata/holdout isolation, altered-label rejection, paired training, resume accounting and actual Controller-plus-audit integration.

Verifier tests include forged and missing spans, wrong topics, unsupported/nonfinite data, conflicting observations, read-only database/object behavior, and changed hypothesis hashes. Two actual loop counterexamples establish the distinction between target and observation checks: one hypothesis matches the future target but fails earlier observations; another fits observed cubic samples with a quadratic and misses the next measurement. The observation checker neither promotes nor relabels the Controller's claim.

The full held-out experiment retained 1,664 Controller runs over 128 shared worlds and a separate source-backed observation audit. Repeated models/seeds are not independent task samples. Descriptive Wilson intervals and seed dispersion do not establish general scientific capability.

## Interpretation limits

- Auxiliary diagnostic samples have at most 32 prompts per task/split; training research diagnostics use 64 worlds. They do not measure exhaustive mastery.
- The worked-only baseline never trains on auxiliary response formats, so an auxiliary accuracy gap can reflect format familiarity as well as arithmetic learning. Primary comparisons use the shared full research task.
- A PASS from the new checker means agreement with cited observations, not future agreement, source credibility, natural-language entailment or proof of a global law.
- A target-plus-observation conjunction is an additional diagnostic, not a silently changed canonical evaluation gate. Process-step correctness is reported separately.
- Read-only memory protects this audit's API path; it is not a sandbox against unrelated external writers.
- No physical VIT cluster, larger proxy model, distributed recovery, or joint RLHF+RLVR+RLCD advantage has been measured here.

## Measured results, failures and decision

### Primary held-out outcome

Every selected model produced valid full-research output on all 64 familiar test worlds. All initial checkpoints achieved 0/64 valid outputs and 0/64 verified successes. The deterministic reference achieved 64/64 familiar target successes and 0/64 cubic OOD target successes.

| Recipe | Seed | Selected update | Validation agreement /64 | Test verified /64 | Test rate, descriptive Wilson 95% | OOD verified /64 |
|---|---:|---:|---:|---:|---|---:|
| Worked | 17 | 1800 | 3 | 4 | 6.25% [2.46, 15.00] | 2 |
| Worked | 23 | 1200 | 6 | 3 | 4.69% [1.61, 12.90] | 0 |
| Worked | 41 | 1200 | 7 | 6 | 9.38% [4.37, 18.98] | 1 |
| Curriculum | 17 | 1800 | 8 | 4 | 6.25% [2.46, 15.00] | 2 |
| Curriculum | 23 | 1200 | 3 | 4 | 6.25% [2.46, 15.00] | 0 |
| Curriculum | 41 | 1200 | 4 | 4 | 6.25% [2.46, 15.00] | 2 |

Worked mean test success was **6.77%** (sample standard deviation 2.39 percentage points); curriculum was **6.25%** (zero dispersion across these three seeds). Curriculum minus worked was **−0.52 percentage points**. Paired gained/lost worlds were 3/3, 4/3 and 3/5 for seeds 17/23/41, respectively. Equal totals do not imply the models succeeded on identical worlds.

Both arms passed the familiar validity threshold but failed the 25% verified-success and 10-percentage-point improvement thresholds on every seed. The curriculum also failed its required mean advantage of at least 5 percentage points and no-seed-regression condition. These are negative capability results from completed experiments, not infrastructure failures. There were zero unbacked VERIFIED claims under the original provenance-plus-target definition.

**Selection nuance:** curriculum seeds 23 and 41 selected update 1200, before reconstruction training. All six runs still completed the fixed 1800-update budget. Choosing those checkpoints follows the registered validation rule; selecting final checkpoints after seeing test results would change the protocol.

### Observation and process audit

| Recipe / seed | Familiar target passes | Of those, observation passes | Target passes without observation pass | Target passes with incorrect worked steps |
|---|---:|---:|---:|---:|
| Worked / 17 | 4 | 0 | 4 | 4 |
| Worked / 23 | 3 | 0 | 3 | 3 |
| Worked / 41 | 6 | 3 | 3 | 3 |
| Curriculum / 17 | 4 | 3 | 1 | 2 |
| Curriculum / 23 | 4 | 2 | 2 | 4 |
| Curriculum / 41 | 4 | 0 | 4 | 4 |

Across selected learned models, **17 of 25 familiar target passes failed observation consistency**; **20 of 25 contained an incorrect worked process**. All seven learned OOD target passes failed observation consistency and had incorrect process steps. Thus neither a checked future target nor valid arithmetic-looking JSON establishes a sound hypothesis or process.

The reference fit all 64 cubic observation sets but missed every cubic target. Curriculum seeds 17 and 23 each fit one cubic observation set, also missing its target. All models see only three observations; a cubic term proportional to `x*(x-1)*(x-2)` is invisible there. See the [finite-observation argument](../docs/OBSERVATION_SCOPE.md). No three-point observation checker can resolve that unrestricted-law ambiguity.

Joint familiar target-plus-observation counts were worked 0/0/3 and curriculum 3/2/0, each out of 64. This is a useful additional diagnostic, but the small difference does not rescue the failed registered promotion gate. Original Controller statuses and historical evaluation definitions were not changed. The audit confirmed unchanged state and database hashes for every case.

### Training mastery and transfer

Full-research training-sample agreement versus validation agreement at the **selected** checkpoint:

| Recipe | Seed | Training sample /64 | Validation /64 |
|---|---:|---:|---:|
| Worked | 17 | 18 | 3 |
| Worked | 23 | 4 | 6 |
| Worked | 41 | 7 | 7 |
| Curriculum | 17 | 10 | 8 |
| Curriculum | 23 | 0 | 3 |
| Curriculum | 41 | 8 | 4 |

At update 1800, worked training-sample agreement was 18/19/24 of 64, versus validation 3/5/6. Curriculum training-sample agreement was 10/8/8, versus validation 8/3/3. Neither recipe demonstrates exhaustive training mastery; the worked final-checkpoint gap is consistent with limited generalization, without isolating capacity, optimization or representation as its cause.

Curriculum auxiliary validation diagnostics, shown as **exact answers / schema-valid answers / 32 prompts**:

| Diagnostic at stage endpoint | Seed 17 | Seed 23 | Seed 41 |
|---|---|---|---|
| Subtraction, update 600 | 1 / 32 / 32 | 0 / 32 / 32 | 1 / 32 / 32 |
| Second difference, update 1200 | 7 / 32 / 32 | 6 / 32 / 32 | 6 / 32 / 32 |
| Reconstruction, update 1800 | 20 / 32 / 32 | 6 / 32 / 32 | 25 / 31 / 32 |
| Subtraction, update 1800 | 0 / 0 / 32 | 0 / 2 / 32 | 0 / 0 / 32 |
| Second difference, update 1800 | 0 / 0 / 32 | 0 / 0 / 32 | 0 / 0 / 32 |

Reconstruction training-sample exact counts at update 1800 were 26/11/27 of 32. It receives correct `d1` and `d2` in its auxiliary input; full research generation must compute them from observations. Better reconstruction on that input does not establish end-to-end arithmetic competence. Earlier auxiliary response validity collapses after task replacement; forgetting or format interference is a hypothesis, not an identified mechanism. The worked baseline had zero valid auxiliary outputs throughout, since those formats were never trained.

The subtraction validation pool excluded 52 of 124 unique prompts present in training, leaving 72 eligible and sampling 32. Second-difference and reconstruction validation pools each had 64 eligible prompts, sampling 32. All fitting curves and selected raw generations are exported; these samples do not support exhaustive arithmetic claims.

### Compute and integrity

The full run completed in **2,311.39 seconds (38.52 minutes)** on the recorded local macOS arm64 CPU, Python 3.12.14 / PyTorch 2.8.0, one torch thread. Dataset creation took 1.57 seconds, training with diagnostics 1,908.60 seconds, Controller evaluation 379.92 seconds, and observation audit 2.03 seconds. Worked seed runs took 328.07/334.60/360.49 seconds; curriculum runs took 282.64/309.50/292.84 seconds. These are single-run elapsed times including diagnostics, not sustained throughput or a controlled speed comparison.

Each seed processed 7,372,800 padded positions, for **44,236,800 across all six runs**. Worked supervised response tokens were 1,815,675 / 1,816,191 / 1,815,661; curriculum 1,373,881 / 1,374,408 / 1,373,892. Padding controls dense work shapes while supervision quantity differs. Greedy diagnostic generation is a plausible runtime cost; no profile separates it from training here, so no measured bottleneck percentage is claimed.

All six source snapshots share code hash `c3288eb339b5d0c8347a20547a294090e927c174f0a6b41a1ef44f2dbdf99709`. Documentation commits occurred during execution; manifests retain their actual Git commit and dirty status, and executable source hashes stayed identical. Frozen selection SHA-256 is `7d5ac8b27c3e0f100ccbecd9140f6df1dfa8b246b554a36c36b764db78e43a5f`; holdout SHA-256 is `817e5a05828aa8aae242ad31da5c52ad07c94da43ea5ab15188e977c7d906e60`.

The exporter checked stage completion, checkpoint/holdout hashes, paired sampling digests, task/position accounting, all 1,664 case statuses, aggregate counts and unchanged audited state/database hashes. An export-only path-normalization issue was corrected for dictionary keys, checked directly, and the bundle regenerated; original experiment artifacts were untouched. The 92-test preflight result belongs to the training implementation, not a claim that the later reporting edit reran that suite.

### Decision and next experiment

**Evidence-backed decision:** retain the deterministic default and preserve both learned recipes as experiments. Do not enlarge the model or merge rewards on this evidence. The registered curriculum recipe did not demonstrate its intended benefit.

**Recommended next phase: 2B, separate Judge calibration and decisions under shift.** Define the event as future target-verifier passage before that measurement, compare constant/base-rate and deterministic policies with learned proper-score predictors, and evaluate abstention/measurement cost. Use fresh world/family partitions, freeze calibration and thresholds before final evaluation, and exclude target measurements and post-outcome fields from Judge features. Preserve the observation-consistency result as a scoped feature; it cannot prove an unrestricted future law. Existing exposed test/OOD worlds are historical diagnostics, not new holdouts.

An interleaved replay curriculum could test the suspected forgetting mechanism later, with matched task counts and ordering ablations; that is a hypothesis for a new protocol, not an approved replacement or a post-hoc retuning of this study. Judge feature choice, utility costs, shift families, calibration robustness and final parameter scales remain unresolved. Physical VIT hardware measurements are still needed before a scale commitment.

No Phase 2A.2 infrastructure failure or invalidated training attempt was recorded. All poor checkpoints, unsuccessful predictions and raw selected diagnostics are retained. The prior Phase 2A.1 metadata violation remains documented in its historical report.

## Evidence and exact files changed

- [Audited results, configurations, selection and fitting curves](phase-2a2-evidence/results.json)
- [All Controller case results](phase-2a2-evidence/case-results.json)
- [Observation checks with state/database hashes](phase-2a2-evidence/observation-checks.json)
- [Selected raw fitting diagnostics](phase-2a2-evidence/selected-fit-diagnostics.json)
- [Actual preflight test log](phase-2a2-evidence/tests.log)
- [Export file hashes](phase-2a2-evidence/export-manifest.json)
- [Exact changed-file inventory against `6324b01`](phase-2a2-files.json)

The inventory covers implementation, configs, tests, protocol, report exporter, evidence and engineering documentation. Local run directories retain checkpoints/source archives and are excluded from Git; no unrelated owner file is included. Historical PDFs and earlier reports are preserved.
