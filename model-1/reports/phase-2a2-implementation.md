# Phase 2A.2 — arithmetic curriculum and observation consistency

**Date:** 2026-09-25. Experiment execution in progress; final outcomes will be recorded after frozen evaluation.

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

The full held-out experiment will retain 1,664 Controller runs over 128 shared worlds and a separate source-backed observation audit. Repeated models/seeds are not independent task samples. Descriptive Wilson intervals and seed dispersion do not establish general scientific capability.

## Interpretation limits

- Auxiliary diagnostic samples have at most 32 prompts per task/split; training research diagnostics use 64 worlds. They do not measure exhaustive mastery.
- The worked-only baseline never trains on auxiliary response formats, so an auxiliary accuracy gap can reflect format familiarity as well as arithmetic learning. Primary comparisons use the shared full research task.
- A PASS from the new checker means agreement with cited observations, not future agreement, source credibility, natural-language entailment or proof of a global law.
- A target-plus-observation conjunction is an additional diagnostic, not a silently changed canonical evaluation gate. Process-step correctness is reported separately.
- Read-only memory protects this audit's API path; it is not a sandbox against unrelated external writers.
- No physical VIT cluster, larger proxy model, distributed recovery, or joint RLHF+RLVR+RLCD advantage has been measured here.

## Measured results, failures and decision

Pending the registered frozen evaluation. No decision is taken from partial validation results.
