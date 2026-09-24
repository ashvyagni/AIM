# Phase 2A — learned structured Researcher

**Completed:** 2026-09-24. **Capability gate: FAILED. Engineering implementation: working.**

The native Researcher now has a real training dataset, evidence-bound generation contract, three independently trained checkpoints and a complete held-out evaluation through the existing research loop. It learns JSON formatting and produces some numerically correct hypotheses, but its prediction accuracy is insufficient to replace the deterministic reference.

## Artifacts and reproduction

- [Registered protocol](../docs/experiments/phase-2a-protocol.md), committed as `20492e7` before training or holdout evaluation.
- [Structured Researcher guide](../docs/STRUCTURED_RESEARCHER.md): architecture changes, precise commands and limitations.
- [Portable results and provenance](phase-2a-evidence/results.json): exact metrics, source-run identities, hashes, selected checkpoint identities and gates.
- [All 896 backend/case records](phase-2a-evidence/case-results.json): raw outputs, schema outcomes, actual verification results, errors and local run references.
- [Complete 63-test log](phase-2a-evidence/tests.log), with temporary local test-directory paths anonymized.
- [One passing and one rejected example](phase-2a-evidence/illustrative-cases.json): first instances in fixed seed-17 evaluation order, explicitly illustrative. The complete case export prevents presenting these as representative accuracy.
- [Exact changed-file inventory](phase-2a-files.json).

Full local artifacts are preserved in:

| Stage | Run directory under `model-1/runs/` |
|---|---|
| Dataset/source memory | `20260924T143247-research-dataset-7003fc09` |
| Three-seed training and frozen selections | `20260924T143332-research-training-122c26fb` |
| Actual held-out Controller runs | `20260924T143706-research-holdout-ca8de777` |
| Post-training validation-only coefficient diagnostic | `20260924T143707-research-validation-analysis-9c515733` |
| Final test execution | `20260924T143951-phase2-tests-9a718514` |

The training and holdout runs used code commit `65029fc` with recorded working-tree status and source archives. A later documentation-only commit added the guide. Checkpoints and full run stores remain local; portable evidence is committed to Git. All selected checkpoints were frozen before the evaluator opened test/OOD cases.

## What was built

1. **Versioned Researcher contract:** compact structured observations, explicit evidence aliases, exactly one three-coefficient hypothesis and strict JSON/citation validation. Duplicate fields, nonfinite/boolean coefficients, unknown references and malformed output fail closed.
2. **Native adapter integration:** the existing TransformerResearcher selects the new contract only for appropriately tagged AIM checkpoints. The Controller records generated output and parse errors. No deterministic fallback is substituted for a failed neural generation.
3. **World-disjoint supervision:** 512 train, 64 validation, 64 familiar-family test and 64 cubic OOD worlds, with versioned source hashes and separate train/validation versus holdout files.
4. **Actual SFT:** 228,096 parameters, local random initialization, unchanged dense-decoder architecture and byte tokenizer; seeds 17, 23 and 41; 1,800 updates per seed, batch 16, LR 0.001, FP32 CPU. No RLVR or preference reward was added to this phase's loss.
5. **Validation-only selection:** initialization and steps 300, 600, 1200 and 1800 were evaluated and saved. Selection used agreement rate, output validity, response NLL and earliest tie-break. All three selected step 1800.
6. **End-to-end comparison:** one-candidate deterministic interpolator, each random initialization and each selected trained checkpoint; same Controller, rule Judge, evidence, tools, verifier and eight-action ceiling.
7. **Measurement and tests:** 20 additional tests, per-family metrics, paired changes from initialization, nominal Wilson intervals and immutable selection/hash checks.

The approved Researcher/Judge/Controller/Verifier/Memory separation remains intact. The known calibrated-Judge failure is held outside this experiment by using the verification-first rule Judge. The original phase-1 evaluation suite and records were not edited.

## Tests and execution

**63 tests passed; 0 skipped.** New coverage includes evidence alias binding, nonexistent/missing/duplicate citations, invalid and duplicate JSON keys, nonfinite coefficients, prohibited post-action inputs, raw failure trace retention, no fallback, deterministic split membership, world/split independence, exact teacher labels, holdout rejection by the training loader, data tampering, checkpoint selection priorities, training without opening a holdout file, update-budget limits, paired scoring, confidence-interval boundaries and a matched one-candidate reference.

The actual evaluation executed **896 investigations**: 128 reference cases plus 128 cases for each of six random/trained neural backends. These are repeated evaluations of **128 distinct holdout worlds**, not 896 independent scientific tasks. Each case has a retained state, tool/verifier records and source memory.

## Main held-out results

Familiar test split: 16 linear/constant and 48 quadratic worlds. Success means at least one independently VERIFIED prediction at the requested target and tolerance; each backend is limited to one candidate.

| Backend | Contract-valid outputs | Verified predictions | Success rate | Nominal Wilson 95% interval |
|---|---:|---:|---:|---:|
| Deterministic reference | 64/64 | 64/64 | 100% | 94.34–100% |
| Random initialization, each seed | 0/64 | 0/64 | 0% | 0–5.66% |
| Trained seed 17 | 64/64 | 7/64 | 10.94% | 5.40–20.90% |
| Trained seed 23 | 64/64 | 2/64 | 3.13% | 0.86–10.70% |
| Trained seed 41 | 61/64 | 2/64 | 3.13% | 0.86–10.70% |

The trained-seed mean success is 5.73%, with sample standard deviation 4.51 percentage points. Seeds use the same worlds; this is seed dispersion, not three independent task samples. Wilson intervals are descriptive nominal intervals for the fixed miniature test sample and do not establish general scientific capability or a multiple-comparison-adjusted significance claim.

All three trained models produced syntactically valid JSON on every familiar test case. Seed 41 used the misspelled field `coeffficients` on three cases, so those failed the output schema and stayed unresolved. The remaining incorrect candidates were contradicted by measurement; valid JSON was never treated as sufficient evidence.

Paired gains over each seed's initialization were 7, 2 and 2 worlds, with no losses because initialization solved none. Only seed 17 exceeded the predeclared 10-percentage-point improvement threshold. **No seed reached the required 25% verified-success threshold.** All exceeded the 90% output-validity threshold. The overall capability gate is therefore false; it was not relaxed after observing results.

## Unseen-family results and verification controls

Every trained model produced contract-valid output on all 64 cubic OOD worlds, but **0/64 predictions per model passed measurement**. The deterministic degree-2 reference also failed all 64 cubic targets. A degree-2 interpolant matching three observations cannot establish a cubic law.

The dataset audit found one exact OOD/training prompt overlap: different underlying worlds can share the three observed points and target x. That ambiguity was retained and disclosed in the split manifest. OOD labels did not enter training or checkpoint selection.

Across all 896 investigations, **zero VERIFIED claims lacked passing checks**. No fabricated evidence alias was accepted. This supports the implemented control boundary on these cases; it is not a universal guarantee about arbitrary future tools or scientific claims.

## Why low token loss was misleading

Selected validation response NLL was approximately 0.0799, 0.0925 and 0.1026 for seeds 17/23/41. Yet validation measurement agreement was only 4/64, 3/64 and 3/64. Formatting and numerical correctness must therefore be measured separately.

A post-training diagnostic on validation outputs, never used to reselect models, counted exact coefficient recovery:

| Seed | Correct c0 | Correct c1 | Correct c2 | Entire coefficient vector correct |
|---|---:|---:|---:|---:|
| 17 | 63/64 | 10/64 | 19/64 | 3/64 |
| 23 | 64/64 | 7/64 | 15/64 | 2/64 |
| 41 | 51/64 | 7/64 | 18/64 | 2/64 |

**Inference:** the model handles the directly observed intercept much better than coefficients requiring arithmetic across observations. This is consistent with learning copying/formatting more readily than numerical reconstruction. It does not prove a specific internal algorithm. Whole-vector correctness can differ from one-target agreement because different polynomials can coincide at one target.

## Runtime and failures

Per-seed training, including scheduled validation and checkpoint writes, took 54.31, 52.44 and 52.32 seconds on the local CPU host. The 896-case holdout evaluation took 117.97 seconds. These are actual local experiment wall times, not VIT measurements or large-model throughput estimates; they include harness/I/O overhead and are not sustained cluster benchmarks.

No infrastructure exception interrupted the dataset/training/holdout runs. The retained **experimental failure is poor numerical generalization and the missed capability gate**. Schema failures, contradictions and all raw outputs remain available; they were not removed from denominators.

One component push returned a GitHub Internal Server Error and an earlier push reported a transport timeout. Subsequent normal pushes succeeded and preserved the component commit history. No force push or rewrite was used.

## Decision and next experiment

**Decision:** retain the deterministic Researcher as the default. Keep these learned checkpoints as explicit experimental backends. Do not present Phase 2A as a successful reasoning model, raise parameter scale because of formatting success, or feed the weak outputs into a blended reward and call that integration validated.

**Recommended next experiment:** add independently checkable arithmetic/process supervision and coefficient-specific evaluation. Compare plain response SFT with worked finite-difference traces at matched model/data/compute budgets. For observations at x=0,1,2, useful exact teacher quantities are Δ1=y1−y0, Δ2=y2−2y1+y0, c2=Δ2/2 and c1=Δ1−c2. Treat these as an experimental training representation, not a production shortcut that quietly computes the answer for the neural model.

Predeclare the next protocol and reserve fresh coefficient worlds not used in this phase's training, validation or inspected test set. Keep the current test as an exposed regression diagnostic. Add tested continuation support to the research-specific trainer before longer runs. Larger model scales, preference/RLVR integration, Judge shift handling and physical VIT measurements remain separate gates.

## Component commits

- `20492e7` — protocol and fixed experiment configuration.
- `379c445` — agent output contract, evidence binding and trace integration.
- `4685f89` — deterministic research datasets and split tests.
- `f0b52df` — structured SFT and frozen checkpoint selection.
- `65029fc` — independent held-out loop evaluation and scoring tests.
- `359ce38` — structured Researcher technical guide.

Results and navigation updates follow in separate commits. [phase-2a-files.json](phase-2a-files.json) records the exact source, documentation and evidence files changed relative to phase 1.
