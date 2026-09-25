# Phase 2B — separate Judge calibration and decision study

**Completed 2026-09-25. Evidence gate: FAILED. Default Judge unchanged.**

## What was built and what works

- An explicit pre-measurement input contract: observed points, candidate coefficients and target coordinate only. Extra outcome/metadata fields, invalid numbers, duplicate coordinates and non-past observations are rejected.
- Fresh grouped synthetic worlds with intentional three-point ambiguity, separate calibration and held-out family/prior shift. Source spans and measurements are retained in Memory; the existing independent MeasurementVerifier produces resolved labels.
- Twelve random-initialization feature Judges: five versus seven features, log versus Brier loss, three seeds. The two feature networks contain **225 and 289 parameters**, respectively. No language model or final-scale Judge was trained.
- Calibration-only temperature fitting, frozen checkpoint/data identities, raw/scaled forecasts, constant and deterministic baselines, utility/risk/coverage metrics, family/observation-count slices and whole-group bootstrap comparisons.
- An opt-in Judge adapter exercised through the existing Controller, plus a full reproduction driver and a post-run evidence auditor. The default reference behavior and previous checkpoints remain unchanged.

This is supervised probability estimation and a one-step decision experiment. It is not sequential RLCD, a proprietary algorithm reproduction, or integrated RLHF+RLVR+RLCD training.

## Registered design and reproducibility

The [protocol](../docs/experiments/phase-2b-protocol.md) and [configuration](../configs/judge-shift.json) were committed as `2c388a1` before dataset generation/training. The full run used implementation commit `f2fa192627958c1b994feada6a7bee4ff196a0d9`:

`runs/20260925T141603-judge-shift-reproduction-a1a228f5`

From `model-1/`: `.venv/bin/python -m aim.judge_shift_reproduce`.
Read the [reproduction and interface guide](../docs/JUDGE_SHIFT.md) before using a new checkpoint. The legacy `--judge` loader remains specific to the old checkpoint format; use the documented `ShiftJudge` interface for these experimental models.

The fixed reference Researcher generates linear and quadratic candidates. All variants and candidates for a base polynomial remain in one split. There are **128/64/64/64 base groups** in training/calibration/test/OOD, producing **512/256/256/512 candidate records**. The audit covers all **1,536 records**, with **768 held-out records** per forecast variant. Twenty-four model variants (raw/scaled) and two constant forecasts yield **19,968 held-out probabilities**, not 19,968 independent worlds or full Controller runs.

Base intercepts 30..90 are outside the earlier fixture ranges. Train/calibration/test contain unperturbed and cubic-perturbed worlds; OOD contains unperturbed and previously unseen quartic-perturbed worlds. Half the groups have three observations and half four. OOD changes both family and mixture proportion, so these effects are not isolated. The actual training positive rate was **0.40234375**, familiar test **100/256**, and OOD **96/512**. Accidental target matches are retained according to the defined verifier event.

Each model used CPU FP32, one torch thread, 600 replacement-sampled batches of 64, AdamW learning rate .01/weight decay .01, clipping 1, and seeds 17/23/41. Log and Brier fits are separate. Temperature is chosen only by calibration log loss over the fixed grid. There is no test checkpoint or threshold selection and no combined reward.

## Actual tests and audit

**102 tests passed, zero skips, in 11.996 seconds.** The ten added tests cover strict inputs, exact residuals, outcome-field exclusion, grouped ambiguities, source-backed labels, altered feature/data rejection, paired initialization/parameter counts, frozen holdout integrity, grouped intervals, policy arithmetic, and the real Controller adapter's verification/abstention/conflicting-source behavior.

A development integration test initially called `Controller.run` with a directory instead of its required Run object. It failed with `AttributeError`; the test harness was corrected and the focused suite passed before the full registered run. The [failed development log](phase-2b-evidence/development-failure.log) is retained. It was not a failed training experiment. No registered-run infrastructure failure or invalidated attempt occurred.

The post-run exporter independently checked all source/measurement spans, recomputed synthetic measurements and verifier labels for 1,536 rows, confirmed group separation and feature reconstruction, checked checkpoint/data identities, recomputed forecast metrics, deterministic policy metrics and all primary bootstrap contrasts, and confirmed the source database was unchanged. All twelve fit source snapshots share hash `9bf09c463435a00acab42f5d4465411ff761fcc71d35e0e7679488b937f0a5e5`.

Frozen selection SHA-256: `b8a7af05efde98d55489163bc83962c5d23c602e1e943809288be5cef74dcd4d`.
Holdout SHA-256: `c6a4c4c987a76a3d44a3769c3f6875513055155bd6adcf920c3ae18ffe4850c6`.
The original source archives, optimizer/RNG states, checkpoints and all failed predictions remain in the local run directory. Portable evidence retains their hashes and manifests.

## Results

Lower Brier is better. Utility below uses the registered illustrative **per-candidate cost .5**: VERIFY returns Y−.5, ABSTAIN returns zero. It does not estimate actual shared-tool costs in the Controller.

| Primary temperature-scaled forecast | Familiar Brier | Familiar utility | OOD Brier | OOD utility |
|---|---:|---:|---:|---:|
| Five features, log, seed17 | .15459 | .09180 | .08977 | .04688 |
| Five features, log, seed23 | .14720 | .07813 | .11539 | −.02344 |
| Five features, log, seed41 | .13659 | .08984 | .08889 | .02930 |
| Rich features, log, seed17 | .11430 | .09375 | .09789 | .00977 |
| Rich features, log, seed23 | .11612 | .09375 | .10292 | .00977 |
| Rich features, log, seed41 | .11722 | .09375 | .10998 | −.02344 |
| Training base rate | .23817 | .00000 | .19850 | .00000 |
| Constant half | .25000 | −.10938 | .25000 | −.31250 |

Rich mean familiar Brier improved from **.14612 to .11588**, while OOD mean Brier worsened from **.09802 to .10360**. Familiar utility improved from .08659 to .09375, while OOD mean utility fell from .01758 to **−.00130**. Sample standard deviations and every raw/scaled model are in the [complete metric tables](phase-2b-metrics.md); no favorable post-hoc variant is promoted.

### Registered gate

Rich/log/temperature seeds 17 and 23 passed all six provisional criteria. Seed41 passed familiar Brier/coverage/utility, OOD Brier and high-confidence error limits, but **failed OOD utility: −.0234375 < 0**. Since every seed was required to pass, the evidence gate failed.

All forecast variants recorded zero incorrect probabilities >=.95. That does not establish calibration or safe decisions below .95: seed41's negative OOD utility occurred without such extreme-confidence errors. A single high-confidence-error count would miss this failure.

### Paired uncertainty

The preregistered rich-minus-five Brier differences on familiar worlds were −.04029, −.03108 and −.01937. Their 500-draw whole-group percentile intervals were [−.04986,−.03042], [−.04278,−.01994] and [−.02831,−.01094]. On OOD worlds, differences were +.00812, −.01246 and +.02109: the feature benefit did not transfer consistently. Full paired Brier and utility intervals are retained. They are descriptive finite-fixture intervals, not a multiple-testing-adjusted claim of general superiority.

### Temperature scaling and alternative objectives

All three rich/log models selected T=1, so their scaled and raw forecasts are identical. Five/Brier models selected T=1.5; this worsened both familiar and OOD Brier/log loss relative to their raw versions despite the calibration-set selection. Other temperature effects were mixed. Scalar temperature scaling therefore supplied no consistent holdout improvement in this run.

At cost .5, positive temperature scaling cannot change the sign of a logit, so it cannot change these decisions; raw/scaled utility is accordingly identical at that threshold. Utilities at costs .2 and .8 are also recorded. Brier-trained alternatives sometimes scored better, but the registered primary comparison remains log-trained. Their complete results are included rather than selecting the best seed/objective after evaluation.

### What the slice diagnostics show

For rich/log models, the familiar three-observation slice had utility **0** for every seed; the four-observation slice had **.1875**. On OOD, three-observation utility was **−.07422/−.07422/−.14063**, while four-observation utility was **.09375** for all three seeds.

This is consistent with the constructed information boundary: different worlds agree at three observed coordinates, while the fourth observation reveals residuals for the specified perturbations. The data supply that fourth observation without charging for its acquisition. These slices do not establish the value of a learned information-acquisition policy.

The simple observed-fit policy achieved familiar utility .09375, exactly matching every primary rich model, but OOD utility −.046875. Its familiar success count was 96, leaving four target matches unselected because observation fit and target passage are different events. A learned policy's confidence cannot replace either scoped check.

At cost .8 the rich models selected only correct predictions on these held-out records, at familiar/OOD coverage .1875/.09375. This is a recorded secondary operating point, not evidence that increasing the threshold universally solves shift or a reason to change the preregistered gate.

## Compute, limits and decisions

The full reproduction took **18.24 seconds** on the recorded macOS arm64 CPU with Python3.12.14/PyTorch2.8.0. Dataset creation: 1.90 seconds; twelve fits and calibration: 2.48 seconds; evaluation: .67 seconds. Most elapsed time was preflight testing. These tiny feature networks do not measure transformer throughput, VIT hardware, cluster scaling or general scientific reasoning.

**Evidence-backed decision:** retain the default verification-first Judge. Keep the richer input contract and both proper-score trainers as opt-in research infrastructure. Better familiar forecasts did not establish reliable decisions under the combined family/prior shift. The result neither rejects the separate-Judge architecture nor establishes a final Judge size.

**Recommended next experiment — Phase 2B.1:** specify and test a Controller-compatible decision policy with explicit shared-measurement and additional-observation costs. Compare verify-all, abstain, observed-fit and calibrated forecasts; account for duplicate/equivalent candidate predictions and define utility at the actual research-question level. Evaluate whether obtaining a fourth observation is worth its cost on fresh grouped worlds. Keep policy development separate from a new sealed evaluation, and keep the Researcher fixed initially. Changing family and prior independently would help distinguish their effects.

**Unresolved:** calibration under unknown mixture shifts; transfer to learned Researcher candidates; language/evidence representations; censored or unavailable outcomes; robust cost choices; sequential credit assignment; and physical training feasibility. These require new protocols rather than retuning on this exposed holdout. No major architecture change, default promotion, large model training or joint reward is approved by this run.

## Artifacts and files changed

- [Audited results, configurations and manifests](phase-2b-evidence/results.json)
- [Every held-out forecast and row/group identity](phase-2b-evidence/predictions.json)
- [Full raw/scaled tables and grouped intervals](phase-2b-metrics.md)
- [Actual 102-test preflight log](phase-2b-evidence/tests.log)
- [Retained development test failure](phase-2b-evidence/development-failure.log)
- [Export integrity manifest](phase-2b-evidence/export-manifest.json)
- [Exact changed-file inventory against `2c44a95`](phase-2b-files.json)

The method references are linked in the preregistered protocol. No externally reported benchmark or model result is attributed to this local experiment. Historical PDFs, prior evidence and canonical evaluation sets were preserved.
