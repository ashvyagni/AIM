# Phase 2B.1 — shared measurement and paid evidence acquisition

**Completed 2026-09-26 (Asia/Kolkata). Evidence gate: FAILED. Defaults unchanged.**

## What was built

1. A default no-op evidence extension hook in the deterministic Controller, preserving the existing loop.
2. An opt-in paid-evidence Controller that decides before acquisition, obtains x=3 through the bounded MEASURE worker, stores the result as a new immutable observation source, and re-runs the same fixed Researcher.
3. Shared question-level verification decisions with one target measurement for all supported candidates. Individual Judge forecasts retain their original target events; the joint action uses a separately recorded heuristic.
4. Dispatch-based costs and success capped at one per question, including duplicate predictions and failed/unavailable acquisition attempts.
5. Fresh grouped worlds, frozen historical Judge loading, 13 policy variants, factorial family/prior evaluation, whole-group bootstrap contrasts and fixed-action cost sensitivity.
6. A read-only post-run auditor for state replay, source integrity, decision timing, costs, verification backing, frozen identities and aggregate metrics.

This establishes actual evidence acquisition and question-level accounting. No new model is trained, no combined reward is introduced, and no default policy or architectural component is replaced.

## Reproduction and frozen choices

- [Preregistered protocol](../docs/experiments/phase-2b1-protocol.md), committed as `d30f250` before fixture generation/evaluation.
- [Configuration](../configs/paid-evidence.json) and [implementation/reproduction guide](../docs/PAID_EVIDENCE.md).
- Code committed as `44e9dde` before the full run.
- Run: `runs/20260925T193029-paid-evidence-reproduction-751ae7ad` (directory uses UTC).
- Command from `model-1/`: `.venv/bin/python -m aim.paid_evidence_reproduce`.

The three rich/log/temperature Judge checkpoints are exactly those frozen in Phase 2B, seeds17/23/41, each 289 parameters. Hashes and integrity sidecars are required; absent assets stop the new run rather than triggering substitution. A fresh clone needs the matching retained local checkpoints restored. The 225/289-parameter comparison remains a mechanism study, not a final Judge-scale choice.

The same deterministic Researcher produces linear/quadratic candidates. Initial observations are x=0,1,2; purchased evidence is x=3; targets are 4,5,6. The Researcher still fits the first two/three points. The extra point changes observed-residual features and fit checks without injecting corrected coefficients.

Thirty-two fresh base groups, balanced linear/quadratic, each produce a base, cubic-perturbed and quintic-perturbed world: 96 worlds. The known cubic perturbation becomes visible at x=3; the new quintic perturbation vanishes there too. Family and prior effects are separated by reweighting each base/perturbed pair with priors .5 and .25, yielding four scenario views of shared episodes. There is no training or threshold fitting on this fixture.

The fresh intercept range 120..200 also differs from Phase 2B training (30..90). That common covariate change is held fixed across the four scenarios; “known family” does not mean identical feature distribution or in-distribution calibration. This study cannot isolate every source of forecast error relative to historical scores.

## Utility and interpretation

R is 1 iff at least one target claim actually passes the existing provenance and measurement verifiers, otherwise zero. U=R−.5*M−.1*A, with at most one dispatched target measurement M and additional observation A. Unsuccessful dispatched measurements still cost. Duplicate candidates cannot multiply reward; one shared target measurement cannot be charged twice. Costs are illustrative units; calculation/retrieval are uncharged in utility, with actual action counts and tool times retained.

Learned policy aggregation groups exactly equal predicted values, takes each group's maximum candidate probability and clips the sum at one. That is an explicit heuristic, not a calibrated joint forecast. The selective acquisition band .2<p<.8 is also a fixed heuristic, not learned information value. Cost sensitivity re-scores the same actions; it does not pretend policies were optimized for each price.

The quintic construction is deliberately indistinguishable from its base world even after x=3. At balanced prior and target cost .5, these paired inputs yield zero expected target-measurement utility for the fixed candidate family; unconditional acquisition subtracts .1. A gate failure in this stress fixture is therefore an expected information/accounting limit, not an unexpected finding about general intelligence. Only the known-family benefit and policy-specific decisions require empirical comparison here.

## Tests, failures and audit scope

**118 preflight tests passed, zero skips, in 35.698 seconds.** Sixteen new tests cover probability validation, duplicate credit, shared dispatch, stored purchased evidence, temporal inputs, exact band boundaries, unavailable/error/timeout acquisition, missing acquisition contracts, exhausted budgets, conflicting sources, fresh groups, family ambiguity, factorial accounting, frozen Judge hashes and a complete small seven-policy evaluation. Tests use separate unit worlds/checkpoints; the registered fixture was evaluated only after preflight and freezing.

A development assertion initially compared decoded JSON lists directly with in-memory tuples. It was changed to the repository's canonical serialization comparison, consistent with the Controller's replay invariant. The focused tests then passed; the original failed log is retained in the final evidence bundle. No runtime repair of a state mismatch was needed.

Failure behavior is explicit: unknown/error/timeout acquisition stops unresolved without a target fallback; invalid initial evidence prevents dispatch; exhausted budget charges only work actually dispatched. No confidence value bypasses independent claim checks.

## Measured results and decision

All **1,248 actual Controller episodes** completed and passed the read-only audit. They dispatched **3,988 tools**: 2,496 candidate calculations, 672 additional observations and 820 shared target measurements. There were zero non-PASS tool outcomes in the registered fixture and **zero unbacked VERIFIED claims**. Failure behavior was exercised separately by the tests; the all-available fixture does not estimate a real failure rate.

### Question utility

| Policy | Known cubic, prior .5 | Known cubic, prior .25 | New quintic, prior .5 | New quintic, prior .25 |
|---|---:|---:|---:|---:|
| Abstain all | .000000 | .000000 | .000000 | .000000 |
| Verify all / fit direct | .046875 | −.179688 | .000000 | −.250000 |
| Judge direct, seed17 | .015625 | −.070313 | .000000 | −.093750 |
| Judge direct, seed23 | .046875 | −.078125 | .000000 | −.148438 |
| Judge direct, seed41 | .046875 | −.156250 | .000000 | −.226563 |
| Acquire fit / all three acquire Judges / all three selective Judges | .150000 | .025000 | −.100000 | −.350000 |

The paid policies made the same acquisition/target-dispatch choices on these worlds. In the known family, they purchased for every world and measured the target only on the unperturbed fraction, cutting weighted target dispatch to .5 or .25. Their question success was .5 or .25. In the new family, they still purchased for every world and then measured every target, with success only .5 or .25. This separates purchasing evidence from obtaining useful evidence.

**Selectivity did not help:** every initial aggregate for every frozen Judge fell inside the fixed acquisition band, so the selective policies bought in 100% of cases. The learned acquisition variants also did not outperform the simple acquire-fit policy here. These are important negative comparisons; no result is presented as learned information-value estimation.

Direct verify-all found three cubic target coincidences among the 32 perturbed worlds. The acquired observation exposed residuals and the fit-based route abstained on those worlds, even though some target values could have passed. The target-verification event and observation-consistency event remain distinct. Canonical success was not retrospectively redefined to favor acquisition.

### Registered contrasts and gate

For acquire_judge minus judge_direct in the balanced known family, utility gains were:

| Seed | Gain | Paired group percentile95 |
|---|---:|---|
| 17 | .134375 | [.103125, .150000] |
| 23 | .103125 | [.040625, .150000] |
| 41 | .103125 | [.040625, .150000] |

All three exceeded the predeclared .02 improvement requirement. All three failed nonnegative utility in **both** quintic scenarios. The overall gate failed. Balanced quintic contrasts were exactly −.1 for every seed/group, with degenerate [−.1,−.1] intervals, reflecting the designed information/accounting identity rather than a strong empirical generalization claim. Full contrasts, 500-draw bootstrap intervals and seed dispersion are in the [complete tables](phase-2b1-metrics.md).

### Price sensitivity of unchanged actions

At target cost .5, the realized acquire-Judge actions have utility `.25−cA` in the balanced known family and `.125−cA` under its lower base prior. Thus they break even against abstention at acquisition prices .25 and .125. At the registered price .1, utility is .15 and .025; at price .3 it becomes −.05 and −.175.

The balanced quintic scenario yields `−cA`; the low-prior quintic scenario yields `−.25−cA`. Free acquisition alone would not fix the latter policy's bad target decisions. These expressions summarize this fixture's realized actions. They are not universal cost thresholds, and no policies were retuned during the sensitivity analysis.

## Compute and evidence integrity

The full run took **454.53 seconds (7.58 minutes)** on the recorded local macOS arm64 CPU/Python3.12.14/PyTorch2.8.0 environment, with one torch thread. Dataset creation took .14 seconds, evaluation 416.91 seconds. Summed measured tool execution time was 230.47 seconds. The remainder includes provenance, snapshots, scoring, process/bookkeeping overhead and preflight; it is not a separately profiled bottleneck decomposition.

The audit independently replayed each acquisition/target decision from its **preceding observed state and frozen Judge**, checked ledger ordering against actual tool dispatches, replayed final state, validated source spans, rechecked every VERIFIED claim, recomputed capped rewards and attempt costs, and regenerated all factorial metrics and bootstrap contrasts. State and database hashes remained unchanged. Root, evaluation and all 1,248 episode executable-source snapshots match hash `5edce7b02944dcdc054fcf0e24e1a902cd93a22e6a70506fb9fbb44b25f8d31a`. Documentation/report commits made during execution did not change those sources.

Frozen configuration/parents SHA-256: `89504ed186ae5669af55563c7bb76ca0e8ec471d0909f730ff4aeaeacfb20790`.
Fixture SHA-256: `c69c4ea0d34407c3120f8a3559592609fb0e578378b931209f8c7950a7672dfd`.

No registered episode or infrastructure stage failed. The failed development assertion, rejected target claims and failed evidence gate are retained and distinguished. Actual acquisition error/timeout/unavailable handling was verified in unit/integration tests, not observed as random failures in the registered fixture. No VIT node, GPU, cluster or larger-model training measurement is claimed.

## Decision and next work

**Evidence-backed engineering decision:** keep all paid policies opt-in. Retain shared accounting, the deterministic acquisition extension and temporal/provenance audit as reusable infrastructure. The known-family improvement is conditional on the measurement being informative and cheap enough. The learned policies supplied no measured advantage over the simple acquisition/fit rule on this fixture.

**Recommended next implementation: Phase 2C, one bounded symbolic verification domain.** Introduce an explicitly versioned claim contract for exact polynomial/algebraic identities, an independent checker with declared assumptions and resource limits, provenance-linked inputs, and adversarial evaluation. Preserve the numerical prediction loop and canonical suites. This broadens the evidence types the modular system can check, rather than repeatedly tuning acquisition thresholds on exposed synthetic worlds. Begin with contracts and a deterministic end-to-end reference before training a proposer for the new domain.

**Still unresolved:** a coherent question-level forecast model, expected information value, variable probe selection, retries and censored outcomes, learned Researcher transfer, real measurement pricing, human preference data, integrated decision learning and the physical compute budget. A future acquisition-learning experiment needs fresh held-out groups, explicit family assumptions and proper treatment of equivalent candidates; the current stress worlds are now exposed. The historical handbook and separate SFT/preference/RLVR/calibration thesis remain unchanged.

## Exact artifacts and changed files

- [Audited results, manifests and frozen parents](phase-2b1-evidence/results.json)
- [Every episode's decisions, dispatches, costs and state identity](phase-2b1-evidence/episodes.json)
- [Full factorial policy tables, intervals and cost sensitivity](phase-2b1-metrics.md)
- [Actual 118-test preflight](phase-2b1-evidence/tests.log)
- [Retained development assertion failure](phase-2b1-evidence/development-failure.log)
- [Export integrity manifest](phase-2b1-evidence/export-manifest.json)
- [Exact changed-file inventory against `935af7a`](phase-2b1-files.json)

Local run directories retain full source archives, immutable objects, ledgers and responses; the portable Git bundle records inspectable outcomes and hashes. Historical evidence, PDFs and canonical evaluation definitions were preserved.
