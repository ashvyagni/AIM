# Phase 2B.1 — shared measurement and paid evidence acquisition

Registered before new fixture generation or evaluation. Version
`aim-paid-evidence-v1`. Uses the approved separate Researcher/Judge/Controller/
Verifier/Memory architecture; existing defaults and historical suites stay fixed.

## Question and information boundaries

Does purchasing a fourth observation improve question-level verified utility
after its acquisition cost is charged? Do the frozen Phase 2B Judges support
that decision under independently varied family and mixture shifts?

Question reward R is 1 iff at least one candidate's target claim passes the
existing provenance and measurement verifiers, else 0. Multiple candidates or
equal predictions cannot earn extra reward. U=R−0.5*M−0.1*A, where M and A are
numbers of dispatched target-measurement and additional-observation attempts.
At most one of each is allowed. Failed/unavailable attempts cost the same units;
they cannot earn verified credit. Costs are illustrative utility units, not
measured currency, elapsed time or physical instrument charges. Calculation and
local retrieval have no utility charge; report their actual action counts/time.

Only the deterministic Controller sees hidden acquisition/target measurements.
The policy receives past observations, fixed candidate coefficients, target
coordinate and frozen forecasts. Source/group/family identities, coefficients
of the environment and future values never enter policy features. Log each
decision and input before dispatch. Source provenance and scoped verification
remain authoritative. An acquisition failure stops that investigation unresolved;
no hidden fallback or free retry.

## Fresh factorial fixture

Generate 32 base groups, 16 linear and 16 quadratic, hash-ranked by version and
q=(c0,c1,c2), c0 in 120..200, c1 in -20..20, c2 in -3..3. Positive c0>=120
excludes all historical fixture coefficient ranges. Each group has three worlds:
q; q+k*x*(x-1)*(x-2) (known cubic perturbation); and
q+k*x*(x-1)*(x-2)*(x-3)*(x+1) (new quintic perturbation).
Choose k in {-1,1} and target t in {4,5,6} by group hash. All worlds supply only
x=0,1,2 initially; an explicit purchase requests x=3. The quintic perturbation
is deliberately invisible even there. This is a registered information-limit
stress test, not a claim of naturally sampled scientific tasks.

Run all policies on all 96 worlds. Report a 2x2 factorial evaluation: perturbation
family cubic/quintic and unperturbed prior alpha=.5/.25. For each family, weight
the base world alpha and its perturbed counterpart 1−alpha within each group;
then average groups equally. These are four weightings of shared episodes, not
four independent datasets. No training, temperature fitting or threshold tuning.

## Frozen policies and candidate aggregation

Use the fixed PolynomialResearcher with both its linear/quadratic candidates.
An optional Controller extension can obtain x=3 before re-running the same
Researcher and forecasting target passage. Its interpolation continues using
the first two/three points; the new observation changes evidence/check features,
not an oracle coefficient repair.

Four deterministic policies: `verify_all`, `abstain_all`, `fit_direct` (measure
target iff any candidate fits all available observations), and `acquire_fit`
(purchase x=3, then apply fit_direct).

For each seed17/23/41 use the already frozen Phase 2B rich/log/temperature Judge:
`judge_direct`, `acquire_judge` (purchase then forecast), and `selective_judge`
(purchase iff initial aggregate forecast is strictly between .2 and .8, then
forecast again; otherwise decide directly). Total: 13 policy variants, **1,248
actual Controller episodes**. No new learned model is trained or selected.

Aggregate the per-candidate forecasts heuristically: group exactly equal target
predictions, take the maximum probability within each group, then cap the sum
of group probabilities at 1. Measure iff this aggregate >=.5, and request
verification for all supported candidates after one shared target measurement.
Deduplication prevents repeated identical forecasts from multiplying credit.
The aggregate is explicitly a heuristic, not a calibrated question probability:
different candidate forecasts need not form a coherent joint distribution,
and close tolerance intervals can overlap. The .2/.8 acquisition band is a
fixed heuristic, not a learned expected-information-value estimator. Record it
as such. Always preserve the individual forecasts.

Use checkpoint hashes from committed `reports/phase-2b-evidence/results.json`;
require the selected three entries to be rich/log and preserve their identity.
Resolve local checkpoint paths explicitly, check sidecars/hashes before running,
and freeze config, policy list, source evidence hash and dataset hash in the new
run. Missing checkpoints stop the run; never substitute a better-performing
checkpoint or pretrained asset.

## Evaluation, failures and decision

Report question success, utility, purchase and target-dispatch rates, all tool
counts, unresolved cases, source/claim checks, and exact per-episode outcomes.
Retain raw traces and source snapshots. Runtime input validation, acquisition
unavailable/error/timeout, action budget, conflicting sources, invalid forecasts,
and duplicate predictions require tests. No VERIFIED claim without actual
passing independent checks.

Primary contrasts, per seed: acquire_judge minus judge_direct, in all four
family/prior scenarios. Use 500 paired whole-base-group bootstrap draws with
seed101, percentile95 intervals; also report seed dispersion. With only 32
groups, intervals are descriptive and do not establish general capability.
Secondary sensitivity re-scores the **same realized actions** at acquisition
costs 0,.02,.1,.3,.6 with target cost fixed .5. It does not re-optimize policies
or establish that the same actions would be chosen at different costs.

Provisional gate: every acquire_judge seed must improve known-family balanced
utility by at least .02 over its direct counterpart, achieve nonnegative utility
in both quintic scenarios, and produce zero unbacked VERIFIED claims. Any miss
fails the evidence gate. Regardless, promotion/default changes require broader
evidence and review. Keep policies opt-in. No cost or band retuning after seeing
the fixture results.

There is no new sequential training objective or combined RL reward. This phase
establishes actual bounded acquisition/measurement traces and correct accounting
before future decision learning. General language research, learned Researcher
transfer, physical measurement costs and VIT cluster feasibility remain open.
