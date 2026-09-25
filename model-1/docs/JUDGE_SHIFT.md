# Separate Judge forecasts and decisions under shift

Phase 2B implements the [registered protocol](experiments/phase-2b-protocol.md).
It leaves the default Controller/Judge, earlier checkpoints and canonical suites
unchanged. The opt-in Judge is a feature MLP trained from random initialization,
not a language Judge or a new Researcher. No SFT/preference/RLVR rewards are mixed.

## Reproduce

From `model-1/`, using the recorded environment:

```sh
.venv/bin/python -m aim.judge_shift_reproduce
```

The driver runs the full unit suite, rejects skips/failures, creates fresh grouped
worlds, trains twelve small Judges, freezes checkpoint hashes and temperatures,
then loads holdout and writes every forecast. Runs are retained with source
snapshots, config/environment/commit, dataset hashes, optimizer/RNG states and
explicit completion/failure. Completion does not imply an evidence gate passed.
The fixed new-study trainer does not yet support checkpoint resume.

Read root `summary.json`, `tests.log`, `dataset-stage.json`, `training-stage.json`,
the training `frozen.json`, and evaluation `metrics.json`/`predictions.json`.
Configuration: `configs/judge-shift.json`. Results use the exact version and
dataset hashes; never overwrite a historical experiment or reuse its exposed
holdout for selection.

## Information boundary

`judge_shift_features.extract()` accepts exactly `observations`, `coefficients`
(the proposed candidate), and `target_x`. It rejects extra fields, nonfinite or
Boolean numbers, duplicate coordinates and observations at/after the target.
The runtime adapter projects these fields from state; it does not pass claims,
tool outcomes, world labels or hidden generator coefficients to the network.
Claim identity and candidate prediction must agree before a forecast is attached.

Five features preserve the prior Judge baseline: candidate coefficient count,
observation count, mean normalized residual, target distance, and normalized
candidate prediction magnitude. The seven-feature version additionally receives
an exact finite-fit indicator and maximum normalized residual. The indicator is
a numerical feature; it does not independently approve source provenance.

Source-backed generated observations and exact rational synthetic measurements
are retained in the dataset Memory. `MeasurementVerifier` supplies labels. The
training file contains only training/calibration records. A metadata allowlist
checks the training hash and carries a holdout hash without holdout labels.
Hidden coefficients and measurement/check details live in a separate audit file.
Both hidden variants and both candidates for a base group stay in one split.

The loader recomputes all features from the permitted input and rejects altered
values. Row IDs, families, source records and groups are attribution/splitting
metadata; only the recomputed numerical vectors enter tensor training. This is
an auditable code boundary, not an operating-system sandbox.

## Forecast and decision objectives

For the defined binary future-verifier event Y, p=sigmoid(z). Fit log loss
`-mean(Y*log(p)+(1-Y)*log(1-p))` or Brier `mean((p-Y)^2)` separately. Each
feature/loss/seed run uses a fresh random model, 600 sampled batches of 64, and
the same source training worlds. The five/seven input networks have 225/289
parameters. These numbers establish a mechanism comparison, not final sizing.

Temperature scaling substitutes sigmoid(z/T), fitting only calibration log loss
over the registered temperature grid. It cannot add information absent from z.
Raw and scaled forecasts are both reported; holdout never selects temperature,
steps or a preferred checkpoint. A constant-half and training-base-rate forecast
provide explicit baselines.

For illustrative cost c, VERIFY utility is Y-c and ABSTAIN utility is zero.
Expected VERIFY utility is p-c, so the policy uses p>=c. Costs .2/.5/.8 are fixed
before evaluation. Always-verify, always-abstain and observed-fit policies are
reported separately from probability models. Risk is failure frequency among
selected predictions, and coverage is selected fraction; zero coverage has
undefined risk, not perfect measured accuracy.

These are per-candidate costs with complete offline labels. The existing
Controller shares a measurement across candidates. The study therefore does not
claim actual tool-cost savings, sequential RL, optimal acquisition of new points,
or handling of unresolved/censored real-world outcomes. The runtime adapter's
threshold is an experimental heuristic in that shared-tool environment.

## Integration

The experimental adapter implements the existing Judge interface:

```python
from aim.judge_shift_train import ShiftJudge
from aim.controller import Controller
from aim.tracking import Run
from pathlib import Path

judge = ShiftJudge(checkpoint_path, calibrated=True, cost=0.5)
with Run(Path("runs"), "shift-judge-loop", {"cost": 0.5}, inputs=[checkpoint_path]) as run:
    state = Controller(judge=judge).run(case, run)
```

Use a real checkpoint from the frozen manifest and a valid existing case. The
legacy `--judge` CLI loader remains specific to the older five-feature checkpoint
format. Unit integration executes the new adapter through the actual Controller
with both verification and abstention, plus conflicting evidence. Confidence
does not bypass independent verifiers or promote an unmeasured claim.

## Shift and interpretation

Three observed points cannot distinguish q from q+k*x*(x-1)*(x-2), or the
registered quartic perturbations. Four points reveal a residual for these
particular alternatives. Half the base groups receive each observation count.
The OOD split changes both polynomial family and base/perturbed mixture, so the
experiment measures their combined effect. It does not isolate which shift
caused a score change. See [finite-observation scope](OBSERVATION_SCOPE.md).

Metrics include proper scores, risk/coverage, high-confidence errors, costs,
forgone successes and slices by family/observation count. Primary differences
use paired seeds and 500 whole-base-group bootstrap draws; candidate rows and
indistinguishable variants are dependent. Percentile intervals are descriptive
and do not correct for all reported comparisons.

The preregistered evidence gate compares rich/log/temperature models with the
base-rate and verify-all baselines, including an OOD utility and error constraint.
Passing it would justify further study, not default replacement. Learned
Researcher candidates, language evidence, robust calibration on new domains,
human feedback and sequential decisions require separate experiments.

## Modules

| Module | Purpose |
|---|---|
| `judge_shift_features.py` | Restricted input contract and observed-only features |
| `judge_shift_data.py` | Grouped fresh worlds, Memory evidence, verifier labels |
| `judge_shift_train.py` | Separate proper-score fits, calibration freeze, runtime adapter |
| `judge_shift_eval.py` | Frozen scoring, policies, slices and grouped intervals |
| `judge_shift_reproduce.py` | Tests and retained full experiment |
| `tests/test_judge_shift.py` | Input leakage, data/label invariants and actual Controller cases |

The [protocol](experiments/phase-2b-protocol.md#primary-evidence-for-methods)
links the primary scoring-rule and temperature-scaling literature. These methods
are prior work; AIM's eventual contribution must be supported by measured system
advantages rather than the presence of familiar components.
