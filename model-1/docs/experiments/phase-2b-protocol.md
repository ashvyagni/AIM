# Phase 2B — separate Judge calibration and measurement decisions

Registered before generating data or training. Version `aim-judge-shift-v1`.
This is a supervised forecast/one-step decision experiment within the approved
separate Judge architecture, not joint RL or a reproduction of proprietary RLCD.

## Event, information and hypotheses

Forecast Y=1 when a fixed reference Researcher's candidate prediction agrees with
the next synthetic measurement at absolute tolerance 1e-8. Only observed points,
candidate coefficients and target coordinate enter the feature extractor. Hidden
world coefficients, family, split, group/record identifiers, labels, future
measurements and post-verification state cannot enter model features. Full
provenance remains in an audit store. The label is produced by the existing
MeasurementVerifier using a real stored synthetic measurement.

Hypotheses: exact observed-fit features improve proper scores and one-step
verification allocation; validation temperature scaling helps familiar forecasts
but need not survive changed family mixtures. Report negative results. Neither
observation fit nor model confidence establishes a unique global law.

## Fresh grouped worlds

Base polynomial q has c0 in 30..90, c1 in -20..20, and c2 in -3..3. Half the base
groups are linear (c2=0), half quadratic (c2 nonzero). Rank each stratum by hash
of version and coefficients, then allocate 128 training, 64 calibration, 64 test,
64 OOD base groups consecutively, preserving equal strata. Positive c0>=30 makes
these worlds distinct from all preceding Phase 1 and Phase 2A fixture ranges.

Train/calibration/test groups each contain two worlds: q and
q+k*x*(x-1)*(x-2), where k is -1 or +1 by group hash. OOD groups each contain q
and three quartic worlds q+k*x*(x-1)*(x-2)*(x+1), k in {-2,1,2}.
Thus quartic perturbations are held out from both training and calibration;
cubic shift is explicitly now a known training family. OOD also changes the
unperturbed-world proportion from 1/2 to 1/4; family and prior shift effects are
not isolated. All variants of one base group stay in the same split.

Observe x=0,1,2 or x=0,1,2,3, balanced within each split/stratum by ranked position.
Choose target x from 4,5,6 by group hash. Fixed PolynomialResearcher produces
linear and quadratic candidates using only observations. Keep both candidates,
including equal predictions: candidate dependence and indistinguishable
three-point worlds are intentional. Bootstrap entire base groups, not rows.

Training file contains train and calibration rows only, with features recomputed
from an allowlisted pre-measurement payload and declared programmatic labels.
Holdout and world-coefficient audit are separate files. Freeze model/checkpoint,
temperature and policy configuration hashes before loading holdout. Training must
not open holdout/audit files. This is code-level separation, not a sealed service.

## Models, baselines and budget

Keep the Researcher fixed. Train random-init tanh MLP Judges with 32 hidden units:
five existing features (225 parameters) versus those five plus exact finite
observation-fit indicator and log1p(max normalized residual) (289 parameters).
The fit feature is a numerical feature from observations, not provenance approval.
Compare log loss and Brier loss separately, seeds 17,23,41: twelve fitted models.
CPU FP32, one torch thread, AdamW lr0.01/weight_decay0.01, clip1, 600 minibatches
of 64 sampled with replacement. No early selection or adaptive budget extension.
Save optimizer and RNG as evidence; new-study resume support is not claimed.

L_log=-mean[Y log p+(1-Y)log(1-p)]; L_Brier=mean[(p-Y)^2].
For each final model report raw p=sigmoid(z) and temperature-scaled p=sigmoid(z/T).
Fit T on calibration log loss only from {0.5,0.75,1,1.25,1.5,2,3,5}; tie by listed
order. No test tuning. Baselines: p=0.5, empirical training positive rate,
verify-all, abstain-all, and verify-if-observed-fit deterministic policy. The last
three are policies, not calibrated forecasts.

## Decisions, metrics and uncertainty

Toy per-candidate utility: U(VERIFY,Y)=Y-c; U(ABSTAIN,Y)=0, for fixed costs
c in {0.2,0.5,0.8}. Analytical policy VERIFY iff p>=c; no threshold tuning.
Report Brier, log loss, accuracy, fixed-threshold risk/coverage, high-confidence
wrong counts, utility, verification fraction and forgone successes. Missing
outcomes are never negative labels; this complete-outcome fixture has none.
These costs are illustrative independent candidate work units. The Controller
shares a measurement across candidates, so these utilities are not measured
Controller cost savings or an optimal sequential policy.

Report all models/splits and observation-count/family slices. Primary contrast:
rich/log/temperature versus five/log/temperature, paired by seed and world.
Use 500 bootstrap draws of whole base groups, seed101, percentile 95% intervals
for Brier difference and utility difference at c=0.5. Report seed dispersion;
multiple seeds do not create new independent worlds. No multiple-testing claim.

Provisional evidence gate for each rich/log/temperature seed: familiar Brier at
least .02 better than training-base-rate forecast; OOD Brier no more than .02
worse than that baseline; familiar utility at c=.5 >= both base-rate and
verify-all; OOD utility >=0; familiar verification coverage >=10%; OOD fraction
of all rows wrong with p>=.95 <=5%. All seeds must pass. These engineering
thresholds do not establish significance. Regardless, no default replacement
without review and broader evidence. Primary paired contrasts explain the gate;
they are not tuned to make it pass.

## Execution and limits

Run unit/integration tests, generate data, fit all models, freeze selection, then
evaluate once. Retain source archives, actual environment, dataset/checkpoint
hashes, seeds, every prediction, failure and metric. Exercise the new Judge
adapter through the existing Controller with success, abstention and conflicting
source cases. Canonical suites and previous checkpoints remain unchanged.

The finite polynomial domain and small feature networks cannot select a final
Judge parameter count, prove natural-language calibration, justify VIT-scale
training, or demonstrate joint RLHF+RLVR+RLCD advantage. Learned Researcher
candidate-distribution transfer and sequential information acquisition remain
separate experiments. Do not train on exposed holdouts after this study and
call them unseen.

## Primary evidence for methods

Gneiting and Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and
Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf),
supports proper-score forecast evaluation. Guo et al. (2017), [On Calibration of
Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), studies
temperature scaling. Neither source establishes that calibration transfers to
this synthetic shift or that our proposed features improve AIM; those are tested
here. The utility threshold follows directly from p-c>=0.
