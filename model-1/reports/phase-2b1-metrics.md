# Phase 2B.1 — complete measured policy tables

Each scenario averages 32 base groups with the registered base/perturbation weights. Utility uses question reward capped at one, target cost .5 and acquisition cost .1. Scenarios reuse the same 1,248 actual episodes; they are not independent datasets.

## cubic-prior0.25

| Policy | Utility | Question success | Target dispatch rate | Acquisition rate | Mean tool actions |
|---|---:|---:|---:|---:|---:|
| abstain_all | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.000000 |
| acquire_fit | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| acquire_judge-seed17 | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| acquire_judge-seed23 | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| acquire_judge-seed41 | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| fit_direct | -0.179688 | 0.320312 | 1.000000 | 0.000000 | 3.000000 |
| judge_direct-seed17 | -0.070312 | 0.117188 | 0.375000 | 0.000000 | 2.375000 |
| judge_direct-seed23 | -0.078125 | 0.218750 | 0.593750 | 0.000000 | 2.593750 |
| judge_direct-seed41 | -0.156250 | 0.296875 | 0.906250 | 0.000000 | 2.906250 |
| selective_judge-seed17 | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| selective_judge-seed23 | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| selective_judge-seed41 | 0.025000 | 0.250000 | 0.250000 | 1.000000 | 3.250000 |
| verify_all | -0.179688 | 0.320312 | 1.000000 | 0.000000 | 3.000000 |

## cubic-prior0.5

| Policy | Utility | Question success | Target dispatch rate | Acquisition rate | Mean tool actions |
|---|---:|---:|---:|---:|---:|
| abstain_all | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.000000 |
| acquire_fit | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| acquire_judge-seed17 | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| acquire_judge-seed23 | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| acquire_judge-seed41 | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| fit_direct | 0.046875 | 0.546875 | 1.000000 | 0.000000 | 3.000000 |
| judge_direct-seed17 | 0.015625 | 0.203125 | 0.375000 | 0.000000 | 2.375000 |
| judge_direct-seed23 | 0.046875 | 0.343750 | 0.593750 | 0.000000 | 2.593750 |
| judge_direct-seed41 | 0.046875 | 0.500000 | 0.906250 | 0.000000 | 2.906250 |
| selective_judge-seed17 | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| selective_judge-seed23 | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| selective_judge-seed41 | 0.150000 | 0.500000 | 0.500000 | 1.000000 | 3.500000 |
| verify_all | 0.046875 | 0.546875 | 1.000000 | 0.000000 | 3.000000 |

## quintic-prior0.25

| Policy | Utility | Question success | Target dispatch rate | Acquisition rate | Mean tool actions |
|---|---:|---:|---:|---:|---:|
| abstain_all | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.000000 |
| acquire_fit | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| acquire_judge-seed17 | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| acquire_judge-seed23 | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| acquire_judge-seed41 | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| fit_direct | -0.250000 | 0.250000 | 1.000000 | 0.000000 | 3.000000 |
| judge_direct-seed17 | -0.093750 | 0.093750 | 0.375000 | 0.000000 | 2.375000 |
| judge_direct-seed23 | -0.148438 | 0.148438 | 0.593750 | 0.000000 | 2.593750 |
| judge_direct-seed41 | -0.226562 | 0.226562 | 0.906250 | 0.000000 | 2.906250 |
| selective_judge-seed17 | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| selective_judge-seed23 | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| selective_judge-seed41 | -0.350000 | 0.250000 | 1.000000 | 1.000000 | 4.000000 |
| verify_all | -0.250000 | 0.250000 | 1.000000 | 0.000000 | 3.000000 |

## quintic-prior0.5

| Policy | Utility | Question success | Target dispatch rate | Acquisition rate | Mean tool actions |
|---|---:|---:|---:|---:|---:|
| abstain_all | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.000000 |
| acquire_fit | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| acquire_judge-seed17 | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| acquire_judge-seed23 | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| acquire_judge-seed41 | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| fit_direct | 0.000000 | 0.500000 | 1.000000 | 0.000000 | 3.000000 |
| judge_direct-seed17 | 0.000000 | 0.187500 | 0.375000 | 0.000000 | 2.375000 |
| judge_direct-seed23 | 0.000000 | 0.296875 | 0.593750 | 0.000000 | 2.593750 |
| judge_direct-seed41 | 0.000000 | 0.453125 | 0.906250 | 0.000000 | 2.906250 |
| selective_judge-seed17 | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| selective_judge-seed23 | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| selective_judge-seed41 | -0.100000 | 0.500000 | 1.000000 | 1.000000 | 4.000000 |
| verify_all | 0.000000 | 0.500000 | 1.000000 | 0.000000 | 3.000000 |

## Paired utility contrasts: acquire_judge minus judge_direct

500 paired whole-group bootstrap draws, seed101. Percentile intervals are descriptive; degenerate intervals in the balanced quintic scenario reflect the designed identical-input accounting identity.

| Scenario | Seed | Difference | Percentile95 |
|---|---:|---:|---|
| cubic-prior0.25 | 17 | 0.095313 | [0.040625, 0.150000] |
| cubic-prior0.25 | 23 | 0.103125 | [0.017188, 0.173438] |
| cubic-prior0.25 | 41 | 0.181250 | [0.091211, 0.251563] |
| cubic-prior0.5 | 17 | 0.134375 | [0.103125, 0.150000] |
| cubic-prior0.5 | 23 | 0.103125 | [0.040625, 0.150000] |
| cubic-prior0.5 | 41 | 0.103125 | [0.040625, 0.150000] |
| quintic-prior0.25 | 17 | -0.256250 | [-0.295312, -0.217187] |
| quintic-prior0.25 | 23 | -0.201562 | [-0.240625, -0.162500] |
| quintic-prior0.25 | 41 | -0.123437 | [-0.154687, -0.100000] |
| quintic-prior0.5 | 17 | -0.100000 | [-0.100000, -0.100000] |
| quintic-prior0.5 | 23 | -0.100000 | [-0.100000, -0.100000] |
| quintic-prior0.5 | 41 | -0.100000 | [-0.100000, -0.100000] |

## Seed dispersion

| Scenario | Policy | Mean utility | Sample SD |
|---|---|---:|---:|
| cubic-prior0.25 | judge_direct | -0.101562 | 0.047522 |
| cubic-prior0.25 | acquire_judge | 0.025000 | 0.000000 |
| cubic-prior0.25 | selective_judge | 0.025000 | 0.000000 |
| cubic-prior0.5 | judge_direct | 0.036458 | 0.018042 |
| cubic-prior0.5 | acquire_judge | 0.150000 | 0.000000 |
| cubic-prior0.5 | selective_judge | 0.150000 | 0.000000 |
| quintic-prior0.25 | judge_direct | -0.156250 | 0.066750 |
| quintic-prior0.25 | acquire_judge | -0.350000 | 0.000000 |
| quintic-prior0.25 | selective_judge | -0.350000 | 0.000000 |
| quintic-prior0.5 | judge_direct | 0.000000 | 0.000000 |
| quintic-prior0.5 | acquire_judge | -0.100000 | 0.000000 |
| quintic-prior0.5 | selective_judge | -0.100000 | 0.000000 |

## Acquisition-price sensitivity for realized acquire_judge actions

All three acquire_judge seeds and acquire_fit/selective_judge made matching dispatch choices on this fixture. This table changes prices only; it does not retune actions.

| Scenario | cA=0 | .02 | .1 | .3 | .6 |
|---|---:|---:|---:|---:|---:|
| cubic-prior0.25 | 0.125000 | 0.105000 | 0.025000 | -0.175000 | -0.475000 |
| cubic-prior0.5 | 0.250000 | 0.230000 | 0.150000 | -0.050000 | -0.350000 |
| quintic-prior0.25 | -0.250000 | -0.270000 | -0.350000 | -0.550000 | -0.850000 |
| quintic-prior0.5 | 0.000000 | -0.020000 | -0.100000 | -0.300000 | -0.600000 |

Source: [audited results](phase-2b1-evidence/results.json) and [all episodes](phase-2b1-evidence/episodes.json).
