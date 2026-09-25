# Phase 2B — all forecast variants

Measured from the frozen Phase 2B run. Lower Brier/log loss is better; higher utility is better. Utility uses illustrative per-candidate cost 0.5, not shared Controller measurement cost. No variant is selected after this table. All other costs and slices are in [results.json](phase-2b-evidence/results.json).

| Forecast | Familiar Brier | Familiar NLL | Familiar utility | Familiar coverage | OOD Brier | OOD NLL | OOD utility | OOD coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_rate | 0.23817 | 0.66931 | 0.00000 | 0.00% | 0.19850 | 0.58893 | 0.00000 | 0.00% |
| constant_half | 0.25000 | 0.69315 | -0.10938 | 100.00% | 0.25000 | 0.69315 | -0.31250 | 100.00% |
| five-brier-seed17-raw | 0.12202 | 0.42175 | 0.09180 | 29.30% | 0.09179 | 0.27365 | 0.02734 | 17.19% |
| five-brier-seed17-temperature | 0.13294 | 0.43452 | 0.09180 | 29.30% | 0.09851 | 0.29546 | 0.02734 | 17.19% |
| five-brier-seed23-raw | 0.12190 | 0.42338 | 0.09180 | 25.39% | 0.08967 | 0.26955 | 0.04297 | 10.94% |
| five-brier-seed23-temperature | 0.13248 | 0.43389 | 0.09180 | 25.39% | 0.09713 | 0.29247 | 0.04297 | 10.94% |
| five-brier-seed41-raw | 0.12677 | 0.43441 | 0.09180 | 21.48% | 0.08614 | 0.26290 | 0.04492 | 10.16% |
| five-brier-seed41-temperature | 0.13645 | 0.44228 | 0.09180 | 21.48% | 0.09454 | 0.28743 | 0.04492 | 10.16% |
| five-log-seed17-raw | 0.15459 | 0.47710 | 0.09180 | 18.36% | 0.08977 | 0.28065 | 0.04688 | 9.38% |
| five-log-seed17-temperature | 0.15459 | 0.47710 | 0.09180 | 18.36% | 0.08977 | 0.28065 | 0.04688 | 9.38% |
| five-log-seed23-raw | 0.15232 | 0.46915 | 0.07812 | 54.69% | 0.11515 | 0.33513 | -0.02344 | 37.50% |
| five-log-seed23-temperature | 0.14720 | 0.45518 | 0.07812 | 54.69% | 0.11539 | 0.32788 | -0.02344 | 37.50% |
| five-log-seed41-raw | 0.14394 | 0.45426 | 0.08984 | 31.25% | 0.09423 | 0.28969 | 0.02930 | 16.41% |
| five-log-seed41-temperature | 0.13659 | 0.43950 | 0.08984 | 31.25% | 0.08889 | 0.27146 | 0.02930 | 16.41% |
| rich-brier-seed17-raw | 0.11388 | 0.36252 | 0.09375 | 47.66% | 0.09996 | 0.30486 | 0.00000 | 28.12% |
| rich-brier-seed17-temperature | 0.11388 | 0.36252 | 0.09375 | 47.66% | 0.09996 | 0.30486 | 0.00000 | 28.12% |
| rich-brier-seed23-raw | 0.11463 | 0.36476 | 0.09375 | 41.41% | 0.09564 | 0.29197 | 0.00977 | 24.22% |
| rich-brier-seed23-temperature | 0.11324 | 0.35582 | 0.09375 | 41.41% | 0.09427 | 0.27458 | 0.00977 | 24.22% |
| rich-brier-seed41-raw | 0.12224 | 0.37998 | 0.09375 | 56.25% | 0.12629 | 0.35629 | -0.04688 | 46.88% |
| rich-brier-seed41-temperature | 0.12224 | 0.37998 | 0.09375 | 56.25% | 0.12629 | 0.35629 | -0.04688 | 46.88% |
| rich-log-seed17-raw | 0.11430 | 0.36184 | 0.09375 | 41.41% | 0.09789 | 0.30285 | 0.00977 | 24.22% |
| rich-log-seed17-temperature | 0.11430 | 0.36184 | 0.09375 | 41.41% | 0.09789 | 0.30285 | 0.00977 | 24.22% |
| rich-log-seed23-raw | 0.11612 | 0.36362 | 0.09375 | 41.41% | 0.10292 | 0.30495 | 0.00977 | 24.22% |
| rich-log-seed23-temperature | 0.11612 | 0.36362 | 0.09375 | 41.41% | 0.10292 | 0.30495 | 0.00977 | 24.22% |
| rich-log-seed41-raw | 0.11722 | 0.36553 | 0.09375 | 51.56% | 0.10998 | 0.31617 | -0.02344 | 37.50% |
| rich-log-seed41-temperature | 0.11722 | 0.36553 | 0.09375 | 51.56% | 0.10998 | 0.31617 | -0.02344 | 37.50% |

All evaluated forecast variants had zero wrong predictions with probability >=0.95. This does not establish calibration at lower confidences or zero OOD decision risk.

## Primary paired differences: rich/log/temperature minus five/log/temperature

Intervals are 500 whole-base-group bootstrap percentile intervals, seed101. They are descriptive; there is no multiple-comparison significance claim.

| Split | Seed | Brier difference [95%] | Utility difference [95%] |
|---|---:|---|---|
| ood | 17 | 0.00812 [-0.00032, 0.01768] | -0.03711 [-0.05571, -0.01953] |
| ood | 23 | -0.01246 [-0.01800, -0.00622] | 0.03320 [0.01953, 0.04883] |
| ood | 41 | 0.02109 [0.01210, 0.03091] | -0.05273 [-0.07524, -0.03413] |
| test | 17 | -0.04029 [-0.04986, -0.03042] | 0.00195 [0.00000, 0.00586] |
| test | 23 | -0.03108 [-0.04278, -0.01994] | 0.01562 [0.00391, 0.03032] |
| test | 41 | -0.01937 [-0.02831, -0.01094] | 0.00391 [0.00000, 0.00977] |

## Seed dispersion for primary variants

| Split | Features | Mean Brier | Sample SD | Mean utility | Sample SD |
|---|---|---:|---:|---:|---:|
| test | five | 0.14612 | 0.00905 | 0.08659 | 0.00739 |
| test | rich | 0.11588 | 0.00147 | 0.09375 | 0.00000 |
| ood | five | 0.09802 | 0.01505 | 0.01758 | 0.03659 |
| ood | rich | 0.10360 | 0.00607 | -0.00130 | 0.01917 |
