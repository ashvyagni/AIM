# AIM Research Intelligence Model — Evaluation Specification

## Primary question

Does the complete system produce more independently verified, well-supported research progress at a given compute/tool budget than a strong retrieval-and-tool baseline, and which components cause the gain?

## Evaluation set governance

Create train/dev/sealed test with split boundaries by source, generator, time and task template. Publish item provenance/rights, hashes and scoring scripts. Benchmark operators do not expose sealed labels or hidden tests to training. Every externally maintained benchmark is run under its terms; note possible contamination and report version/date. Use exact same context, tools, wall-clock, number of samples and budget for comparisons.

## Core metrics

- Outcome correctness: exact verifier pass, theorem proof, hidden code tests, statistical recomputation.
- Claim support: supported claim precision/recall and citation span entailment; unsupported-claim rate weighted by severity.
- Research process: appropriate hypothesis set, discriminating predictions, experiment controls/power, contradiction handling, successful revision, negative-result retention.
- Calibration: Brier and log loss on defined resolved events; reliability curves/intercept/slope; ECE secondary; selective risk/coverage and utility under fixed abstention costs.
- Tool performance: action success, invalid call, unnecessary call, recovery, latency/cost, reproducibility.
- Long-horizon: solved subtask coverage, state/provenance retention, missed contradiction, final claim consistency and time/cost.
- Human-facing output: blinded expert preference and factual completeness; normalize/stratify length.

## Benchmark domains

Mathematics; computer science/software; physics; quantum information; astronomy/astrophysics; chemistry; biology; psychology/cognitive science; statistics; engineering; economics; philosophy/logic; scholarly writing; research methodology. Each gets knowledge, reasoning, synthesis, hypothesis/experiment, calibration, citation and tool slices. Report per-domain metrics; aggregate only with stated weights.

Public anchors include GPQA, MATH, FrontierMath (access controlled and benchmark flaw review required), Lean/Mathlib tasks, SWE-bench variants, SciCode, ALCE, PaperBench, and long-context retrieval benchmarks. These complement internally authored longitudinal research tasks; none alone measures scientific discovery.

## Human/judge procedure

For open-ended tasks, two blind domain reviewers grade evidence correctness and research utility using rubric; adjudicate disagreements and preserve inter-rater statistics. LLM grader agreement is assessed on a human-labeled subset and not treated as truth. Score output with citations opened and claim spans visible. Auditors inspect a random sample, all high-impact claims, and disagreement cases.

## Adversarial suites

False premise; non-existent citation; valid citation with unsupported claim; conflicting sources and dates; retracted study; fabricated table; unit mismatch; near-correct algebra; code that passes visible but not hidden tests; confounded experiment; underpowered null result; tool timeout/stale response; source license conflict; prompt injection in paper text; verbose plausible wrong derivation; self-confirming reward exploit.

## Reporting and release gate

Report accuracy and proper scores with bootstrap or suitable confidence intervals, seed counts, per-domain breakdown, all conditions, budget and failures. Release only after verifier audit, reproducibility run, contamination analysis, license review, security review of tools, calibration shift tests and transparent limitations. Do not claim zero hallucinations. Claim measurable reductions in unsupported claims and abstention quality with tested scope only.
