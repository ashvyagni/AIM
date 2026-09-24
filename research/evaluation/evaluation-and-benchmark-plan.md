# Evaluation and benchmark plan

## Principles

Build internal held-out tasks before training. Public benchmarks are diagnostic, not proof of research ability. Separate base knowledge, post-training, retrieval, tool policy, verifier, calibration and system-level effects. Test contamination by source- and time-based splits; preserve a sealed set and third-party red-team items. Report paired runs, confidence intervals, sample size, inference budget and tool cost. Never compare models with different access/budget without disclosure.

## Matrix

| Capability | Measures | Task design |
|---|---|---|
| Knowledge | Accuracy, temporal error, calibration, source support | Domain-stratified factual Qs; date cutoffs; conflict/false premise; with/without retrieval. |
| Mathematics | Exact correctness, proof validity, verifier pass, compute/calls | MATH-style and expert-held-out tasks, symbolic equivalence, Lean proof, novel generator families, partial credit by proof rubric. FrontierMath access/use only under terms; avoid relying on exposed answers. |
| Code | Hidden-test pass, compile, runtime, complexity, security | SWE-bench/Verified where licensed and reproducible, internal issue tasks, property tests; fresh container reproduction. |
| Science | Hypothesis discrimination, experiment validity, power/confound checks | Synthetic causal worlds plus expert-authored novel scenarios; score predictions against held-out environment outcomes. |
| Literature | Supported-claim precision/recall, citation entailment, contradiction recall, source quality | ALCE-style attributed synthesis with exact spans; versioned papers; include contradictory/withdrawn evidence. |
| Research replication | Reproduction rubric, result closeness, artifact quality, cost | PaperBench-style separated agent/reproduction/judge containers; add non-ML and negative-results tasks. |
| Calibration | Brier, log score, reliability plots, ECE as secondary, selective risk–coverage | Correctness for clearly defined claims, decision/action outcomes, source-support judgments; report each separately and by shift. |
| Long-horizon | Verified evidence gained, claim traceability, missed contradictions, budget efficiency | 10–50-step investigation; perturb order, inject tool outages and conflicting sources; audit event log. |
| Writing | Blind expert preference, factual precision, citation completeness, compression | Pairwise rating; length controlled; judge disagreement reported. |
| Domain breadth | All above by math, CS, physics, quantum, astronomy, chemistry, biology, psych/cog sci, statistics, engineering, economics, philosophy/logic, methods and scholarly writing | Report per-domain intervals; no equal-mastery claim from pooled score. |

## Calibration details

Define event `Y` before asking a probability: e.g., “Does passage S entail claim C under the stated scope?”, “Will hidden tests pass?” or “Will experiment outcome fall in interval by date?”. A system forecast about future scientific truth needs horizon and resolution rule. Score resolved events with mean Brier and log loss; report calibration intercept/slope and stratified reliability. ECE is descriptive and bin-sensitive, not the sole optimization/evaluation target. For multiple choices use multiclass proper score. For claims with a truth spectrum, use ordinal/distribution targets or expert rubric, not arbitrary binary collapse.

For abstention, show risk vs coverage, area under risk-coverage curve, utility at preregistered review costs, and confidence intervals. Count abstentions on answerable tasks and confidently wrong answers separately. Verify calibration under domain shift, source removal, temporal shift and tool failure.

## Judge governance

LLM-as-judge is not ground truth. Create human adjudication subset with ≥2 independent domain reviewers and disagreement adjudication; validate judge against it, blind model identities, rotate rubric order and test verbosity/position bias. Report judge sensitivity/specificity and subgroup performance. For objective verifiers, independently audit grader false passes/fails and property-test edge cases.

## Ablations and gates

Baseline ladder: base model; SFT; retrieval+SFT; tool loop; DPO; RLVR; calibrated judge; research-state memory; process reward; integrated controller. Factorially remove each component. Test fixed token/GPU-hour and fixed tool-call budgets. A component graduates only if it improves predeclared primary trajectory metric without violating safety/factuality/calibration guardrails. Repeat at least 3 seeds for small runs and report all; larger runs with fewer seeds need paired tasks and explicit uncertainty.

Primary system endpoint: **independently verified supported research progress per unit cost**, decomposed into (a) correct resolved subquestions, (b) claim-level citation support, (c) falsified hypotheses correctly rejected, (d) reproducibility, and (e) unresolved question reporting. This is a proposed operational metric; validate that it reflects expert judgment before optimizing it.
