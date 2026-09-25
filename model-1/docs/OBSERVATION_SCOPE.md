# What finite observation checks can establish

## A concrete ambiguity in Model-1

Let `q(x)` be any quadratic candidate consistent with observations at x=0,1,2. Define another possible law

`f_k(x) = q(x) + k*x*(x-1)*(x-2)`.

For every k, `f_k(0)=q(0)`, `f_k(1)=q(1)`, and `f_k(2)=q(2)`. At any requested target t in {3,4,5}, however,

`f_k(t)-q(t) = k*t*(t-1)*(t-2)`.

For nonzero k this is nonzero. The two laws can produce identical observation evidence yet disagree on the next measurement. This is a mathematical construction, not a measured model result. It explains why no algorithm observing only those three points can uniquely identify an unrestricted underlying law without additional assumptions or evidence.

A unique quadratic interpolation requires the assumption that the relevant law has degree at most two. Good fit alone does not establish that assumption. A fourth measurement can distinguish the particular candidates above; it still does not prove uniqueness among unrestricted functions.

## Four different questions

| Question | Current mechanism | What a pass establishes |
|---|---|---|
| Did the model cite an intact source span? | Provenance verifier | Exact source/version/span integrity |
| Does this polynomial match its cited points? | Observation-consistency verifier | Numerical agreement at those finite observed points |
| Are the generated arithmetic steps correct? | Worked-output diagnostics | Correctness of the specified finite-difference calculations |
| Does the predicted target value match a new measurement? | Measurement verifier | Agreement at that target and tolerance |

These mechanisms have distinct scopes. In particular, a single target match does not imply observation consistency or a correct process trace. Nor does an observation fit imply agreement at a future target. The [counterexample tests](../tests/test_observation_verifier.py) execute both cases through the actual Controller.

The new `ObservationCheck` is bound to a hypothesis and its evidence. It is deliberately separate from the existing target `Verification`. The optional audit does not relabel historical results or change the canonical target-success metric.

## Consequences for the separate Judge

**Established mathematical constraint:** if two possible worlds produce identical Judge inputs but disagree about the future event, those inputs cannot deterministically reveal which world is present. A confidence forecast therefore needs a defined event and an assumed or learned distribution over possible worlds. Calibration on one mixture does not by itself establish calibration on another mixture.

**Engineering recommendation:** preserve observations, hypotheses, process checks, future measurements and their uncertainty as separate state fields. Confidence should govern whether to obtain more evidence or abstain under an explicit utility/cost policy. It must not override a failed verifier or convert finite fit into a global-law claim.

**Unresolved research questions:** which features identify useful distribution shifts; which priors or task restrictions are defensible; how to choose additional measurements; and whether learned decisions outperform constant forecasts and deterministic policies at matched costs. These require separate Judge experiments, not assumptions hidden inside a new score.

Multiple models agreeing on the same hypothesis does not remove the observational ambiguity. An ensemble may improve estimation under a supported distribution, but consensus alone is not independent evidence about an unobserved target. AIM's independent tools and scoped verifiers remain necessary components of the approved design.

## Current implementation boundary

The polynomial domain is synthetic and tightly bounded. Observation checks use exact rational arithmetic with tolerance 1e-8 after validating source spans. A source's intact bytes do not establish real-world credibility. Neither this checker nor the target verifier evaluates natural-language scientific entailment, experimental methodology, causal identification or theorem proof.

Future domains need their own explicit claim types, assumptions, trusted measurements and failure outcomes. Reusing the word VERIFIED without carrying the check scope forward would discard information the current implementation preserves.
