# Reward and verifier architecture

## Principle

No evaluator is a universal truth oracle. Treat each verifier as an instrument with a declared input contract, target, error profile, version, and audit set. The final system should retain raw outputs and never convert “judge says supported” into “verified fact” without the declared evidence check.

## Stack

1. **Schema/contract checker:** JSON schema, tool arguments, required claim IDs, references resolving to stored evidence. Deterministic and fail-closed.
2. **Domain verifiers:** code build/tests and complexity checks; symbolic algebra with assumptions; theorem proof in Lean/Isabelle where formalization exists; numerical simulation with fixed seed/tolerances; statistics recomputed from data; dimensional/unit consistency. Each verifier reports pass/fail/unknown plus logs.
3. **Source checker:** resolve bibliographic identity, archive source/version/hash, map claim to exact page/section/evidence span, test entailment and counterevidence. Entailment models can triage, not certify.
4. **Independent judge models:** compare claim-evidence pairs for support, contradiction, missing qualification; separate from researcher weights/data when possible. Calibrate on human-labeled hard negatives.
5. **Human expert panel:** adjudicate high-impact claims, ambiguous proofs, novelty and research quality; record disagreement rather than force consensus.
6. **Calibrated controller:** on explicit candidate action set, emit action probabilities and abstention/review trigger. Output contract prevents invalid types; correctness depends on calibration and coverage.

## Reward signals

Maintain decomposed event ledger per trajectory: task success, independently verified claim fraction, citation entailment, experiment reproducibility, hypothesis discrimination, calibration proper score after resolution, human preference, tool cost, wall time, and policy violations. Use hard gates for invalid/malicious tool behavior and missing provenance. Use scalar reward only within an isolated, well-defined environment with reward scale/variance recorded. Use constrained/Pareto comparisons at system level.

Outcome reward is preferred for objective tasks. Process score may improve search or identify a bad step in a proof, but use as a proposal prior or bounded shaping term. A candidate step reward should be terminally checked and outcome-conditioned: `r_t = f(step_t, context, eventual verified outcome)`, trained on correct and adversarially plausible traces. Do not pay per token or reward “reasoning-looking” text. Ensure any shaping preserves the desired optimal policy (e.g., potential-based `F(s,s')=γΦ(s')-Φ(s)` under its assumptions); arbitrary shaping can change it.

## Anti-reward-hacking controls

- Isolate verifier inputs from model-writable state; tool results are signed/immutable and provenance-bound.
- Use sandboxed execution, no secret leakage, CPU/memory/time quotas, network disabled unless explicitly required, and fresh containers for reproduction.
- Keep hidden tests and test generators outside training access. Rotate held-out seeds and challenge sets.
- Use differential verification (two independent implementations, symbolic + numeric, reference + metamorphic properties), especially when a single parser can be exploited.
- Evaluate false-accept and false-reject rates separately with confidence intervals. Human-audit a random sample and all high-impact cases.
- Freeze reward versions per run; log input, version/hash, output and policy decision. Canary old/new judges to detect drift.
- Check reward against task outcomes under optimization pressure, not only IID classifier accuracy. Track proxy score vs independent score as policy moves away from reference.
- Penalize unnecessary tool calls and verbosity only after useful work is measured; otherwise incentives encourage premature stopping.
- Red-team answer leakage, test memorization, formatting tricks, citation laundering, and code that exploits harness state.

## Verifier contracts and output states

`VERIFIED` means a specific declared check passed for a specific claim and input (not “scientifically true forever”). `STRONGLY_SUPPORTED` means multiple relevant independent sources with adequate spans and no detected contradiction; this remains an evidence judgment. `PARTIALLY_SUPPORTED` indicates a narrower statement is entailed. `UNCERTAIN`, `UNVERIFIED`, `CONTRADICTED`, `HYPOTHESIS`, and `UNKNOWN` preserve epistemic distinctions. Every state includes method, timestamp, scope, and provenance. Confidence and verification state are separate fields.

For numerical/code facts store artifact hashes, environment/container, dependencies, command, seed, tolerances and stdout/stderr. For citations store DOI/URL, version date, retrieval timestamp, source hash, page/line/span offsets and quote hash. Keep extracted text alongside offsets and parser version.
