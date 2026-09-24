# Research-loop design

## Architecture choice

Use a hybrid deterministic shell + learned researcher + explicit decision model + verifier tools. Rules enforce permissions, provenance, schema and stop conditions; learned policies choose what to read, compare, calculate, test or ask next. An entirely rule-based workflow is brittle; an unconstrained agent can silently lose state and provenance. Compare the proposed controller with (1) one LLM acting from transcript, (2) LLM with structured state but no learned router, and (3) separate planner/judge. Evaluate at equal token/tool budget.

## Typed research state

```text
ResearchState = {
  question: Question {text, scope, time_cutoff, requested_artifact},
  assumptions: [Assumption {id, text, status, source}],
  known_facts: [ClaimRef], unknowns: [QuestionRef],
  sources: [Source {id, url_or_doi, version, retrieved_at, license, hash}],
  evidence: [EvidenceSpan {source_id, locator, text_hash, parser, relevance}],
  hypotheses: [Hypothesis {id, statement, predictions, alternatives}],
  experiments: [Experiment {id, protocol, preregistration, code_hash, seed}],
  results: [Result {experiment_id, raw_artifact, status, uncertainty}],
  contradictions: [Relation {claim_a, claim_b, evidence_ids, resolution}],
  confidence: [Forecast {claim_id, target, probability, reference_class, horizon}],
  open_tasks: [Task {action, rationale, expected_information_gain, cost}],
  provenance: [DerivationEdge], final_claims: [Claim {text, epistemic_state, evidence_ids}]
}
```

State is append-only event history with materialized current view. Every tool result carries immutable event ID, time, environment/version and source references. Edits are new events with supersedes/retracts links. This supports “why believe it?”, contradictions and belief revision. Do not use a vector index as sole memory.

## Action protocol

Actions: SEARCH, READ, EXTRACT, COMPARE, CALCULATE, CODE, RUN, SIMULATE, PROVE, CHALLENGE, HYPOTHESIZE, VERIFY, ABSTAIN, REVISE, WRITE. Each has JSON schema, capability scope, cost limit, timeout, expected observation schema, and idempotency policy. The system must distinguish failure, empty result, timeout, unsupported and verified false. Execute code in a disposable sandbox; Internet access only through explicit retrieval tools; no hidden filesystem/secret access.

## Agent cycle

1. Normalize question, scope, target and deadline; surface ambiguities as explicit assumptions.
2. Build claim graph and research plan with competing explanations.
3. Search primary/credible sources; capture exact version and spans, not just snippets.
4. Extract claims/method/data/limits separately; cross-check source identity and dates.
5. Generate at least two plausible hypotheses where the task calls for discovery; derive distinguishing predictions.
6. Choose next action by expected value of information (EVI), risk and cost; controller abstains/asks for human input when environment lacks needed authority or evidence.
7. Run independent checks/experiments; record raw outputs.
8. Update claims and forecasts; preserve negative results and unresolved conflicts.
9. Draft only from claim graph; automated citation coverage check; human review gates for high-impact assertions.
10. Stop when preregistered evidence/utility criteria met, budget exhausted, or marginal EVI below cost; report what remains unknown.

## Memory architecture

Use relational tables for source/version/claim/evidence/experiment/provenance and constraints, object storage for PDFs/code/output artifacts, graph edges for support/refute/derived-from/cites/supersedes, and a vector index for candidate retrieval only. Add lexical BM25 and citation/metadata search; rerank with source quality and date filters. Store embeddings with model/version. Retrieval returns passages with locators and rights boundaries. Version source changes; never silently overwrite. Use user/project workspaces with access controls and retention/deletion policy.

## Training loop as sequential decision process

State `s_t` includes bounded evidence graph summary plus durable database IDs; action `a_t` is a typed research operation; observation `o_{t+1}` is retrieved/verified result; terminal report `y`. Reward components include verified task progress, supported claim coverage, experiment reproducibility, resolved uncertainty, preference quality, and resource cost. Credit assignment should use terminal outcome and intermediate *observable progress* (e.g., newly verified relevant evidence), not verbose intermediate rationale. Train trajectories offline from reproducible tasks, then cautious online policy optimization in sandboxed environments. Randomize tool failures/data order and hold out entire environment families.

## Novelty gate

“Research-Loop Reinforcement Learning” is not established as novel by naming it. Define contribution as a measurable policy-learning method over evidence-state actions with clean environment reward and superior independent research outcomes at fixed compute/cost versus ReAct/tool-using SFT, retrieval+same model, and a fixed planner. Require ablation for learned decision policy, provenance state, verifier ensemble and reward components. Preregister before test evaluation.
