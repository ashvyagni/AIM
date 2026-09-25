# Model-1 implementation phases and next experiments

## Completed foundation

See the [phase report](../reports/phase-1-implementation.md) for actual observations. The current release establishes all sixteen requested areas at miniature engineering scope; it does not make all sixteen production-grade systems.

| Requested area | Implemented now | Remaining work |
|---|---|---|
| Researcher interface | Reference and native-transformer adapters; strict hypothesis schema | Train useful structured/free-form research behavior |
| Decision/Judge interface | Separate rule and trained feature Judge | Language evidence inputs; robust calibration under shift |
| ResearchState | Versioned explicit state and snapshots | Rich scientific claim types and schema migrations |
| Deterministic Controller | Budget, phase transitions, source/check gates | Multi-step revision/replanning and recovery policies |
| Tool/action contracts | Allowlisted calculation/measurement workers and deadlines | Sandboxed code, approved network retrieval and resource accounting |
| Claim/provenance | Exact spans, content hashes, scoped checks, graph edges | Entailment, source credibility and retraction models |
| Evidence/research memory | Local SQLite and immutable objects | Cross-run retrieval, literature parsers and distributed storage |
| Verifier framework | Numeric and provenance implementations | Symbolic/theorem/code/scientific verifier adapters |
| Evaluation harness | Tests, immutable public suite, training diagnostics | Sealed research tasks, family holdouts, humans and uncertainty intervals |
| Training harness | Native random-init decoder; separate stage checkpoints | Large-corpus pretraining, streaming, distributed optimizer state |
| RLHF/preference | DPO plus provenance-aware external data intake | Real human annotation study; optional reward-model/PPO comparison |
| RLVR | Sampled bounded REINFORCE and exact arithmetic verifier | Open-ended generation, process rewards and credit assignment |
| RLCD/calibration | Proper scores, separate Judge, validation temperature | Sequential decision-learning formulation; OOD robustness |
| Experiment tracking | Unique runs, metrics, configs, lineage, failures | Shared artifact service, access controls and multi-worker coordination |
| Reproducibility | Pinned local environment, source archives, test/experiment runner | Independent machine reproduction and platform-specific locks |
| Hardware/cluster | Local CPU and Gloo/DDP infrastructure | Physical VIT-host audit, sustained scaling, failure recovery |

## Phase 2A — Learned structured Researcher

**Completed first experiment; capability gate failed.** See the [Phase 2A report](../reports/phase-2a-implementation.md) and [registered protocol](experiments/phase-2a-protocol.md). Three models learned output structure, but only 2–7 of 64 familiar test-world predictions per seed passed verification. Keep the default deterministic Researcher. The following requirements continue to apply to successor experiments.

Generate training examples from disjoint procedural worlds with observation/evidence IDs and target hypothesis JSON. Preserve metadata and generator version. Train the native decoder from random initialization, starting with an explicitly bounded model/config within the harness guard. Keep claims scoped to predictions, not discovery of a unique law from finite observations.

Predeclare train/validation/test/OOD world-family partitions. Evaluate syntax validity, fabricated reference rate, numerical candidate quality, and end-to-end independently verified success. Compare the deterministic Researcher, untrained native decoder and trained native decoder under equal action budgets. Keep malformed output as a recorded failure; no automatic reference-backend substitution.

Exit gate: produce valid supported candidates on a predeclared held-out set and complete the loop through independent verification. Select numerical acceptance thresholds before evaluating the new holdout; current inspected fixtures cannot supply that evidence. Successful JSON generation alone does not establish scientific reasoning.

### Phase 2A.1 — Numerical and process supervision (completed)

Research-trainer continuation now matches uninterrupted training, including an audit of the actual Phase 2A checkpoint. The preregistered plain/worked comparison completed 77 tests and 1,664 Controller cases. Mean verified test success rose from 5.21% to 12.50%, but one seed regressed and neither arm passed its numerical gate. Keep the deterministic default. Eight worked-arm familiar-case successes contained wrong process steps. Read the [report](../reports/phase-2a1-implementation.md), including the invalidated first attempt and corrected data boundary.

### Phase 2A.2 — Arithmetic curriculum (completed; promotion gate failed)

A [registered same-scale comparison](experiments/phase-2a2-protocol.md) implements worked SFT with and without staged signed arithmetic, second differences and coefficient reconstruction. Both recipes match dense training shapes and candidate-world draws; task exposure differs explicitly. New worlds exclude both previous experiments. Fixed train/validation fitting diagnostics do not select models. See the [completed report](../reports/phase-2a2-implementation.md) and [reproduction guide](ARITHMETIC_CURRICULUM.md). All 92 preflight tests and 1,664 Controller cases completed. Worked/curriculum mean success was 6.77%/6.25%; both arms and the curriculum advantage gate failed. Preserve the default and move to the separate Judge phase.

A separately scoped observation-consistency verifier and read-only memory audit found that 17 of 25 learned familiar target passes contradicted earlier observations. They preserve existing target-prediction statuses and report an additional observation diagnostic. Earlier auxiliary output validity largely collapsed after task replacement; replay/order ablations remain future hypotheses. Do not reward unchecked model-generated steps, compute answers behind the neural interface, or interpret these tiny synthetic results as sufficient evidence for a larger parameter scale.

## Phase 2B — Judge shift and decision experiments

**Completed first comparison; all-seed evidence gate failed.** Twelve random-init Judges, 102 passing tests, 768 held-out candidate records, independent audit of 1,536 labels and grouped intervals are recorded in the [Phase 2B report](../reports/phase-2b-implementation.md). Richer observed features improved familiar Brier but did not consistently improve shifted decision utility. Keep the verification-first default. The [guide](JUDGE_SHIFT.md) documents reproduction and the opt-in Controller adapter.

The experiment defines a future measurement-pass event, excludes outcomes from features, uses fresh grouped worlds, and separately compares log/Brier training, temperature scaling, constant/base-rate forecasts, observed-residual features and abstention. Cubic worlds are now explicitly training data; quartic perturbations were held out along with a changed mixture. Both shifts act together. All resulting holdouts are now exposed historical diagnostics.

### Phase 2B.1 — Shared costs and evidence acquisition (next)

Specify a question-level utility with shared measurement cost and explicit cost for an additional observation. Avoid double-crediting equal candidate predictions. Compare verify-all, abstain, observed-fit and calibrated policies with the Researcher fixed, before attempting learned sequential decisions. Freeze the cost model and grouped family/prior partitions before new evaluation; vary family and mixture separately where feasible. A new observation's benefit must include its acquisition cost. Existing Phase 2B per-candidate utility is an offline diagnostic, not measured shared-tool savings.

Retain proper scores, risk/coverage, cost, lost opportunities and whole-group intervals. No confidence forecast may bypass a verifier or imply global-law proof. Calibration transfer to learned Researcher candidates remains a separate required experiment. The [observation-scope note](OBSERVATION_SCOPE.md) remains a mathematical constraint on what the Judge can infer.

## Phase 2C — Trusted evidence/tool expansion

Add one new domain at a time: exact symbolic arithmetic, code tests in a real resource-limited sandbox, or a licensed local scientific-paper corpus with source-span retrieval. Each adapter needs a schema, version, timeout/failure semantics, independent checker and adversarial cases. Merely having multiple learned Judges agree is not independent verification.

Extend memory with persistent schema migrations, provenance-preserving chunking, deduplication and source validity/retraction records before adding broad retrieval. Define natural-language claim entailment separately from exact quote integrity. No tool may execute arbitrary source instructions.

## Phase 3 — Corpus/tokenizer and 100M–300M proxy

Complete the physical hardware audit, corpus licensing/PII/quality review, document/task split policy and deduplication. Train/evaluate tokenizer candidates on held-out prose, code, math and Unicode. Add a streaming next-token pretraining objective, resumeable data cursor, checkpoint recovery and memory estimates. Decide precision from actual kernel/hardware measurements.

Run short matched dense proxy studies with explicit token budgets, loss curves, downstream verifier tasks, validation contamination checks and sustained training throughput. Compare meaningful data mixtures and scale/compute trade-offs. The current micro-model arithmetic experiment cannot select a large-corpus data mixture or training token count.

## Phase 4 — Conditional ~1B systems experiment

Only after the proxy and physical-node benchmarks: define an approved scale run with active/total parameter count, token budget, storage, recovery plan, node schedule and stop criteria. Evaluate physical-node weak/strong scaling before committing the full fleet. Test failure recovery and checkpoint correctness. Keep technical scaling success distinct from research-model quality.

## Phase 5 — Conditional 7B+ and integrated research-loop training

Justify the compute/corpus/human-evaluation budget. Compare separated training baselines at matched budgets before proposing multi-objective or constrained joint training. For an integrated decision process, formally define actions, episode termination, verification and human-feedback timing, confidence targets, costs and missing outcomes.

Possible future formulations include alternating generator/Judge updates, constrained expected verified utility, lexicographic safety/correctness constraints, and process rewards from independently checked steps. These are hypotheses, not approved implementation changes. MoE/shared-head/14B/32B alternatives require new architecture records and evidence.

## Open questions with owners/measurements needed

| Question | Required evidence |
|---|---|
| Can the fleet train a useful dense proxy in available lab hours? | Sustained tokens/sec, memory, network, host availability and corpus budget |
| What Judge scale/input format is useful? | Calibrated event prediction and decision utility versus inference/training cost |
| Does preference learning improve verifier-grounded reasoning here? | Matched-budget ablations across seeds, not one tiny staged run |
| Can calibration survive new scientific domains? | Family/time holdouts and shift-aware evaluation with resolved outcomes |
| What should RLCD mean beyond this prototype? | Explicit sequential decision objective, actions, feedback and estimators |
| How can process rewards avoid verifier exploitation? | Independently constructed tests, counterexamples and adversarial traces |
| Which tokenizer/context strategy is feasible? | Corpus coverage, tokens per document, quality and actual memory/latency |
| What is AIM's defensible novelty? | Reproducible measured advantage over clear baselines and primary prior-art review |

## Reporting and Git discipline

At each meaningful phase, update a dated report with files changed, commands, actual tests, benchmark scope, failed runs, uncertainty and next experiment. Commit code, configs and reviewed documentation with descriptive names and push to the owner's `main` branch. Retain local run artifacts separately; publish selected portable metrics/logs with their provenance. Do not force-push, invent positive findings, silently change canonical evaluations, or promote a hypothesis to an approved architecture.
