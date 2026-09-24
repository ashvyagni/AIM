# Model-1 evaluation specification

## Evaluation layers

1. **Contract and numerical tests:** serialization, provenance integrity, action identity, timeouts, confidence/status separation, data partitions, masking, causal attention, objective gradients, frozen references and exact CPU checkpoint resume.
2. **Public engineering regression suite:** eight versioned miniature investigations with explicit expected claim counts and unresolved conditions.
3. **Training diagnostics:** held-out procedural response NLL, preference ordering, finite-candidate expected reward and separate Judge proper scores.
4. **Hardware diagnostics:** local forward/backward/optimizer timing, prefill, tokenization, checkpoint persistence and local Gloo/DDP correctness.
5. **Scientific research capability:** not evaluated by this phase. No domain benchmark, human research assessment or discovery claim is supported by the above layers.

## Canonical engineering fixtures

`eval/suite-v1.json` is paired with SHA-256 `5a0d80fd18c74efc1e91655af3be617cda947bfa81c43ac9b03260b598272b02`. Loading a changed JSON file without a matching versioned review fails. The sidecar is a consistency check, not a digital signature or secret test-set seal. Preserve v1; change the suite by adding a reviewed v2 with a new hash and decision note.

| Fixture | Expected behavior |
|---|---|
| Linear world | Both candidate representations agree at the target; two passed claims are not two independent discoveries |
| Quadratic world | Reject linear extrapolation; verify quadratic prediction at target only |
| Cubic world | Reject both candidate predictions |
| Unavailable measurement | UNKNOWN; no numeric success inferred |
| Conflicting sources | Unresolved contradiction; no supported final prediction |
| Empty retrieval | Unresolved; no invented evidence |
| Instruction-like source text | Treat prose as source data; never execute it or let it change Controller status rules |
| Exhausted action budget | Stop within budget; existing unchecked claim stays UNKNOWN |

The learned Judge is evaluated with threshold 0 so both rule and learned backends request verification of all candidates. Identical suite pass counts demonstrate preserved Controller behavior; they do not establish superior decision efficiency or calibration.

## Metrics

- **Response token NLL:** teacher-forced held-out response negative log probability, including EOS. It is not free-form accuracy.
- **Preference accuracy:** proportion with chosen response sequence score above rejected. Interpret label origin and response-length bias.
- **Bounded expected reward:** probability mass assigned to the correct answer in an environment-supplied set of three answers. Report the privileged candidate construction.
- **Brier and log loss:** evaluate only resolved forecasts. Unknown/error/timeout outcomes are missing, never automatic zero labels.
- **Forecast classification accuracy:** threshold 0.5 on a defined event; prevalence can make this misleading, so use proper scores too.
- **Selective coverage/risk:** at thresholds 0, 0.5, 0.8, 0.95, report fraction retained and fraction of retained predictions that fail. Empty retained sets have risk null, not zero.
- **Runtime reliability:** final status, action count, unsupported citation rejection, ledger replay and source/check integrity.
- **Performance:** measured samples and median/p95, explicit batch/context/precision/thread count, wall time and peak process RSS. Five-sample p95 is effectively the observed maximum; it is not a stable tail estimate.

This phase does not implement broad citation entailment metrics, reliability diagrams/ECE, statistical confidence intervals, calibration under selective missing feedback, or paired human preference studies. Those remain next-phase work. Do not replace proper scores with an ECE-only headline later; binning choices must be recorded.

## Data independence and current limitations

Split by operand pair for arithmetic and complete coefficient world for Judge data. Cubic worlds form a distinct OOD diagnostic. Hypothesis features are generated before target labels. The same generator family underlies all miniature tasks, so successful in-family generalization is deliberately weak evidence.

The eight regression cases are public and hand-authored; they are not a concealed holdout. Models are not trained on them, but developers inspect them. For future scientific evaluation, reserve document/task families, source versions and temporal cutoffs; maintain access controls; run near-duplicate checks across data stages; and require owner review before changing canonical cases.

## Release gate for this foundation

- Full test suite passes with neural dependencies installed and no skipped neural tests.
- All eight reference-loop regressions pass; learned-Judge integration preserves verifier gating.
- SFT, preference and RLVR produce separate checkpoints with nonzero updates and correct lineage; Judge is separate.
- Recorded checkpoint resume equals uninterrupted CPU training under the same environment.
- Every actual failed experiment remains available with its cause; diagnostics reporting poor capability are not reclassified as successes.
- Source snapshot, environment/config/data hashes, test log and exact result paths exist.

The local distributed check is a separate infrastructure result, not a prerequisite for claiming the standard-library loop works. Physical VIT-host scaling is not certified by a loopback check.

## Next scientific evaluation gate

Before a large training commitment: repeat across at least three predeclared seeds, report whole-task confidence intervals, compare matched training and verifier budgets, include constant/base-rate Judge baselines, test unseen task families and adversarial verifier failures, and measure actual structured Researcher output validity and verified task success. Specify thresholds before looking at new holdouts. A scientific improvement requires stronger evidence than “loss decreased” or “all engineering tests passed.”
