# Phase 3E — Fitting and generation diagnostics

Registered 2026-09-27 before running the study. Baseline: `227f9d0`. This is a bounded engineering diagnostic, not a capability promotion study. The approved architecture and all canonical evaluations remain unchanged.

## Questions and fixed plan

Can the versioned SFT path fit small authored examples, emit its output contract and terminate, and where do likelihood, output validity and scoped correctness disagree? Phase 3D's two-update adapters emitted no valid hypotheses. Longer fitting may improve syntax while leaving numerical/generalization errors; this is a hypothesis, not an expected success claim.

Run all combinations of tasks structured numerical / symbolic, tokenizers byte / existing corpus-fitted BPE32, and seeds 17 / 29. Use native random initialization, width 32, two layers, four query/two KV heads, FFN width 64, context 256; no pretraining or downloaded weights. Observe step zero, then 32, 128 and 384 SFT updates via exact stage continuation. AdamW learning rate .001, weight decay .01, batch two, clipping norm one; checkpoint every 32 steps. All arms and all budgets remain in the report. No early stopping or best-checkpoint selection.

Numerical fitting uses eight explicitly listed train worlds and four validation worlds in `language_fit.structured_fixture`. It constructs no test/OOD file. Symbolic training uses the existing 24 train/eight validation binomial rows. This task exposure difference is deliberate and must be reported; it is not a matched task-difficulty comparison. Probe the first four rows of each split in prepared order. Freeze their complete manifests/hashes before training. These examples are exposed fitting probes, not unseen capability evidence.

BPE uses 32 merges fitted only on the existing 20-document corpus's training partition, as in Phase 3D. It is not fitted on probe validation responses. Full specs and fitting provenance are saved. Equal dimensions imply different vocabulary parameter counts. Token counts/bytes and objective exposure are recorded; this is not a compute-matched tokenizer contest.

## Measurement definitions

- Teacher-forced response-token NLL includes EOS, excludes prompt/padding. Aggregate by response tokens. Record all reference IDs and greedy IDs conditioned on gold prefixes, token accuracy and gold-position EOS probability.
- Greedy free generation uses exactly the existing strict decoder, 128 new tokens, with no repair, constrained decoding, source-label access or deterministic fallback.
- Record prompt/reference lengths, whether the generation budget can contain the reference plus EOS, stop reason, raw generation and first divergence from reference token IDs. A valid alternative can diverge: this is not an invalid-token assertion.
- Report JSON syntax validity separately from domain contract validity. JSON parser offsets are zero-based character/UTF-8 byte offsets when available; duplicate-key/schema errors may have no position. Record emitted invalid BOS/PAD positions separately.
- Numerical scoring parses the existing hypothesis contract, evaluates all prompt observations and the fixture target with rational arithmetic at tolerance 1e-8, and reports both components. A diagnostic PASS requires both; it is not a production claim or global-law proof. The target outcome is available only to the post-generation scorer, never the model.
- Symbolic scoring invokes the existing bounded Q[x] checker. UNKNOWN remains distinct from FAIL; neither is PASS. Contract failures do not run a semantic check.
- At the final fixed budget, run actual Controllers on the first two train and first two validation probes per arm, plus deterministic positive controls for those cases. Replay ledgers and preserve failed generation. The Controller's target verification and the numerical diagnostic's combined observation/target check have different scopes and must not be conflated.

All metrics are descriptive counts/curves per split, seed and tokenizer. Do not infer significance from eight probes or pool dependent checkpoints into an enlarged sample. No arbitrary accuracy gate is used to promote models; all learned backends remain experimental.

## Reproducibility, rejection and acceptance

Configuration: `configs/language-fitting.json`. Execute from `model-1`:

```sh
.venv/bin/python -m aim.language_fit --regression --export reports/<new-directory>
```

Require no skipped regression tests, complete planned arms/budgets, unchanged model/RNG during diagnostics, dataset/tokenizer/checkpoint identity bindings, independent candidate re-scoring, and actual Controller ledger replay. Unexpected errors fail the run and retain logs. Test negative controls include altered probes, altered artifacts, malformed JSON, valid-but-wrong answers, correct alternative forms, insufficient budgets and wrong dataset bindings.

The audit reconstructs deterministic random initialization and checks saved model digests; it replays candidate scoring and Controller ledgers. It does not independently recompute the neural likelihoods or prove optimizer mathematics. Local timings do not establish VIT throughput. Full checkpoints/source snapshots remain local; selected probe/generation/metric evidence is exported with hashes.

After this diagnostic, choose a further experiment from observed failure categories. Fresh world-family test/OOD partitions and predeclared capability gates are required before any generalization claim. Do not retune on old holdouts, change canonical suites or increase scale solely because syntax fits.
