# Phase 3E — Fitting and free-generation diagnostics

Date: 2026-09-28. **Implemented and locally audited. Output structure improved; useful learned research remains unproven.**

## Built

- `language_diagnostics.py`: frozen training/validation probes, teacher-forced response scoring, strict greedy generation, error locations, reference divergence, output-contract checks and scoped semantic checks.
- `language_fit.py`: fixed multi-seed byte/BPE fitting chains, checkpoint-bound probes, actual final-budget Controller episodes, deterministic positive controls, artifact re-scoring/replay and portable export.
- A registered [study protocol](../docs/experiments/phase-3e-protocol.md), explicit [configuration](../configs/language-fitting.json), [operator guide](../docs/LANGUAGE_FITTING.md), and 13 diagnostic tests.

The implementation preserves the approved Researcher / separate Judge / deterministic Controller / independent verifiers / provenance-memory architecture. This phase changes no canonical evaluation set, default policy, parameter guard or training objective. SFT fitting is studied separately; no human-feedback or integrated-RL benefit is asserted.

## Method and interpretation

Eight arms cover two research tasks, byte/BPE32 tokenizers and seeds 17/29. Each begins from native random weights and is measured at step zero and after 32, 128 and 384 SFT updates. Model width is 32 with two layers, four query/two KV heads, FFN width 64 and context 256. Byte/BPE parameter counts are 26,880/27,904. Batch size is two; learning rate .001, weight decay .01 and clipping norm one.

Numerical fitting has eight train/four validation worlds, explicitly constructed without a test/OOD artifact. Symbolic fitting uses the existing 24 train/eight validation binomial rows. Each arm measures the first four rows of each split, selected before training. These exposed probes do not support a generalization claim. Task exposures differ, vocabulary sizes differ, and raw NLL across tokenizers is not a quality ranking.

The 128-token generation budget and the gold response's length are recorded independently. Teacher-forced accuracy/NLL conditions on correct prefixes; free generation feeds back the model's own choices. EOS probability at the gold position is not the probability that free generation will stop correctly. Reference-token disagreement can be a valid alternative and is not called a syntax error.

Numerical diagnostic PASS requires agreement with all prompt observations and the fixture target at tolerance 1e-8 using rational arithmetic. It does not check source provenance or mutate a research claim. The actual numerical Controller separately checks its existing target/provenance scope. Symbolic PASS is the existing bounded Q[x] identity check, not scientific discovery or a requirement that a particular expansion string be reproduced.

## Execution evidence

The full regression suite passed **221 tests, zero skips, in 459.576 seconds**. All eight fitting arms completed, with 32 checkpoint/baseline measurements, 256 individual probe measurements and 40 replayed actual Controller episodes. The tracked study took **798.968 seconds**, including regressions; export/review time is separate. These timings include integration and filesystem overhead and are not training-throughput benchmarks.

- Baseline: `227f9d0`; executed source: `a1748ea43d0c259c92d6d82988f68828c874c252`.
- Source hash: `81e052ac707dd35519be9a399603d945573fde81e0fa018023be187941f9e027`.
- Local run: `model-1/runs/20260928T103031-phase-3e-language-fitting-01036454`.
- [Export manifest](phase-3e-evidence/manifest.json), [complete arm/budget results](phase-3e-evidence/results.json), [test log](phase-3e-evidence/tests.log), [64-row metric table](phase-3e-metrics.csv).
- Environment: Python 3.12.14, PyTorch 2.8.0, NumPy 2.2.6, macOS 26.3.1 arm64, eight logical CPUs. Study training/measurement used one Torch compute thread; the regression suite also exercised its existing local distributed workers. No physical VIT host was accessed.

Two earlier targeted runs each passed 11 tests, before the final two additional test methods were added. [Their source hashes and logs](phase-3e-targeted-evidence/manifest.json) distinguish these revisions. No earlier test count is attributed to the final revision. The [standalone CLI check](phase-3e-cli-evidence/result.json) also exactly reproduced the registered symbolic/byte/seed-17 step-384 diagnostic with one compute thread requested through OMP/MKL environment variables. Its original run and log remain retained.

## Measured results

Each train/validation count below is **out of four probes**. Pairs mean **train / validation**. NLL is token-weighted response NLL, including EOS, rounded to six decimals; do not compare different vocabularies as a quality ranking.

| Task / tokenizer / seed | Train NLL | Validation NLL | Valid output contract, train / validation | Scoped checker PASS, train / validation |
|---|---:|---:|---:|---:|
| Numerical / byte / 17 | 0.112714 | 0.162152 | 4 / 3 | 1 / 0 |
| Numerical / byte / 29 | 0.082198 | 0.205481 | 4 / 3 | 3 / 0 |
| Numerical / BPE32 / 17 | 0.123567 | 0.264521 | 4 / 2 | 2 / 0 |
| Numerical / BPE32 / 29 | 0.097440 | 0.231110 | 4 / 4 | 2 / 0 |
| Symbolic / byte / 17 | 0.246008 | 0.343035 | 4 / 4 | 0 / 0 |
| Symbolic / byte / 29 | 0.254096 | 0.352116 | 4 / 4 | 0 / 0 |
| Symbolic / BPE32 / 17 | 0.229921 | 0.335125 | 4 / 4 | 0 / 0 |
| Symbolic / BPE32 / 29 | 0.253863 | 0.332987 | 4 / 4 | 0 / 0 |

At steps zero, 32 and 128, every probe failed the output contract. At step 384, all numerical training probes and all symbolic probes met the output contract; four numerical validation output instances remained syntactically invalid. All 64 final probe instances terminated with EOS. Every reference fit comfortably inside the 128-token request: reference lengths including EOS were 42–51 tokens. Thus an undersized generation budget does not explain the recorded final failures.

Teacher-forced validation token accuracy at step 384 was approximately **93.60–95.11%** for numerical outputs and **91.33–92.86%** for symbolic outputs, while every arm's validation checker pass count was zero. Most predicted tokens can agree with the gold-prefix format while answer content is wrong. This is an observation about these probes, not evidence of general reasoning or a causal explanation of training behavior.

Numerical cumulative SFT response-target counts were 35,232 per byte arm and 32,928 per BPE arm. Symbolic counts were 37,664/36,896. Each arm completed 384 updates, with 768 row exposures: 96 cycles over eight numerical training rows or 32 cycles over 24 symbolic rows. These unequal task exposures and different tokenizations prevent interpreting this as a matched task/tokenizer superiority experiment. All budgets, including unsuccessful earlier ones, remain exported.

### Actual Controllers

The final-budget Controllers used the first two train and first two validation probes per arm, rather than all eight diagnostic probes. All **eight deterministic reference episodes** produced verified claims. Of **32 neural episodes**, exactly **two training episodes** produced verified claims, both numerical byte/seed-29; the other 30 produced none. No neural validation episode verified a claim. Thirty neural episodes produced a hypothesis; two numerical validation episodes failed JSON parsing and retained an unresolved error. No reference fallback substituted for a learned output.

The audit checked every saved final diagnostic generation against the corresponding actual Controller trace and replayed all 40 state ledgers. The differing numerical diagnostic/Controller verification scopes and differing probe subsets are explicit; do not substitute one count for the other.

The same few worlds are repeated across seeds/tokenizers/budgets. These are correlated measurements, not 256 independent research tasks. No statistical significance, model promotion or scientific-discovery claim follows.

## Failure handling and scope

Negative controls cover changed probe labels with recomputed hashes, altered output/metrics/artifacts, removed planned controls, wrong dataset bindings, malformed or duplicate/nonfinite JSON, valid-but-wrong expressions, alternative correct expressions, EOS/special-token/UTF-8 behavior, context rejection and nonfinite model failure. A correct hash declaration alone does not authorize a probe: its rows must match the checkpoint's saved training/validation data.

The exporter reconstructs initial weights, checks final diagnostic generation against actual Controller traces, decodes saved token IDs, re-scores candidates, recomputes aggregate metrics and replays ledgers. Saved neural likelihoods receive consistency checks but no second independent neural implementation. Hashes are integrity records, not authenticated signatures. Full checkpoints, sources and ledgers remain local; selected authored evidence is exported.

There were **no unexpected failed top-level training/study runs in this phase**. Unit-test rejection cases were expected negative controls. The poor learned outputs remain in the completed experiment rather than being deleted, rerun with selectively chosen settings or presented as an implementation pass for model capability. Earlier Phase 3D failed audits and interruption runs are unchanged.

## Remaining decisions

No learned backend, tokenizer or larger scale is promoted by fitting. Physical VIT nodes, representative licensed corpus quality, sustained throughput, long-context literature retrieval, broader tools and integrated research-loop learning remain separate work. A fresh family-disjoint capability protocol is required before asserting generalization.

**Recommended next ML experiment:** add answer-field accuracy and paired evidence-change probes under a new fixed protocol. Determine whether valid JSON contains prompt-sensitive numerical content, and separate template fitting from correct transformations. This study cannot establish the cause of incorrect answers. Reweighting losses, constrained decoding, extra training or increased parameters remain experiments, not approved fixes. Exclude every previously exposed world from any future fresh capability holdout.

**Next engineering build:** extend the existing provenance memory with local document/chunk ingestion, cross-run retrieval and source-update/retraction handling. Preserve exact source spans and keep source support distinct from factual verification. This addresses an existing research-system gap without requiring a capable new neural checkpoint or changing the Researcher/Judge split. A bounded local implementation can precede external literature connectors.

## Reproduction and changed files

Follow the [operator guide](../docs/LANGUAGE_FITTING.md). Run the study into a new directory; do not overwrite retained results. The [exact file inventory](phase-3e-files.json) lists implementation, tests, configuration, protocol, evidence and updated handoff against `227f9d0`, excluding the inventory itself. The CSV is a direct flattening of all result summaries; its source results SHA-256 is `00d02f7a3e3915dc189e4002dc58c4de8ccc073f4d82c62f6963619314074636`.

Implementation commits: `4cd008c` (diagnostics/study runner) and `a1748ea` (registered configuration/protocol, operator guide and tests). Both were pushed before the complete study executed. Canonical numerical/symbolic evaluation files and their sidecars were unchanged.
