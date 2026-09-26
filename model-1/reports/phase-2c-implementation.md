# Phase 2C — Symbolic components and training build

Date: 2026-09-26. Status: **implemented and build-validated; learned capability remains inadequate**.

## What was built

- A separately versioned symbolic ResearchState, hypotheses, claim identities, Researcher/Judge interfaces and deterministic Controller.
- Exact rational polynomial verification over Q[x], with a restricted AST grammar, intermediate degree/coefficient limits, operation limits, subprocess deadline and explicit PASS/FAIL/UNKNOWN semantics.
- An independent binomial-formula reference proposer; a native transformer adapter with strict output validation and no hidden fallback.
- Pre-dispatch Judge decisions, immutable evidence spans, request-bound certificates, separate source-integrity checks, scoped final responses and read-only ledger/claim audits.
- Symbolic SFT, programmatic preference optimization, finite-candidate RLVR and a separately trained/calibrated Judge. Checkpoints have distinct symbolic contracts, parent hashes and exact continuation support.
- CLI commands, four runnable training configs, a checksummed 18-case evaluation suite, 22 additional tests, a complete reproduction entry point and portable evidence export.

The [guide](../docs/SYMBOLIC.md) specifies contracts, accepted grammar, limits, objectives, commands and limitations. This implements the existing approved Researcher + separate Judge + Controller + Verifiers + Memory architecture. It does not introduce a joint reward or choose a larger final model.

## What actually works

The deterministic reference completed planning, source retrieval, proposal generation, separate judgement, actual subprocess checking, state revision and traceable response for all eight supported binomial test cases. The optional learned Judge also completed that loop with the reference proposer. All resolved claims passed replay, provenance and exact-identity audits.

The exact checker passed all 18 regression cases, including false identities and unsupported inputs. Each of six evaluation variants dispatched the whole checker suite: **108 successful expected-outcome checks**. These are 18 distinct cases repeated six times, not 108 independent math problems.

The three 90,624-parameter Researcher stages saved native checkpoints. A separate 225-parameter symbolic Judge trained successfully. These are tiny mechanism models initialized within AIM; none is a recommended final model size or pretrained import.

| Research-loop variant | Valid proposals | Verified episodes | Actual loop tool calls |
|---|---:|---:|---:|
| Formula reference | 8/8 | 8/8 | 8 |
| SFT Researcher | 8/8 | 0/8 | 8 |
| Preference Researcher | 0/8 | 0/8 | 0 |
| RLVR Researcher | 0/8 | 0/8 | 0 |
| Formula reference + trained Judge | 8/8 | 8/8 | 8 |
| RLVR Researcher + trained Judge | 0/8 | 0/8 | 0 |

All **48 episodes** were retained and audited. There were 24 loop tool calls plus 108 checker-suite calls: **132 actual dispatches**, excluding unit tests and direct training-time reward checks. The reference+Judge arm exercised learned decisions. The RLVR+Judge arm never reached a Judge decision because generation failed first; it is not evidence of successful learned-to-learned cooperation.

## Tests and execution evidence

**140 tests passed, zero skips**, in 40.095 seconds. New coverage includes exact rational identities, independent integer convolution, nonexecution of calls/attributes, resource limits, false versus unresolved outcomes, wrong-domain Judge rejection, abstention/budget handling, confident-but-wrong forecasts, forged certificates, timeouts/errors, source/state tampering, strict JSON/citation parsing, frozen-suite integrity, and exact Researcher/Judge continuation. Existing numerical tests passed too; `eval/suite-v1.json` and its checksum were unchanged.

The first targeted development run also passed all 21 then-existing symbolic tests in 8.485 seconds. The additional canonical evaluation test brought the final new-test count to 22. No development assertion failures or infrastructure failures occurred in this phase. Expected error/timeout tests are simulated fault injections, not measured operational failure rates.

Completed local run:

`runs/20260926T181327-phase-2c-build-5e038999`

Source commit: `6c07d4ef97ca5a342d47037fcac64c32c20d61db`.

Source bundle hash: `d350bfe62905d527291c62762a1af1fe49b9fb4af48e8b08ff3c5aa811f1bcc1`.

Total recorded runtime was **63.918 seconds** on local macOS arm64, eight logical CPUs; training set Torch to one CPU thread. Recorded stage times: SFT 1.804 s, preference 1.101 s, RLVR 1.816 s, Judge 0.103 s. Summed evaluation tool time was 9.787 s. These include the harness behavior described in the source; they are integration timings, not sustained pretraining throughput or VIT fleet benchmarks. No physical lab machine was accessed.

Portable artifacts:

- [Full test log](phase-2c-evidence/tests.log)
- [Stage and evaluation metrics](phase-2c-evidence/results.json)
- [All checker outcomes and episode states](phase-2c-evidence/episodes.json)
- [Configuration, software, source hashes and export checksums](phase-2c-evidence/manifest.json)
- [Exact changed-file inventory](phase-2c-files.json)

Full datasets, optimizer-bearing checkpoints, raw generations, source snapshots, response documents, per-run manifests and SQLite provenance remain under the local run. The portable bundle contains review evidence, not the full checkpoint/ledger store. Reproduce with:

```sh
.venv/bin/python -m aim.symbolic_reproduce --export reports/<new-directory>
```

## Failed capability outcomes

The SFT model emitted well-formed but wrong expressions on every test offset. For `(x+20)**2`, it proposed `x**2+(1)*x+(1)`; the checker correctly marked it CONTRADICTED. All eight SFT proposals were contradicted.

Preference and RLVR checkpoints emitted malformed JSON or invalid field names on all eight cases. Their raw generations were retained in each episode's `researcher-output.json`; the Controller recorded unresolved generation failures without producing fabricated hypotheses or making verifier calls.

Validation finite-candidate expected reward was .76405 after SFT, .46500 after preference and .86818 after RLVR. These figures score a supplied three-answer menu. **The higher RLVR menu score did not produce a single verified free-generation answer.** Do not use it as a substitute for research-loop performance. The staged single-seed smoke run has no matched-budget control arms and cannot establish causal benefit or harm from an algorithm.

The symbolic Judge's test Brier score was .21140 versus .22222 for the training-prior baseline; log loss was .61179 versus .63651. Temperature selection chose 1.0. There were only 24 test candidate records from eight offsets. Validation and test had identical aggregate metrics because this weak syntactic representation maps these candidates to the same feature patterns. Correct and incorrect expansions often share identical features. This is a functioning calibration pipeline, not a useful general mathematical Judge or validated calibration transfer to generated candidates.

## Decisions and unresolved work

**Implemented decision:** retain the exact symbolic checker as a separate opt-in domain and the formula proposer as its default reference. Preserve the default numerical loop and all historical evaluations.

**Evidence-backed decision:** keep every new learned checkpoint experimental. No learned Researcher met even these eight elementary examples. The trained Judge does not replace verification-first behavior.

**Unresolved:** richer task distributions, a language-capable Judge, calibration transfer, a second independently implemented checker, human preference collection, reliable free-form RL, process verification, sequential decision learning and integrated-loop training. Shared checker code remains a trusted software dependency; replay is not an independently proven proof kernel.

**Next build:** create license-tracked corpus intake, resumable streaming next-token batches and tokenizer comparison interfaces for the from-scratch pretraining path. Validate them first with small local allocations. The 100M–300M proxy and subsequent ~1B/7B+ candidates still require corpus readiness and measured physical hardware. A fresh, broader symbolic task experiment can follow this component build; do not retune this exposed eight-case diagnostic to claim general reasoning.

## Changed files

The [machine-readable inventory](phase-2c-files.json) records status and SHA-256 for every phase file relative to baseline `b140bd1`, excluding the inventory itself. It includes verifier/controller/training modules, CLI, configs, separate canonical suite, tests, reproduction tooling, evidence, specification updates, READMEs and owner handoff. Historical result bundles and the research handbook PDF were not rewritten.
