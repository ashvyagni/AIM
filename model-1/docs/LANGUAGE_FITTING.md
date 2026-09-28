# Language fitting diagnostics

Phase 3E extends the versioned language path with fixed train/validation probes. It does not introduce a new model architecture or merge training objectives. Read the [predeclared protocol](experiments/phase-3e-protocol.md) for the exact experiment and interpretation limits.

**Executed evidence:** [221 tests passed; eight arms, 32 measurements and 40 Controller replays](../reports/phase-3e-implementation.md). Structure improved after 384 updates, but each arm had zero validation checker passes. The standalone diagnostic command exactly reproduced one recorded measurement with one compute thread requested through `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Neither the study nor the CLI check promotes a learned model.

## Interfaces

`aim.language_diagnostics.freeze_probes` selects the first requested rows from both prepared splits, binds the full prepared dataset hash, and freezes all selected rows. Only structured numerical and symbolic SFT tasks are supported. Test/OOD rows and changed manifests reject. Checkpoint evaluation also reconstructs the expected probes from the checkpoint's adjacent `dataset.json`; a matching declared dataset hash alone is insufficient.

`measure` runs teacher-forced scoring and strict greedy generation without changing model weights, mode or CPU RNG. `measure_checkpoint` additionally binds task, tokenizer, model and dataset identities. Keep a checkpoint with its `.sha256.json` sidecar and `dataset.json` when using this command:

```sh
.venv/bin/python -m aim.language_diagnostics \
  --checkpoint <training-run>/checkpoint.pt \
  --probes <study-run>/structured-probes.json \
  --max-new-tokens 128
```

The command creates an immutable tracked run containing `diagnostics.json`; failures remain failed tracked runs. It does not modify the checkpoint or research memory.

## Measurements and boundaries

Each probe records gold-prefix response NLL, response-token accuracy, gold-position EOS probability, reference/argmax token IDs, response-length coverage, generated IDs/text, termination reason and first reference divergence. EOS counts as a response target. The aggregate NLL weights tokens; the EOS diagnostic is a mean over examples. Direct raw token-NLL comparisons across vocabularies are not a quality ranking.

JSON syntax, output-contract validity and semantic checks are separate. Parser positions refer to character/UTF-8 byte offsets where the parser supplies one. A missing closing delimiter can point to end-of-input; that is not a particular invalid generated token. Duplicate/schema errors may have null positions. Invalid generated BOS/PAD has a separate token index. A different valid polynomial expression can pass the checker without matching reference tokens.

The numerical diagnostic uses exact rational residuals against prompt observations and the fixture target, with tolerance 1e-8. Both must agree for its diagnostic PASS. Its target is evaluator-only fixture information; only prompt text reaches generation. This check has no claim-changing authority and does not prove source provenance or global correctness. The real numerical Controller checks its own target/provenance scope separately. The symbolic diagnostic uses the existing bounded Q[x] identity checker; UNKNOWN, FAIL and invalid output are distinct.

## Study runner

```sh
.venv/bin/python -m aim.language_fit \
  --config configs/language-fitting.json \
  --regression --export reports/<new-directory>
```

The fixed plan uses two domains, byte/BPE32 tokenizers, two seeds, a random-init baseline and three SFT budgets. Checkpoint continuation retains the original frozen reference. The runner writes prepared data, frozen probes, tokenizer specs, per-checkpoint diagnostics, training runs, actual final-budget Controller episodes, and positive reference controls. It creates no research test/OOD file. Existing tokenizer corpus partitions retain their historical role; BPE fitting reads its train split only.

`audit_study` rechecks all planned arm/budget/case identities, reconstructs initial weights, binds checkpoints and probes, decodes stored generation IDs, re-scores candidates, recomputes aggregates, compares final diagnostic generation with actual Controller traces, and replays state ledgers. It checks saved likelihood summaries for internal consistency; it does not independently recompute neural logits. Hashes detect ordinary changes but are not authenticated signatures.

Export is explicit and creates a new directory. It copies only this authored study's diagnostics/probes/tokenizers and summary, with source/environment/configuration hashes. Full model/optimizer checkpoints, source snapshots, data and state ledgers remain in local `runs/`. This exporter is not a generic publishing tool for private external data.

No best checkpoint is selected. Report every predeclared budget and seed, including lower validity or failed fitting. Eight probes per arm are too few for a broad capability claim, and repeated budgets share examples. Learned models remain experimental regardless of whether fitting succeeds. A fresh family-disjoint capability protocol is a separate next decision.
