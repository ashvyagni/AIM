# Phase 3B — Distributed corpus training and recovery

## Build scope

Extend the existing CPU/Gloo infrastructure to train the native decoder on the Phase 3A corpus stream. Keep the 2M-parameter allocation limit, fixed world size, separate post-training objectives and unchanged research-loop architecture. The local launcher permits 1, 2 or 4 processes on loopback. Direct worker entry points support a declared shared-filesystem deployment, but no physical VIT host or network is validated here.

This adds actual corpus-driven DDP updates, gradient accumulation, deterministic ownership, complete-rank checkpoints, recovery, data replay audits, a serial mathematical reference and operator commands. It does not implement model/optimizer sharding, automatic node discovery, elastic membership, remote deployment or a production scheduler.

## Data ownership

`aim-document-partition-v1` sorts training document IDs and gives rank r every W-th document starting at r. It records the full assignment, byte counts, corpus hash, world size and algorithm version. Every rank must own at least one document; the contract caps W at 128. Each rank's TokenStream identity incorporates the partition hash and rank. Another rank or world size cannot reuse that cursor.

The assignment is disjoint and complete at the document level. Every rank repeats its own documents independently and executes the same configured optimizer-step count. There is no shared global epoch, document shuffling or guarantee that every source gets equal sampling weight. Shorter partitions repeat sooner. Byte-count balancing and mixed-source sampling remain separately versioned experiments; simple ID ownership provides a reproducible starting point.

The training stream retains the preceding phase's packing semantics: causal attention may cross documents inside a window, BOS/PAD prediction targets are masked, and EOS is trained. Validation is a fixed prefix evaluated centrally on rank 0's unwrapped model and distributed to the other workers. The trainer never reads test text.

## Correct loss with unequal masks and accumulation

Let S[r,a] be the sum of valid-token negative log likelihoods on rank r, accumulation microbatch a. Let C[r,a] be the number of valid targets, and C the sum of these counts across all ranks and microbatches. The intended objective is:

`L = (sum_r sum_a S[r,a]) / C`.

DDP averages rank gradients. Each local backward therefore uses `W*S[r,a]/C`, making the resulting averaged gradient equal to the global token-mean gradient. Averaging local mean losses would overweight microbatches/ranks with fewer valid targets. The global count is reduced before any backward pass; no step has a rank-specific denominator.

All but the last microbatch use `no_sync`, enclosing **both forward and backward**. The final backward synchronizes the accumulated gradient. Gradient clipping and AdamW execute once per optimizer update. These framework behaviors are documented in the pinned [PyTorch 2.8 DDP reference](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html). Gloo/collective semantics are described in the [2.8 distributed guide](https://docs.pytorch.org/docs/2.8/distributed.html).

A separate serial implementation consumes exactly the same rank-owned microbatches and differentiates the global token mean without DDP. Its parameter comparison uses a predeclared absolute tolerance of 2e-6 because reduction order differs. In contrast, resume/recovery at the same world size and environment must be bitwise exact on saved tensors and state. Do not conflate these two criteria.

## Worker and launcher contracts

Before training, ranks compare source hash, Python/Torch version, complete configuration, corpus, partition and tokenizer fingerprints. BPE fitting provenance must match the global corpus train partition. Configured accumulation is 1..8; collective timeouts are 5..120 seconds. Existing base-model, context, batch and step limits remain.

The local supervisor records command/environment overrides, captures stdout/stderr, disables automatic elastic restarts, enforces an overall deadline and cleans up the process group it launched on failure. It forces an IPv4 loopback coordinator and loopback Gloo interface. A release/bind race remains possible when choosing an available port; launch errors are retained, not hidden by automatic retry.

The direct worker uses environment-supplied ranks and Gloo rendezvous under `torchrun`. A multi-host deployment requires identical reviewed code/environment/data and a job/checkpoint directory consistently visible to every rank. It assumes authorized trusted peers and filesystem; it supplies no authentication or remote-copy service. Shared-filesystem availability and physical-node behavior have not been established.

## Coordinated checkpoints

The checkpoint protocol uses one pending directory per optimizer step:

1. Rank 0 exclusively creates `.pending-step-NNNNNN`.
2. Each rank saves its own RNG, cursor, token count, shared identities and model/optimizer digests, with a hash sidecar.
3. All workers reach the synchronization boundary.
4. Rank 0 verifies every rank file exists and agrees on step/data/configuration and replica state. It writes the common model/optimizer state and a hash-bound manifest.
5. Rank 0 renames the directory to `step-NNNNNN` within the same checkpoint parent; workers acknowledge publication.

Only published step directories load. A pending directory remains unresumable even if it contains several complete files. Missing/corrupt rank members, differing replica hashes, unexpected manifest paths and token-count inconsistencies are rejected. Prior checkpoints remain intact; nothing is overwritten or silently deleted. `checkpoint-inspect` distinguishes VALID, INVALID and INCOMPLETE entries and does not silently select an older checkpoint.

The common file stores replicated model/optimizer state once. Small rank files store per-rank stream/RNG state. This reduces checkpoint duplication, **not training-memory replication**. DDP still holds the model and optimizer on each worker. Directory rename provides a publication boundary; no fsync/power-loss guarantee, signed provenance or cross-filesystem transaction is claimed.

Resume restores the common state plus the caller's rank state. Configuration, source corpus, partition, tokenizer and world-size changes are rejected. Exact continuation is scoped to the recorded local CPU environment; changed devices, libraries, kernels or topology may alter numerical behavior. Mid-microbatch recovery is not supported: recovery starts at a completed optimizer-step checkpoint.

## Failure drills and audits

Two explicit test modes raise an exception on one worker after its rank file is saved:

- `before_publish`: the in-progress step must remain INCOMPLETE; recover from the previous published step.
- `after_publish`: the complete checkpoint remains loadable even though the job subsequently fails; recover from that published step.

The worker exception, peer/launcher failure and all checkpoint directories are retained. A terminated peer may lack a final per-rank status file; the supervisor's FAILED record and launcher log remain authoritative for job failure. These drills exercise worker-exception propagation, not a physical node loss, OS kill or power failure.

Read-only audits reconstruct each rank's stream from its recorded starting cursor, replay microbatch hashes and masks, verify global token denominators, audit ownership, and validate the final checkpoint's replica/cursor bindings. They do not independently prove optimizer math; the serial oracle supplies a separate numerical check for the measured configuration.

## Operator commands

From `model-1/`, first prepare a corpus using the [Phase 3A commands](PRETRAINING.md). Then:

```sh
.venv/bin/python -m aim distributed-preflight --corpus runs/<intake>/corpus.json --ranks 2
.venv/bin/python -m aim distributed-pretrain --corpus runs/<intake>/corpus.json --ranks 2
.venv/bin/python -m aim checkpoint-inspect runs/<job>/job/checkpoints
.venv/bin/python -m aim checkpoint-inspect runs/<job>/job/checkpoints/step-000006 --corpus runs/<intake>/corpus.json
.venv/bin/python -m aim distributed-pretrain --corpus runs/<intake>/corpus.json --ranks 2 --config configs/<longer-run>.json --resume runs/<job>/job/checkpoints/step-000006
.venv/bin/python -m aim distributed-export --checkpoint runs/<job>/job/checkpoints/step-000006
.venv/bin/python -m aim.distributed_reproduce --export reports/<new-directory>
```

The preflight checks training/validation object integrity, ownership, effective batch size and the analytical FP32 Adam parameter/gradient/moment floor, `16*N` bytes **per rank**. It excludes activations/runtime and is not an allocation guarantee. It reports local free disk without certifying cluster storage.

The initialization exporter creates an explicitly initialization-only checkpoint linked to its source bundle. The existing SFT path accepts it only with matching byte vocabulary and model dimensions; it is not a distributed-resume file. BPE→SFT remains an explicit future adapter.

For an authorized multi-host trial, invoke `aim.distributed_pretrain` under a separately configured `torchrun`, supplying `--config`, `--corpus`, `--job` and optional `--tokenizer`/`--resume` paths. Do not reuse the local launcher's loopback override for real hosts. No such deployment is performed by this phase.

## Acceptance gates, fixed before the full reproduction

All tests must pass in the pinned environment. Partition ownership must be disjoint/complete, and wrong-rank/world cursors rejected. Two-rank byte/BPE training must complete with equal replicas. Byte accumulation must agree with the serial global-token oracle within 2e-6. Split continuation and recovery from both publication-boundary faults must be exact. A four-local-rank smoke job must pass data/checkpoint audits. Initialization export must enter compatible SFT. Missing/corrupt/partial checkpoints must not load, and existing canonical research suites must remain unchanged.

Timings are reported as local integration measurements, including launch/checkpoint/audit overhead where applicable. This tiny constructed corpus does not provide model-quality, sustained-throughput, physical-node scaling or VIT feasibility evidence. Larger-scale allocation gates remain in effect.

## Next work

Phase 3C now supplies [portable node observations, policy-bound trial preparation and lexical corpus review/release](OPERATOR_AND_CORPUS_READINESS.md). Actual physical-node measurements and representative data review remain pending. Next implement tokenizer-compatible supervised training without silent vocabulary remapping. Any model/optimizer sharding, larger allocation, accelerator precision, elastic world-size change or new mixture policy requires a separately specified experiment. The current CPU DDP implementation is a reproducible systems foundation, not the final cluster strategy.
