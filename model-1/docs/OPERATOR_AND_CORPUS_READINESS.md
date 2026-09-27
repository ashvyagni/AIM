# Phase 3C — Operator observations and reviewed corpus releases

## Scope and acceptance protocol

This build prepares inputs for the existing from-scratch training system. It adds portable local observations, a bounded storage probe, fleet trial preparation, lexical contamination auditing, explicit document review, immutable subset export and replay checks. It does not discover/connect to remote hosts, authorize a lab job, import an external corpus or change a model-size guard.

Acceptance criteria, set before the full reproduction: all regression tests pass without skips; real local observations validate and omit automatic hostname/IP/path collection; absent policy blocks trial preparation; altered observations fail import; the known repeated-template corpus triggers cross-split flags; accepting all flagged documents blocks export with a retained FAILED run; the predefined fixture subset exports without changing parent objects; release replay verifies exact original bytes/metadata; and the released corpus completes a two-worker training/checkpoint audit. Synthetic four-host records are unit-test fixtures only and must not appear as measured lab results.

## Hardware observations

`aim-node-observation-v1` records an operator-supplied anonymous ID, timestamp, schema, scope and content hash. Each measurement has value, observed/unavailable status, method and error. Unsupported fields stay null, never zero estimates. Numeric observations reject booleans/negative values; zero available memory/disk is a legitimate observation.

Collected fields: host-visible logical CPUs, physical cores where available, CPU model, total RAM, Linux MemAvailable when present, free space on the selected workspace, OS, architecture, Python and installed Torch package version. macOS uses bounded sysctl queries; Linux parses `/proc/meminfo` and physical/core-ID pairs in `/proc/cpuinfo`. macOS available RAM stays unknown because this collector does not implement an equivalent availability query. Other platforms retain unsupported fields as unavailable.

These are OS observations, not an inventory certification. CPU/memory values can exceed container/job quotas. Installed package metadata does not establish successful Torch import or kernel support. NIC, switch topology, GPU throughput, thermal behavior, available lab hours and permission remain unmeasured. The collector does not automatically gather usernames, hostnames, serial numbers, IPs, credentials or full environment variables. Operators must choose non-identifying IDs; custom free-text policy fields need review before publication.

The separate storage probe writes 1–32 MiB (default 4) under its retained run, flushes and calls fsync, then checks every byte on readback. It records size, hash and elapsed write/read intervals. Readback may hit cache. A successful fsync return does not prove power-loss durability, shared storage, disk type or sustained bandwidth. The node's CPU/OS/architecture/total-RAM fields must agree with a fresh local observation; this catches obvious foreign records, not spoofing or physically identical machines. It never deletes the probe file automatically.

## Fleet plans and permission boundaries

`fleet-plan` imports 1–128 observations. Duplicate IDs/hashes are errors. Stale observations (default 24 hours), timestamps more than 60 seconds into the future, missing runtime fields or heterogeneous OS/architecture/Python/Torch create explicit blockers. Comparison is intentionally strict; relaxing it requires an experiment. IDs and hashes are declarations, not proof of distinct physical hosts.

A generated policy template starts with missing reviewer/basis/budgets/expiry and every permission false. An operator fills a separate copy and records the permission basis. `aim-lab-policy-v1` binds the exact sorted observation hashes and requires explicit lab access, background job, network port and shared-storage permissions. It includes an expiration, per-node memory/disk budgets and a 15–600 second trial deadline. The content hash detects changed content; it is not a signature or proof the reviewer has authority. No policy is manufactured for the actual local reproduction.

Preparation checks the analytical FP32 parameter/gradient/Adam-moment floor `16N` against each node's budget and available memory, and the common checkpoint tensor floor `12N × number_of_checkpoints` against its disk budget. Serialization, per-rank files, source snapshots, activations and runtime overhead are excluded; these checks cannot certify fit. Shared filesystem visibility and actual resource quotas remain unmeasured gates.

Output status is BLOCKED or READY_FOR_OPERATOR_TRIAL. A ready plan is preparation for a supervised micro trial, not automatic dispatch or scale approval. It lists feasible 1/2/4-host trial sizes with one process per host and fixed per-rank batch/accumulation/context, so the global batch grows explicitly. Operators select the real distinct hosts and use the [distributed protocol](DISTRIBUTED_PRETRAINING.md). The local loopback launcher remains local.

## Lexical corpus audit

The audit reads immutable corpus objects from all splits solely to inspect contamination. Its output exposes IDs, overlap statistics and type/split coverage, not document text or targets. This is a data-steward operation; do not feed audit holdout content into tokenizer fitting or training. The existing trainer remains train-only. Original bytes, split metadata and canonical evaluation suites are unchanged.

Normalize text with NFC then casefold, tokenize into Unicode word runs or individual non-whitespace punctuation, and form sets of five-token shingles. Very short nonempty text uses one full-token tuple and receives a short-text flag if it cannot meet the shared-shingle threshold. For sets A and B:

`Jaccard(A,B) = |A∩B| / |A∪B|`

`containment(A,B) = |A∩B| / min(|A|, |B|)`

An exact inverted index counts intersections. Default flags require at least five shared shingles and either Jaccard ≥ .8 or containment ≥ .9. Containment detects long passages wrapping a smaller document. Each result records both scores, intersection count, IDs and whether splits differ. No probabilistic sketch or similarity estimate is used. An independent brute-force test checks candidate enumeration and scores on the fixture.

Limits: 1,000 documents, 500,000 total per-document unique shingles, 100,000 candidate pairs, two million posting-pair updates and a 4 MiB compact result budget. Exceeding any bound fails the run, rather than publishing a partial clean report. These limits complement the existing 16 MiB intake cap. The algorithm is a bounded reference, not production web-scale deduplication.

Thresholds are provisional engineering choices, not empirically selected semantic-quality cutoffs. Casefolding can conflate distinct code identifiers; formulas and boilerplate can overlap legitimately; paraphrases/translation/short snippets may escape detection. Reports also flag replacement characters and short texts. NO_LEXICAL_FLAGS does not mean no contamination, good quality, valid licensing or absence of personal information. Declared group leakage and exact duplicates are still handled by intake independently.

## Review, gate and release

The generated review template is unreviewed. A separate review must bind the complete corpus/audit hashes and cover each document ID/content hash exactly once. Its purpose is engineering_fixture or production_candidate, with reviewer and basis. Every acceptance requires a reason plus rights/privacy evidence references or descriptions. Quarantine requires a reason. Unreviewed entries block export even if other documents would form a valid subset.

The gate recomputes the audit against immutable bytes before trusting the review. It requires accepted train and validation documents and rejects any accepted cross-split flagged pair. Within-split flags remain visible and require explicit document acceptance; no silent dropping or arbitrary false-positive override exists. Human/operator rights/privacy statements remain declarations; this module does not authenticate evidence, adjudicate law or detect all PII.

`corpus-export-reviewed` retains the complete decision record and gate result. A blocked export is a FAILED run with no release. A passing export copies original accepted bytes into a new source manifest and runs existing intake, preserving source URI/version, group, split and rights/privacy metadata. All original objects and quarantine decisions remain intact. The release contains parent/new corpus, audit, review and gate hashes plus exact accepted/quarantined membership.

`corpus-release-audit` recomputes the gate, checks all lineage fields, forbids escaped/symlinked corpus paths and compares every exported byte and metadata field against the parent. Only local path changes are allowed. It verifies a chain of declared review decisions and content, not the scientific/legal truth of those decisions. Hashes are unsigned.

The reproduction uses the already authored 20-document fixture. The predeclared passing subset keeps train prose, validation math and test Unicode. This demonstrates quarantine and preserves five documents; it creates a deliberately unrepresentative mixture and cannot recommend data proportions or evaluate model generalization. No external corpus is approved by the demonstration.

## Commands

From `model-1/`, replace placeholders with actual paths:

```sh
.venv/bin/python -m aim node-audit --node-id lab-anonymous-01
.venv/bin/python -m aim storage-probe --node runs/<node-audit>/node.json --mebibytes 4
.venv/bin/python -m aim fleet-plan --nodes runs/<node-audit>/node.json
.venv/bin/python -m aim corpus-audit --corpus runs/<intake>/corpus.json
```

Review a copy of `policy-template.json` or `review-template.json`. Fill actual declarations; do not relabel the fixture's procedural review as human annotation. Then record a hash and run the relevant gate:

```sh
.venv/bin/python -m aim audit-seal --input <completed-review>.json --output <sealed-review>.json
.venv/bin/python -m aim corpus-gate --corpus runs/<intake>/corpus.json --audit runs/<audit>/audit.json --review <sealed-review>.json
.venv/bin/python -m aim corpus-export-reviewed --corpus runs/<intake>/corpus.json --audit runs/<audit>/audit.json --review <sealed-review>.json
.venv/bin/python -m aim corpus-release-audit --release runs/<export>/release.json --parent-corpus runs/<intake>/corpus.json --audit runs/<audit>/audit.json
.venv/bin/python -m aim fleet-plan --nodes <node-records...> --policy <sealed-policy>.json
.venv/bin/python -m aim.readiness_reproduce --export reports/<new-directory>
```

`audit-seal` only appends a content hash to a recognized review/policy schema and refuses overwriting an existing output. It does not validate approvals; consumers validate fields and bindings. `corpus-gate` returns exit code 2 for a structurally valid blocked result, retaining a completed diagnostic run. `corpus-export-reviewed` raises on the same blocked condition, retaining a FAILED attempted export. Failed imports remain FAILED. Read each result status, not only process completion.

## Next gates

Have the authorized lab operator collect real node observations and resolve missing permission/storage/network/runtime measurements. Run one physical host first, then matched 2/4-host jobs with measured peaks, sustained throughput and shared-checkpoint recovery. Separately review a representative licensed corpus with family-aware splits and independently assessed deduplication precision/recall. A scalable tokenizer/loader and larger supervised vocabulary adapter remain future builds. Keep Researcher/Judge separation, stage objectives, the 2M guard and conditional scale roadmap.
