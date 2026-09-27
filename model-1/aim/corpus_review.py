"""Bounded exact lexical overlap audit; no source mutation or semantic assurance."""
from collections import Counter, defaultdict
import re
import unicodedata
from pathlib import Path

from .audit_contracts import fields, number, seal, verify
from .corpus import Corpus, require
from .tracking import Run, canonical, write_json

SCHEMA = "aim-corpus-overlap-audit-v1"
DEFAULT = {"shingle_width": 5, "jaccard_threshold": .8, "containment_threshold": .9,
           "minimum_shared_shingles": 5, "max_documents": 1000, "max_total_shingles": 500000,
           "max_candidate_pairs": 100000, "max_pair_updates": 2000000}


def validate_config(config):
    fields(config, DEFAULT, "overlap configuration")
    for key, maximum in (("shingle_width", 16), ("minimum_shared_shingles", 100), ("max_documents", 1000),
                         ("max_total_shingles", 500000), ("max_candidate_pairs", 100000), ("max_pair_updates", 2000000)):
        require(type(config[key]) is int and 1 <= config[key] <= maximum, key+" outside bounded audit limits")
    for key in ("jaccard_threshold", "containment_threshold"):
        number(config[key], key, .01)
        require(config[key] <= 1, "overlap threshold exceeds one")


def shingles(value, width):
    tokens = re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", value).casefold(), flags=re.UNICODE)
    if not tokens:
        return set()
    if len(tokens) < width:
        return {tuple(tokens)}
    return {tuple(tokens[i:i+width]) for i in range(len(tokens)-width+1)}


def audit(corpus, config=None):
    config = dict(DEFAULT if config is None else config)
    validate_config(config)
    rows = sorted(corpus.index["documents"], key=lambda r: r["id"])
    require(len(rows) <= config["max_documents"], "audit document budget exceeded; no partial clean report")
    postings, features, candidates, documents = defaultdict(list), [], Counter(), []
    total = pair_updates = 0
    # Reads every split solely for contamination audit, never for fitting/tokenizer training.
    for i, row in enumerate(rows):
        content = corpus.text(row)
        feature = shingles(content, config["shingle_width"])
        total += len(feature)
        require(total <= config["max_total_shingles"], "audit feature budget exceeded; no partial clean report")
        for shingle in sorted(feature):
            for previous in postings[shingle]:
                pair_updates += 1
                require(pair_updates <= config["max_pair_updates"], "audit pair-update budget exceeded; no partial clean report")
                candidates[(previous, i)] += 1
                require(len(candidates) <= config["max_candidate_pairs"], "audit candidate-pair budget exceeded; no partial clean report")
            postings[shingle].append(i)
        features.append(feature)
        documents.append({"id": row["id"], "sha256": row["sha256"], "split": row["split"], "group": row["group"],
                          "content_type": row["content_type"], "bytes": row["bytes"], "shingles": len(feature),
                          "quality_flags": (["short_for_overlap_detection"] if len(feature) < config["minimum_shared_shingles"] else [])+
                          (["replacement_character_present"] if "\ufffd" in content else [])})
    pairs = []
    for (i, j), intersection in sorted(candidates.items()):
        union = len(features[i])+len(features[j])-intersection
        jaccard = intersection/union
        containment = intersection/min(len(features[i]), len(features[j]))
        if intersection >= config["minimum_shared_shingles"] and (jaccard >= config["jaccard_threshold"] or containment >= config["containment_threshold"]):
            pairs.append({"left": rows[i]["id"], "right": rows[j]["id"], "shared_shingles": intersection,
                          "jaccard": jaccard, "containment": containment, "cross_split": rows[i]["split"] != rows[j]["split"]})
    counts = {split: dict(Counter(r["content_type"] for r in rows if r["split"] == split)) for split in ("train", "validation", "test")}
    result = {"schema": SCHEMA, "corpus_hash": corpus.fingerprint, "configuration": config, "documents": documents,
                 "pairs": pairs, "counts_by_split_and_type": counts, "total_unique_shingles_per_document": total,
                 "candidate_pairs_examined": len(candidates), "pair_updates": pair_updates,
                 "cross_split_pairs": sum(p["cross_split"] for p in pairs),
                 "status": "REVIEW_REQUIRED" if pairs or any(d["quality_flags"] for d in documents) else "NO_LEXICAL_FLAGS",
                 "scope": "exact set overlap over NFC/casefold word-punctuation shingles; all splits inspected; no semantic/legal/privacy certification"}
    # Leave room for pretty-print whitespace and the integrity field within the read budget.
    require(len(canonical(result).encode("utf-8")) <= 4*1024*1024, "audit output budget exceeded; no partial clean report")
    return seal(result)


def validate_audit(record, corpus):
    verify(record, SCHEMA)
    require(record.get("corpus_hash") == corpus.fingerprint, "audit belongs to another corpus")
    # Recompute against original bytes; a rewritten hash alone cannot approve changed pairs.
    require(audit(corpus, record.get("configuration")) == record, "overlap audit does not replay")
    return record


def audit_run(corpus_path, runs, config=None):
    config = dict(DEFAULT if config is None else config)
    with Run(Path(runs), "corpus-review-audit", config, [Path(corpus_path)]) as run:
        report = audit(Corpus(corpus_path), config)
        write_json(run.path/"audit.json", report)
        from .corpus_release import review_template
        write_json(run.path/"review-template.json", review_template(report))
    return run.path/"audit.json"
