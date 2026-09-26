"""Dependency-free byte baseline and small from-scratch byte-pair experiment."""
from collections import Counter
import re

from .contracts import ContractError
from .corpus import require
from .tracking import digest


class ByteTokenizer:
    version = "aim-utf8-byte-v1"
    vocab_size, bos_id, eos_id, pad_id = 259, 256, 257, 258

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens, strict=True):
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="strict" if strict else "replace")

    def specification(self):
        return {"version": self.version, "vocab_size": self.vocab_size, "encoding": "utf-8",
                "bos": self.bos_id, "eos": self.eos_id, "pad": self.pad_id,
                "learned": False, "scope": "temporary engineering tokenizer; no pretrained assets"}


def merge_pair(tokens, pair, token_id):
    result, i = [], 0
    while i < len(tokens):
        if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == pair:
            result.append(token_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


class BytePairTokenizer:
    version = "aim-byte-pair-v1"
    bos_id, eos_id, pad_id = 256, 257, 258

    def __init__(self, merges, provenance):
        require(isinstance(merges, list) and len(merges) <= 128, "BPE merge budget exceeded")
        require(isinstance(provenance, dict) and set(provenance) == {"corpus_hash", "train_hashes", "algorithm"}, "BPE fitting provenance required")
        require(provenance["algorithm"] == "global-pair-count; lexical-ID-ties; document-boundaries; v1", "unknown BPE fitting algorithm")
        require(isinstance(provenance["corpus_hash"], str) and re.fullmatch(r"[0-9a-f]{64}", provenance["corpus_hash"]) is not None and
                isinstance(provenance["train_hashes"], list) and 1 <= len(provenance["train_hashes"]) <= 10000 and
                all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) is not None for h in provenance["train_hashes"]), "invalid BPE provenance hashes")
        self.merges, self.provenance = [], provenance
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        for i, pair in enumerate(merges):
            require(isinstance(pair, (list, tuple)) and len(pair) == 2 and all(type(t) is int and t in self.vocabulary for t in pair), "invalid BPE dependency or special-token merge")
            require(tuple(pair) not in self.merges, "duplicate BPE merge")
            value = self.vocabulary[pair[0]]+self.vocabulary[pair[1]]
            require(len(value) <= 256, "merged token byte length exceeds budget")
            self.vocabulary[259+i] = value
            self.merges.append(tuple(pair))
        self.vocab_size = 259+len(self.merges)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            tokens = merge_pair(tokens, pair, 259+i)
        return tokens

    def decode(self, tokens, strict=True):
        require(all(type(t) is int and (t in self.vocabulary or t in {256, 257, 258}) for t in tokens), "unknown BPE token")
        return b"".join(self.vocabulary[t] for t in tokens if t in self.vocabulary).decode("utf-8", errors="strict" if strict else "replace")

    def specification(self):
        return {"version": self.version, "merges": [list(p) for p in self.merges], "provenance": self.provenance,
                "vocab_size": self.vocab_size, "bos": 256, "eos": 257, "pad": 258, "learned": True}


def fit_bpe(corpus, merge_count=32):
    require(type(merge_count) is int and 0 <= merge_count <= 128, "BPE merge count must be 0..128")
    rows = corpus.records("train")
    require(rows and sum(r["bytes"] for r in rows) <= 65536, "reference BPE fitter caps training bytes at 65536")
    sequences = [list(text.encode("utf-8")) for _, text in corpus.documents("train")]
    merges = []
    for _ in range(merge_count):
        counts = Counter(pair for sequence in sequences for pair in zip(sequence, sequence[1:]))
        if not counts or max(counts.values()) < 2:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        token_id = 259+len(merges)
        merges.append(list(pair))
        sequences = [merge_pair(s, pair, token_id) for s in sequences]
    return BytePairTokenizer(merges, {"corpus_hash": corpus.fingerprint, "train_hashes": [r["sha256"] for r in rows],
                                     "algorithm": "global-pair-count; lexical-ID-ties; document-boundaries; v1"})


def tokenizer_from_spec(spec):
    if spec == ByteTokenizer().specification():
        return ByteTokenizer()
    require(isinstance(spec, dict) and spec.get("version") == BytePairTokenizer.version, "unknown tokenizer contract")
    require(set(spec) == {"version", "merges", "provenance", "vocab_size", "bos", "eos", "pad", "learned"}, "invalid BPE specification fields")
    tokenizer = BytePairTokenizer(spec["merges"], spec["provenance"])
    require(tokenizer.specification() == spec, "BPE specification mismatch")
    return tokenizer


def compare_tokenizers(corpus, tokenizers, split="validation"):
    require(split == "validation", "comparison reads validation only; test requires a separate evaluation protocol")
    results = {}
    for name, tokenizer in tokenizers.items():
        rows = []
        for row, text in corpus.documents(split):
            tokens = tokenizer.encode(text)
            require(tokenizer.decode(tokens) == text, "tokenizer round trip failed")
            rows.append({"id": row["id"], "type": row["content_type"], "bytes": row["bytes"],
                         "characters": len(text), "tokens": len(tokens), "round_trip": True})
        require(bool(rows), "tokenizer comparison needs validation documents")
        results[name] = {"tokenizer_hash": digest(tokenizer.specification()), "vocab_size": tokenizer.vocab_size,
                         "documents": rows, "bytes_per_token": sum(r["bytes"] for r in rows)/sum(r["tokens"] for r in rows),
                         "scope": "compression and reversibility only; no model-quality comparison"}
    return results
