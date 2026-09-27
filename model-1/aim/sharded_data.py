"""Deterministic document ownership; no implicit elastic repartitioning."""
from .corpus import require
from .tracking import digest

SCHEMA = "aim-document-partition-v1"


def partition(corpus, world_size):
    rows = corpus.records("train")
    require(type(world_size) is int and 1 <= world_size <= min(128, len(rows)), "every rank must own at least one training document; world must be 1..128")
    ordered = sorted(rows, key=lambda r: r["id"])
    owners = [{"rank": rank, "ids": [r["id"] for r in ordered[rank::world_size]],
               "bytes": sum(r["bytes"] for r in ordered[rank::world_size])} for rank in range(world_size)]
    record = {"schema": SCHEMA, "corpus_hash": corpus.fingerprint, "world_size": world_size,
              "algorithm": "sorted-document-id-strided-v1", "owners": owners,
              "epoch_policy": "each rank repeats its own documents; no shared global epoch"}
    return {**record, "partition_hash": digest(record)}


class RankCorpus:
    def __init__(self, corpus, rank, world_size):
        self.parent = corpus
        self.partition = partition(corpus, world_size)
        require(type(rank) is int and 0 <= rank < world_size, "invalid data rank")
        self.rank = rank
        self.ids = self.partition["owners"][rank]["ids"]
        self.fingerprint = digest({"partition_hash": self.partition["partition_hash"], "rank": rank})

    def records(self, split):
        require(split == "train", "rank stream owns training data only; validation is separately evaluated")
        by_id = {r["id"]: r for r in self.parent.records("train")}
        return [by_id[i] for i in self.ids]

    def text(self, row):
        require(row["id"] in self.ids and row["split"] == "train", "rank attempted to read another partition")
        return self.parent.text(row)


def audit_partition(record, corpus):
    expected = partition(corpus, record.get("world_size"))
    require(record == expected, "partition differs from declared corpus/ownership")
    owned = [i for owner in record["owners"] for i in owner["ids"]]
    require(len(owned) == len(set(owned)) == len(corpus.records("train")), "partition coverage or exclusivity failed")
    return {"documents": len(owned), "ranks": record["world_size"], "disjoint_complete_ownership": True}
