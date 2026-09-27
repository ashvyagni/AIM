"""Coordinated checkpoint publication on one shared filesystem.

Rank files and common state are staged first. Only the renamed directory is
loadable. Hashes detect corruption; they are not signatures or power-loss proof.
"""
import re
from pathlib import Path

from .corpus import read_json, require
from .tracking import digest, file_hash, write_json

SCHEMA = "aim-ddp-checkpoint-bundle-v1"


def tree_hash(value):
    import torch
    def encode(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {"tensor_dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": digest(tensor.numpy().tobytes())}
        if isinstance(item, dict):
            return {"mapping": [[repr(k), encode(v)] for k, v in sorted(item.items(), key=lambda kv: repr(kv[0]))]}
        if isinstance(item, (list, tuple)):
            return {type(item).__name__: [encode(x) for x in item]}
        return item
    return digest(encode(value))


def prepare(root, step):
    require(type(step) is int and 1 <= step <= 1000, "invalid bundle step")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pending = root/f".pending-step-{step:06d}"
    require(not (root/f"step-{step:06d}").exists(), "checkpoint destination already exists")
    pending.mkdir(exist_ok=False)
    return pending


def save_record(path, record):
    import torch
    path = Path(path)
    with path.open("xb") as stream:
        torch.save(record, stream)
    write_json(path.with_name(path.name+".sha256.json"), {"sha256": file_hash(path)})


def read_record(path):
    from .neural import read_checkpoint
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64*1024*1024, "checkpoint file absent, linked or over budget")
    require(not path.with_name(path.name+".sha256.json").is_symlink(), "linked checkpoint sidecar")
    return read_checkpoint(path)


def validate_states(common, ranks):
    from .neural import ModelConfig
    from .tokenization import tokenizer_from_spec
    require(common.get("kind") == "distributed_pretraining_lm" and common.get("schema") == SCHEMA, "invalid distributed common state")
    world = common.get("world_size")
    require(type(world) is int and 1 <= world <= 128 and len(ranks) == world, "checkpoint rank count mismatch")
    require(type(common.get("step")) is int and 1 <= common["step"] <= 1000, "invalid distributed checkpoint step")
    require(type(common.get("global_tokens")) is int and common["global_tokens"] > 0, "invalid global checkpoint token count")
    tokenizer = tokenizer_from_spec(common["tokenizer"])
    cfg = ModelConfig(**common["model_config"])
    require(cfg.parameter_estimate() <= 2_000_000 and cfg.context <= 512 and cfg.vocab_size == tokenizer.vocab_size, "checkpoint allocation/tokenizer mismatch")
    model_hash, optimizer_hash = tree_hash(common["model"]), tree_hash(common["optimizer"])
    for rank, state in enumerate(ranks):
        require(state.get("kind") == "distributed_rank" and state.get("schema") == SCHEMA and state.get("rank") == rank and state.get("world_size") == world,
                "checkpoint rank identity mismatch")
        require(all(state.get(k) == common.get(k) for k in ("step", "corpus_hash", "partition_hash", "tokenizer_hash", "config_hash", "global_tokens")),
                "rank/common checkpoint identity mismatch")
        require(state.get("model_hash") == model_hash and state.get("optimizer_hash") == optimizer_hash, "replica weights/optimizer differ at checkpoint")
        require(type(state.get("local_tokens")) is int and state["local_tokens"] >= 0, "invalid local token count")
    require(sum(r["local_tokens"] for r in ranks) == common["global_tokens"], "global/local checkpoint token counts differ")
    require(common["tokenizer_hash"] == digest(common["tokenizer"]) and common["config_hash"] == digest(common["stable_config"]), "checkpoint configuration hash mismatch")
    return {"step": common["step"], "world_size": world, "model_hash": model_hash, "optimizer_hash": optimizer_hash,
            "global_tokens": common["global_tokens"], "replicas_equal": True}


def publish(pending, common):
    pending = Path(pending)
    step, world = common["step"], common["world_size"]
    require(pending.name == f".pending-step-{step:06d}", "checkpoint staging identity mismatch")
    require(type(world) is int and 1 <= world <= 128, "invalid publication world size")
    ranks = [read_record(pending/f"rank-{r:03d}.pt") for r in range(world)]
    audit = validate_states(common, ranks)
    save_record(pending/"common.pt", common)
    names = ["common.pt"]+[f"rank-{r:03d}.pt" for r in range(world)]
    names += [n+".sha256.json" for n in list(names)]
    manifest = {"schema": SCHEMA, "step": step, "world_size": world, "audit": audit,
                "files": {n: file_hash(pending/n) for n in names}}
    write_json(pending/"bundle.json", {**manifest, "bundle_hash": digest(manifest)})
    destination = pending.parent/f"step-{step:06d}"
    require(not destination.exists(), "cannot replace published checkpoint")
    pending.rename(destination)
    return destination


def load_bundle(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink() and re.fullmatch(r"step-\d{6}", path.name) is not None,
            "only published step directories can be loaded; pending bundles are incomplete")
    manifest = read_json(path/"bundle.json")
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA and
            manifest.get("bundle_hash") == digest({k: v for k, v in manifest.items() if k != "bundle_hash"}), "checkpoint manifest hash mismatch")
    world = manifest.get("world_size")
    require(type(world) is int and 1 <= world <= 128 and path.name == f'step-{manifest["step"]:06d}', "invalid checkpoint manifest identity")
    names = ["common.pt"]+[f"rank-{r:03d}.pt" for r in range(world)]
    names += [n+".sha256.json" for n in list(names)]
    require(set(manifest["files"]) == set(names), "unexpected checkpoint file set")
    for name, expected in manifest["files"].items():
        file = path/name
        require(file.is_file() and not file.is_symlink() and file.stat().st_size <= 64*1024*1024 and file_hash(file) == expected,
                "checkpoint member missing or corrupt: "+name)
    common = read_record(path/"common.pt")
    ranks = [read_record(path/f"rank-{r:03d}.pt") for r in range(world)]
    require(common["step"] == manifest["step"] and common["world_size"] == world and validate_states(common, ranks) == manifest["audit"],
            "checkpoint payload differs from manifest")
    return common, ranks, manifest


def scan(root):
    records = []
    for path in sorted(Path(root).iterdir()):
        if path.name.startswith(".pending-step-"):
            records.append({"name": path.name, "status": "INCOMPLETE", "loadable": False})
        elif path.name.startswith("step-"):
            try:
                _, _, manifest = load_bundle(path)
                records.append({"name": path.name, "status": "VALID", "loadable": True, "bundle_hash": manifest["bundle_hash"]})
            except (ValueError, OSError, KeyError) as exc:
                records.append({"name": path.name, "status": "INVALID", "loadable": False, "error": str(exc)})
    return records
