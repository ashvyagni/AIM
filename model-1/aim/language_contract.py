"""Explicit tokenizer, response-boundary and checkpoint contracts for post-training."""
from dataclasses import asdict
from pathlib import Path

from .corpus import require
from .tokenization import ByteTokenizer, tokenizer_from_spec
from .tracking import digest

SCHEMA = "aim-tokenized-lm-v1"
KIND = "tokenized_lm"
ENCODING = {"schema": "aim-response-segments-v1", "prompt": "BOS + encode(prompt)",
            "response": "encode(response) + EOS", "boundary": "encode prompt/response separately; no cross-boundary BPE merge",
            "loss": "response tokens including EOS only", "overflow": "reject without truncation"}
OUTPUTS = {"arithmetic": "aim-arithmetic-text-v1", "external": "aim-free-text-v1",
           "symbolic": "aim-symbolic-researcher-v1"}


def output_contract(task):
    if task == "structured":
        from .research_format import VERSION
        return VERSION
    require(task in OUTPUTS, "unsupported language task")
    return OUTPUTS[task]


def checked_tokenizer(spec):
    tokenizer = tokenizer_from_spec(spec)
    require(tokenizer.decode(tokenizer.encode("AIM α\n")) == "AIM α\n", "tokenizer round trip failed")
    return tokenizer


def tokenizer_for(source=None, explicit=None):
    tokenizer = checked_tokenizer(explicit.specification()) if explicit else checked_tokenizer(source["tokenizer"]) if source else ByteTokenizer()
    if source is not None:
        require(source.get("tokenizer") == tokenizer.specification(), "tokenizer identity differs; vocabulary remapping is forbidden")
    return tokenizer


def model_config(config, tokenizer):
    from .neural import ModelConfig
    require(isinstance(config, dict), "model configuration must be an object")
    values = dict(config)
    if "vocab_size" in values:
        require(values["vocab_size"] == tokenizer.vocab_size, "model vocabulary disagrees with tokenizer")
    values["vocab_size"] = tokenizer.vocab_size
    cfg = ModelConfig(**values)
    require(cfg.parameter_estimate() <= 2_000_000 and cfg.context <= 512, "tokenized model exceeds miniature allocation guard")
    return cfg


def read_source(path):
    from .neural import read_checkpoint
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024*1024, "checkpoint file exceeds source contract")
    return read_checkpoint(path)


def validate_record(record):
    require(isinstance(record, dict) and record.get("kind") == KIND and record.get("schema") == SCHEMA, "not a versioned tokenized LM")
    fields = {"schema", "kind", "origin", "task", "stage", "model_config", "model", "reference", "optimizer", "torch_rng",
              "step", "scored_tokens", "stable_config", "dataset_hash", "encoding", "encoding_manifest_hash", "tokenizer",
              "tokenizer_hash", "output_contract", "research_contract", "parent_sha256"}
    require(set(record) == fields, "versioned checkpoint fields differ")
    from .audit_contracts import sha
    for key in ("dataset_hash", "encoding_manifest_hash", "tokenizer_hash"):
        sha(record[key])
    if record["parent_sha256"] is not None:
        sha(record["parent_sha256"])
    require(isinstance(record["stable_config"], dict) and record["stable_config"].get("task") == record["task"] and
            record["stable_config"].get("stage") == record["stage"] and "steps" not in record["stable_config"], "checkpoint configuration identity mismatch")
    require(record.get("origin") == "aim-random-init-v1", "AIM initialization lineage required")
    tokenizer = checked_tokenizer(record.get("tokenizer"))
    cfg = model_config(record.get("model_config"), tokenizer)
    require(asdict(cfg) == record["model_config"], "incomplete checkpoint architecture")
    require(record.get("tokenizer_hash") == digest(tokenizer.specification()) and record.get("encoding") == ENCODING, "encoding/tokenizer binding mismatch")
    require(record.get("output_contract") == output_contract(record.get("task")), "output contract differs from task")
    expected_research = record["output_contract"] if record["task"] in {"structured", "symbolic"} else None
    require(record.get("research_contract") == expected_research, "research adapter contract mismatch")
    require(record.get("stage") in {"sft", "preference", "rlvr"}, "unsupported checkpoint stage")
    require(type(record.get("step")) is int and 1 <= record["step"] <= 1000, "invalid checkpoint step")
    require(type(record.get("scored_tokens")) is int and record["scored_tokens"] > 0, "invalid checkpoint token accounting")
    from .language_training import validate_config
    validate_config({**record["stable_config"], "steps": record["step"]})
    require(asdict(model_config(record["stable_config"]["model"], tokenizer)) == asdict(cfg), "stable architecture differs from saved model")
    require(record["task"] != "structured" or record["stage"] == "sft", "structured stage is unsupported")
    require(record["task"] != "external" or record["stage"] != "rlvr", "external verifier adapter unavailable")
    import math
    import torch
    require(isinstance(record["model"], dict) and record["model"] and isinstance(record["reference"], dict) and
            record["model"].keys() == record["reference"].keys(), "reference/model state keys differ")
    for key, value in record["model"].items():
        reference = record["reference"][key]
        require(isinstance(value, torch.Tensor) and isinstance(reference, torch.Tensor) and
                value.shape == reference.shape and value.dtype == reference.dtype, "reference/model tensor contract differs")
    def finite(value):
        if isinstance(value, torch.Tensor):
            require(bool(torch.isfinite(value).all()), "nonfinite checkpoint tensor")
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value: finite(child)
        elif isinstance(value, float):
            require(math.isfinite(value), "nonfinite checkpoint scalar")
    for key in ("model", "reference", "optimizer"):
        finite(record[key])
    rng = record["torch_rng"]
    require(isinstance(rng, torch.Tensor) and rng.dtype == torch.uint8 and rng.shape == torch.get_rng_state().shape,
            "invalid CPU RNG state shape/type")
    return cfg, tokenizer


def validate_initialization(source, cfg, tokenizer, task, stage, resume=False):
    require(source.get("model_config") == asdict(cfg) and source.get("tokenizer") == tokenizer.specification(), "source architecture/tokenizer mismatch")
    if resume:
        validate_record(source)
        require(source["task"] == task and source["stage"] == stage, "resume cannot change task or objective")
        return
    kind = source.get("kind")
    if kind in {"pretraining_lm", "pretraining_initialization"}:
        expected = "aim-pretraining-v1" if kind == "pretraining_lm" else "aim-pretraining-initialization-v1"
        require(stage == "sft" and source.get("schema") == expected, "pretraining weights may initialize SFT only")
    elif kind == KIND:
        validate_record(source)
        require(source["task"] == task and source["output_contract"] == output_contract(task), "stage transition changes task/output contract")
    elif kind == "causal_lm":
        require(tokenizer.specification() == ByteTokenizer().specification(), "legacy post-training imports are byte-only")
        contract = source.get("research_contract")
        expected = output_contract(task) if task in {"structured", "symbolic"} else None
        require(contract == expected, "legacy source research contract mismatch")
        if task not in {"structured", "symbolic"}:
            require(source.get("stable_config", {}).get("task", "arithmetic") == task, "legacy task identity is ambiguous")
    else:
        raise ValueError("Unsupported initialization kind; distributed bundles require explicit initialization export")


def load_versioned(record):
    import torch
    from .generation import TokenizedCausalLM
    cfg, tokenizer = validate_record(record)
    model = TokenizedCausalLM(cfg)
    model.load_state_dict(record["model"])
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite model weights")
    model.eval()
    return model, tokenizer, record
