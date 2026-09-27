"""Read-only checkpoint descriptions and stage plans; no inferred training success."""
from pathlib import Path

from .corpus import require
from .language_contract import KIND, load_versioned, read_source, validate_record
from .tracking import Run, digest, file_hash, write_json


def describe(path):
    record = read_source(path)
    cfg, tokenizer = validate_record(record)
    return {"schema": "aim-language-checkpoint-description-v1", "checkpoint_sha256": file_hash(path),
            "kind": record["kind"], "task": record["task"], "stage": record["stage"], "step": record["step"],
            "parameters": cfg.parameter_estimate(), "model_config": record["model_config"],
            "tokenizer_hash": digest(tokenizer.specification()), "vocabulary": tokenizer.vocab_size,
            "output_contract": record["output_contract"], "research_contract": record["research_contract"],
            "dataset_hash": record["dataset_hash"], "encoding_manifest_hash": record["encoding_manifest_hash"],
            "parent_sha256": record["parent_sha256"], "scored_tokens": record["scored_tokens"],
            "scope": "integrity/schema description, not tensor validation or evidence of model capability"}


def generate_run(checkpoint, prompt, max_new_tokens, runs):
    from .generation import generate
    with Run(Path(runs), "language-generate", {"max_new_tokens": max_new_tokens, "prompt_hash": digest(prompt.encode("utf-8"))}, [Path(checkpoint)]) as run:
        model, tokenizer, record = load_versioned(read_source(checkpoint))
        result = generate(model, tokenizer, prompt, max_new_tokens)
        write_json(run.path/"prompt.json", {"prompt": prompt})
        write_json(run.path/"generation.json", {**result, "checkpoint_sha256": file_hash(checkpoint), "output_contract": record["output_contract"]})
        require(result["error"] is None, "generation rejected; retained generation.json")
    return run.path


def stage_plan(checkpoint):
    """A recommendation record only: no launch, reward mix, or automatic promotion."""
    record = read_source(checkpoint)
    cfg, tokenizer = validate_record(record)
    supported = ["sft"] if record["task"] == "structured" else ["sft", "preference"] if record["task"] == "external" else ["sft", "preference", "rlvr"]
    return {"schema": "aim-language-stage-plan-v1", "parent_sha256": file_hash(checkpoint),
            "task": record["task"], "current_stage": record["stage"], "supported_separate_stages": supported,
            "tokenizer_hash": digest(tokenizer.specification()), "model_config": record["model_config"],
            "initialize_semantics": "load weights; reset optimizer and freeze a fresh reference",
            "resume_semantics": "same task/stage/config/data/encoding; restore optimizer/reference/RNG/counters",
            "judge": "separate event-specific training; never inherited from LM tokenizer",
            "required_before_promotion": ["pending regression and exact-resume checks", "held-out generation and verifier evaluation",
                                          "matched stage budgets and shift evaluation", "human annotation audit if claiming RLHF"],
            "scope": "compatibility planning only; no success or final architecture approval"}
