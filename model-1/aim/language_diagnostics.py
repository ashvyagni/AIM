"""Bounded fitting probes: likelihood, generation structure and scoped checks.

Reference-token divergence is not a syntax error: alternative valid responses
can differ from the reference. These diagnostics never mutate research claims.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import ContractError
from .corpus import read_json, require
from .generation import generate
from .research_format import _invalid_constant, _unique_object, parse_hypothesis
from .response_encoding import encode_pair
from .tracking import Run, digest, file_hash, write_json

SCHEMA = "aim-language-fitting-probes-v1"
REPORT = "aim-language-diagnostics-v1"


def freeze_probes(data, count=4):
    require(type(count) is int and 1 <= count <= 16, "probe count must be 1..16 per split")
    require(data.get("schema") == "aim-prepared-language-data-v1" and data.get("task") in {"structured", "symbolic"}
            and data.get("stage") == "sft", "probes require prepared research SFT data")
    require(set(data) == {"schema", "version", "task", "stage", "output_contract", "train", "validation", "dataset_hash"},
            "probe inputs must contain train/validation only")
    require(digest({k: v for k, v in data.items() if k != "dataset_hash"}) == data["dataset_hash"], "prepared data hash mismatch")
    require(all(len(data[s]) >= count for s in ("train", "validation")), "not enough fixed probes")
    result = {"schema": SCHEMA, "task": data["task"], "dataset_hash": data["dataset_hash"],
              "selection": "first rows in immutable prepared split order; no outcome selection",
              "scope": "exposed train/validation fitting probes; no test/OOD inference",
              "rows": [dict(row) for split in ("train", "validation") for row in data[split][:count]]}
    result["probe_hash"] = digest(result)
    validate_probes(result)
    return result


def validate_probes(probes):
    require(isinstance(probes, dict) and set(probes) == {"schema", "task", "dataset_hash", "selection", "scope", "rows", "probe_hash"},
            "probe manifest fields differ")
    require(probes["schema"] == SCHEMA and probes["task"] in {"structured", "symbolic"}, "unsupported probe task")
    require(probes["probe_hash"] == digest({k: v for k, v in probes.items() if k != "probe_hash"}), "probe hash mismatch")
    from .audit_contracts import sha
    sha(probes["dataset_hash"])
    rows = probes["rows"]
    require(isinstance(rows, list) and 2 <= len(rows) <= 32, "probe row budget exceeded")
    ids, groups = set(), {s: set() for s in ("train", "validation")}
    for row in rows:
        require(isinstance(row, dict) and row.get("split") in groups, "holdout/test rows forbidden in fitting probes")
        for key in ("id", "group", "prompt", "response"):
            require(isinstance(row.get(key), str) and 0 < len(row[key].encode("utf-8")) <= 262144, "invalid probe "+key)
        require(row["id"] not in ids, "duplicate probe ID")
        ids.add(row["id"])
        groups[row["split"]].add(row["group"])
    require(all(groups.values()) and not groups["train"] & groups["validation"], "probe groups must be disjoint across splits")


def divergence(expected, observed):
    """Zero-based first reference mismatch, including missing/extra EOS."""
    for index in range(max(len(expected), len(observed))):
        a = expected[index] if index < len(expected) else None
        b = observed[index] if index < len(observed) else None
        if a != b:
            return {"response_token_index": index, "expected_id": a, "observed_id": b}
    return None


def score_text(task, row, text):
    result = {"syntax_valid": False, "contract_valid": False, "error_category": None,
              "error": None, "syntax_error_character_offset": None, "syntax_error_byte_offset": None,
              "check": {"outcome": "NOT_RUN", "scope": "no valid candidate"}}
    if text is None:
        return {**result, "error_category": "decoding", "error": "no decoded text"}
    try:
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        result.update(error_category="syntax", error=str(exc))
        if isinstance(exc, json.JSONDecodeError):
            result.update(syntax_error_character_offset=exc.pos,
                          syntax_error_byte_offset=len(text[:exc.pos].encode("utf-8")))
        return result
    result["syntax_valid"] = True
    try:
        if task == "symbolic":
            from .symbolic import check_identity, identity_request
            from .symbolic_loop import parse_symbolic_response
            candidate = parse_symbolic_response(text, [SimpleNamespace(id="E0")])
            checked = check_identity(identity_request(row["lhs"], candidate.rhs))
            result["check"] = {"checker": "exact-polynomial-checker-v1", "outcome": checked["outcome"],
                               "scope": "bounded Q[x] identity; no literature or discovery claim", "detail": checked}
        elif task == "structured":
            candidate = parse_hypothesis(text, row["aliases"])[0]
            coefficients = [Fraction(str(c)) for c in candidate.coefficients]
            prompt = json.loads(row["prompt"])
            def value(x):
                return sum(c * Fraction(str(x))**i for i, c in enumerate(coefficients))
            residuals = [abs(value(x)-Fraction(str(y))) for points in prompt["evidence"].values() for x, y in points]
            measurements = row["case"]["experiment"]["observations"]
            require(len(measurements) == 1 and measurements[0][0] == prompt["x"], "probe target fixture mismatch")
            target_residual = abs(value(prompt["x"])-Fraction(str(measurements[0][1])))
            tolerance = Fraction(1, 100_000_000)
            observed = bool(residuals) and all(r <= tolerance for r in residuals)
            target = target_residual <= tolerance
            result["check"] = {"checker": "aim-fitting-polynomial-probe-v1", "outcome": "PASS" if observed and target else "FAIL",
                               "observations_agree": observed, "target_agrees": target,
                               "observation_residuals": [str(r) for r in residuals], "target_residual": str(target_residual),
                               "scope": "exact rational fixture residuals at tolerance 1e-8; no provenance or global-law proof; no claim mutation"}
        else:
            raise ContractError("unsupported fitting task")
    except ContractError as exc:
        result.update(error_category=getattr(exc, "category", "contract"), error=str(exc))
        return result
    result["contract_valid"] = True
    return result


@torch.no_grad()
def teacher_forced(model, tokenizer, row):
    encoded = encode_pair(tokenizer, row["prompt"], row["response"], model.cfg.context)
    logits = model(torch.tensor([encoded["inputs"]]))[0]
    require(bool(torch.isfinite(logits).all()), "nonfinite diagnostic logits")
    mask = torch.tensor(encoded["mask"], dtype=torch.bool)
    target = torch.tensor(encoded["targets"])[mask]
    logp = logits[mask].log_softmax(-1)
    selected = logp.gather(-1, target[:, None]).squeeze(-1)
    predicted = logp.argmax(-1)
    return {"response_tokens": len(target), "negative_log_likelihood_sum": -float(selected.sum()),
            "response_token_nll": -float(selected.mean()), "correct_tokens": int((predicted == target).sum()),
            "reference_ids": target.tolist(), "argmax_ids": predicted.tolist(),
            "first_reference_divergence": divergence(target.tolist(), predicted.tolist()),
            "gold_eos_probability": float(logp[-1, tokenizer.eos_id].exp()),
            "scope": "conditioned on gold response prefix including gold EOS position; not generated accuracy"}


def summarize(rows):
    result = {}
    for split in ("train", "validation"):
        group = [row for row in rows if row["split"] == split]
        require(bool(group), "diagnostic summary needs both splits")
        tokens = sum(r["teacher"]["response_tokens"] for r in group)
        result[split] = {"examples": len(group), "response_tokens": tokens,
            "response_token_nll": sum(r["teacher"]["negative_log_likelihood_sum"] for r in group)/tokens,
            "teacher_token_accuracy": sum(r["teacher"]["correct_tokens"] for r in group)/tokens,
            "gold_eos_probability_mean": sum(r["teacher"]["gold_eos_probability"] for r in group)/len(group),
            "reference_fits_budget": sum(r["coverage"]["reference_fits_budget"] for r in group),
            "exact_reference_with_eos": sum(r["first_reference_divergence"] is None for r in group),
            "syntax_valid": sum(r["scoring"]["syntax_valid"] for r in group),
            "contract_valid": sum(r["scoring"]["contract_valid"] for r in group),
            "scoped_check_pass": sum(r["scoring"]["check"]["outcome"] == "PASS" for r in group),
            "termination_counts": dict(sorted(Counter(r["generation"]["termination"] for r in group).items()))}
    return result


@torch.no_grad()
def measure(model, tokenizer, probes, max_new_tokens=128):
    from .checkpoint_bundle import tree_hash
    validate_probes(probes)
    require(type(max_new_tokens) is int and 1 <= max_new_tokens <= 512, "diagnostic generation budget must be 1..512")
    require(model.cfg.vocab_size == tokenizer.vocab_size, "diagnostic vocabulary mismatch")
    # Reject invalid coverage before partially executing a measurement.
    for row in probes["rows"]:
        encode_pair(tokenizer, row["prompt"], row["response"], model.cfg.context)
        require(1+len(tokenizer.encode(row["prompt"]))+max_new_tokens <= model.cfg.context, "probe generation request exceeds context")
    before_rng = torch.get_rng_state().clone()
    before_hash = tree_hash(model.state_dict())
    training = model.training
    model.eval()
    records = []
    try:
        for row in probes["rows"]:
            teacher = teacher_forced(model, tokenizer, row)
            generated = generate(model, tokenizer, row["prompt"], max_new_tokens)
            records.append({"id": row["id"], "group": row["group"], "split": row["split"], "teacher": teacher,
                "coverage": {"prompt_tokens": generated["prompt_tokens"], "reference_tokens_including_eos": teacher["response_tokens"],
                             "max_new_tokens": max_new_tokens, "reference_fits_budget": teacher["response_tokens"] <= max_new_tokens},
                "generation": generated, "first_reference_divergence": divergence(teacher["reference_ids"], generated["generated_ids"]),
                "first_invalid_special_token": next((i for i, t in enumerate(generated["generated_ids"]) if t in {tokenizer.bos_id, tokenizer.pad_id}), None),
                "scoring": score_text(probes["task"], row, generated["text"])})
    finally:
        model.train(training)
    require(torch.equal(before_rng, torch.get_rng_state()) and tree_hash(model.state_dict()) == before_hash,
            "diagnostics changed model or CPU RNG")
    return {"schema": REPORT, "probe_hash": probes["probe_hash"], "dataset_hash": probes["dataset_hash"],
            "task": probes["task"], "model_state_hash": before_hash, "tokenizer_hash": digest(tokenizer.specification()),
            "max_new_tokens": max_new_tokens, "rows": records, "summary": summarize(records),
            "scope": "fixed train/validation diagnostics; reference divergence is not syntax invalidity; no capability promotion"}


def measure_checkpoint(checkpoint, probes, max_new_tokens=128):
    from .neural import load_lm
    validate_probes(probes)
    model, tokenizer, record = load_lm(checkpoint)
    require(record.get("kind") == "tokenized_lm" and record["task"] == probes["task"] and
            record["dataset_hash"] == probes["dataset_hash"], "checkpoint does not bind this probe dataset/task")
    data = read_json(Path(checkpoint).parent/"dataset.json", 10*1024*1024)
    count = sum(row["split"] == "train" for row in probes["rows"])
    require(freeze_probes(data, count) == probes, "probe rows are not the declared prefix of the checkpoint dataset")
    result = measure(model, tokenizer, probes, max_new_tokens)
    return {**result, "checkpoint_sha256": file_hash(checkpoint), "step": record["step"]}


def audit_measurement(report, probes, tokenizer, context, max_new_tokens):
    """Re-score stored tokens/text and aggregates without replaying neural logits."""
    validate_probes(probes)
    require(report["schema"] == REPORT and report["task"] == probes["task"] and
            report["probe_hash"] == probes["probe_hash"] and report["dataset_hash"] == probes["dataset_hash"] and
            report["tokenizer_hash"] == digest(tokenizer.specification()) and report["max_new_tokens"] == max_new_tokens,
            "diagnostic identities or budget differ")
    require(len(report["rows"]) == len(probes["rows"]), "diagnostic row count differs")
    for row, original in zip(report["rows"], probes["rows"]):
        require(all(row[k] == original[k] for k in ("id", "group", "split")), "probe order/identity changed")
        encoded = encode_pair(tokenizer, original["prompt"], original["response"], context)
        gold = tokenizer.encode(original["response"])+[tokenizer.eos_id]
        teacher, generation = row["teacher"], row["generation"]
        require(teacher["reference_ids"] == gold and teacher["response_tokens"] == len(gold), "teacher reference changed")
        predicted = teacher["argmax_ids"]
        require(isinstance(predicted, list) and len(predicted) == len(gold) and
                all(type(t) is int and 0 <= t < tokenizer.vocab_size for t in predicted), "invalid teacher argmax IDs")
        require(teacher["first_reference_divergence"] == divergence(gold, predicted) and
                teacher["correct_tokens"] == sum(a == b for a, b in zip(gold, predicted)), "teacher token summary changed")
        for key in ("negative_log_likelihood_sum", "response_token_nll", "gold_eos_probability"):
            require(type(teacher[key]) in (int, float) and math.isfinite(teacher[key]) and teacher[key] >= 0, "invalid teacher metric")
        require(teacher["gold_eos_probability"] <= 1 and math.isclose(teacher["response_token_nll"],
                teacher["negative_log_likelihood_sum"]/len(gold), rel_tol=1e-6, abs_tol=1e-7), "teacher NLL/EOS metric inconsistent")
        prompt_tokens = 1+len(tokenizer.encode(original["prompt"]))
        require(generation["schema"] == "aim-greedy-generation-v1" and generation["prompt_hash"] == digest(original["prompt"].encode("utf-8")) and
                generation["tokenizer_hash"] == report["tokenizer_hash"] and generation["prompt_tokens"] == prompt_tokens and
                generation["max_new_tokens"] == max_new_tokens and prompt_tokens+max_new_tokens <= context, "generation context identity changed")
        ids = generation["generated_ids"]
        require(isinstance(ids, list) and 1 <= len(ids) <= max_new_tokens and len(ids) == generation["generated_tokens"] and
                all(type(t) is int and 0 <= t < tokenizer.vocab_size for t in ids), "invalid generated IDs/count")
        specials = {tokenizer.bos_id, tokenizer.pad_id, tokenizer.eos_id}
        require(not any(t in specials for t in ids[:-1]), "generation continued after a stop token")
        invalid = next((i for i, t in enumerate(ids) if t in {tokenizer.bos_id, tokenizer.pad_id}), None)
        if invalid is not None:
            termination, error, text = "invalid_special_token", "BOS/PAD emitted inside response", None
        else:
            termination = "eos" if ids[-1] == tokenizer.eos_id else "token_budget"
            require(termination == "eos" or len(ids) == max_new_tokens, "generation stopped before budget without EOS")
            try:
                text, error = tokenizer.decode([t for t in ids if t != tokenizer.eos_id], strict=True), None
            except UnicodeError:
                termination, error, text = "invalid_utf8", "generated bytes are not complete UTF-8", None
        require((generation["termination"], generation["error"], generation["text"]) == (termination, error, text), "generated text/termination differs from IDs")
        require(row["coverage"] == {"prompt_tokens": prompt_tokens, "reference_tokens_including_eos": encoded["response_tokens"],
                "max_new_tokens": max_new_tokens, "reference_fits_budget": len(gold) <= max_new_tokens}, "coverage changed")
        require(row["first_invalid_special_token"] == invalid and row["first_reference_divergence"] == divergence(gold, ids), "generation divergence changed")
        require(row["scoring"] == score_text(probes["task"], original, text), "scoped candidate check changed")
    require(report["summary"] == summarize(report["rows"]), "diagnostic aggregate changed")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    with Run(args.runs, "language-diagnostics", {"max_new_tokens": args.max_new_tokens}, [args.checkpoint, args.probes]) as run:
        result = measure_checkpoint(args.checkpoint, read_json(args.probes), args.max_new_tokens)
        write_json(run.path/"diagnostics.json", result)
    print(run.path)


if __name__ == "__main__":
    main()
