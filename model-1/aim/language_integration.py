"""Actual Controller, distributed-initialization and interruption audit scenarios."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .contracts import Evidence, Outcome, Status
from .corpus import read_json, require
from .language_training import train
from .memory import Memory
from .tracking import ROOT, Run, canonical, file_hash, write_json


def audit_numerical(path):
    from .controller import Controller
    path = Path(path)
    state = read_json(path/"state.json")
    memory = Memory(path/"memory", read_only=True)
    try:
        require(canonical(memory.replay()) == canonical(state), "numerical ledger replay mismatch")
        typed = SimpleNamespace(evidence=[Evidence(**e) for e in state["evidence"]],
            claims=[SimpleNamespace(**{**c, "status": Status(c["status"])}) for c in state["claims"]],
            verifications=[SimpleNamespace(**{**v, "outcome": Outcome(v["outcome"])}) for v in state["verifications"]])
        Controller._validate_final(typed, memory)
        return {"claims": len(state["claims"]), "verified": sum(c["status"] == "VERIFIED" for c in state["claims"]),
                "replay_and_checks": "PASS"}
    finally:
        memory.close()


def controller_cases(parent, tokenizer_arms):
    from .controller import Controller
    from .language_contract import read_source
    from .research_data import build_dataset
    from .researcher import PolynomialResearcher, TransformerResearcher
    from .symbolic_data import symbolic_case
    from .symbolic_loop import BinomialResearcher, SymbolicController, SymbolicTransformerResearcher, audit_symbolic_run
    # Existing exposed worlds: integration checks only, not a fresh capability study.
    dataset = build_dataset(parent.path/"structured-data")
    original = read_json(dataset/"train-validation.json")
    selected = {**original, "train": original["train"][:8], "validation": original["validation"][:4]}
    subset = parent.path/"structured-train-validation.json"
    write_json(subset, selected)
    numerical_cases = [r["case"] for r in read_json(dataset/"holdout.json")["test"][:2]]
    symbolic_cases = [symbolic_case(f"(x+{i})**2", f"integration-{i}") for i in (20, 21)]
    records, models = [], {}

    def execute(domain, backend, researcher, cases, checkpoint=None):
        for index, case in enumerate(cases):
            with Run(parent.path/"controllers", f"language-{domain}-{backend}",
                     {"backend": backend, "domain": domain, "case_index": index, "scope": "exposed engineering cases"},
                     [checkpoint] if checkpoint else []) as child:
                if domain == "numerical":
                    state = Controller(researcher=researcher, max_actions=8).run(case, child)
                    audited = audit_numerical(child.path)
                else:
                    state = SymbolicController(researcher=researcher).run(case, child)
                    audited = audit_symbolic_run(child.path)
                trace = getattr(researcher, "last_trace", None)
                if domain == "numerical" and trace is not None:
                    write_json(child.path/"researcher-output.json", trace)
                generation = trace.get("generation") if trace else None
                records.append({"domain": domain, "backend": backend, "case_index": index, "run": str(child.path.relative_to(parent.path)),
                                "audit": audited, "hypotheses": len(state.hypotheses), "unknowns": state.unknowns,
                                "generation": generation, "trace": trace,
                                "state_sha256": file_hash(child.path/"state.json")})

    execute("numerical", "reference", PolynomialResearcher(), numerical_cases)
    execute("symbolic", "reference", BinomialResearcher(), symbolic_cases)
    for name, tokenizer in tokenizer_arms.items():
        models[name] = {}
        for domain, task, context in (("numerical", "structured", 512), ("symbolic", "symbolic", 256)):
            config = {**read_json(ROOT/"configs/language-sft.json"), "task": task, "steps": 2}
            config["model"] = {**config["model"], "context": context}
            if domain == "numerical": config["dataset_path"] = str(subset)
            trained = train(config, parent.path/"adapter-training", tokenizer)
            checkpoint = trained/"checkpoint.pt"
            record = read_source(checkpoint)
            require(record["tokenizer"] == tokenizer.specification(), "adapter tokenizer changed")
            models[name][domain] = {"path": str(trained.relative_to(parent.path)), "checkpoint_sha256": file_hash(checkpoint),
                                    "metrics": read_json(trained/"metrics.json")}
            researcher = TransformerResearcher(checkpoint, max_new_tokens=96) if domain == "numerical" else SymbolicTransformerResearcher(checkpoint, max_new_tokens=96)
            execute(domain, name, researcher, numerical_cases if domain == "numerical" else symbolic_cases, checkpoint)
    require(len(records) == 12, "controller integration case count changed")
    require(all(r["audit"]["verified"] >= 1 for r in records if r["backend"] == "reference"), "positive reference control did not verify")
    return {"models": models, "cases": records, "scope": "2 cases per domain/backend; no neural success threshold or model promotion"}


def recovery_drills(parent, arms):
    from .language_reproduce import compare
    from .training import save_checkpoint
    results = {}
    for stage in ("sft", "rlvr"):
        config = {**read_json(ROOT/f"configs/language-{stage}.json"), "steps": 4}
        base = arms["bpe"]
        source = parent.path/(base["pretraining"] if stage == "sft" else base["stages"]["preference"]["full"])/"checkpoint.pt"
        group = parent.path/("interrupted-"+stage)
        def interrupt(run, record, filename="checkpoint.pt"):
            save_checkpoint(run, record, filename)
            if filename == "checkpoint-step-000002.pt":
                raise RuntimeError("INJECTED_LANGUAGE_INTERRUPTION after completed checkpoint")
        try:
            with patch("aim.language_training.save_checkpoint", side_effect=interrupt):
                train(config, group, initialize=source)
        except RuntimeError as exc:
            require("INJECTED_LANGUAGE_INTERRUPTION" in str(exc), "unexpected recovery failure")
        else:
            raise RuntimeError("interruption did not execute")
        failed = next(group.iterdir())
        require(read_json(failed/"status.json")["status"] == "FAILED", "interrupted training not retained as failed")
        resumed = train(config, parent.path/"recovery", resume=failed/"checkpoint-step-000002.pt")
        full = parent.path/base["stages"][stage]["full"]/"checkpoint.pt"
        results[stage] = {"failed_path": str(failed.relative_to(parent.path)), "status": read_json(failed/"status.json"),
                          "failure_sha256": file_hash(failed/"failure.txt"), "resumed": str(resumed.relative_to(parent.path)),
                          "comparison": compare(full, resumed/"checkpoint.pt")}
    return results


def distributed_bridge(parent, corpus, tokenizer):
    from .distributed_jobs import launch, export_initialization, audit_job
    from .language_contract import read_source
    config = read_json(ROOT/"configs/distributed-pretrain-smoke.json")
    supervised = {**read_json(ROOT/"configs/language-sft.json"), "steps": 2}
    config.update(steps=2, model=supervised["model"])
    job = launch(config, corpus.path, parent.path/"distributed-bridge", tokenizer=tokenizer)
    initialization = export_initialization(job/"job/checkpoints/step-000002", parent.path/"bridge-exports")
    trained = train(supervised, parent.path/"bridge-sft", initialize=initialization)
    record = read_source(trained/"checkpoint.pt")
    require(record["tokenizer"] == tokenizer.specification() and record["parent_sha256"] == file_hash(initialization), "distributed BPE SFT lineage changed")
    return {"job": str(job.relative_to(parent.path)), "distributed_audit": audit_job(job, corpus.path),
            "initialization": str(initialization.relative_to(parent.path)), "sft": str(trained.relative_to(parent.path)),
            "checkpoint_sha256": file_hash(trained/"checkpoint.pt"), "metrics": read_json(trained/"metrics.json")}


def run_integration(parent, corpus, arms):
    from .tokenization import tokenizer_from_spec
    tokenizers = {name: tokenizer_from_spec(arm["tokenizer"]) for name, arm in arms.items()}
    controllers = controller_cases(parent, tokenizers)
    print("Twelve actual Controller episodes completed and replayed", flush=True)
    recovery = recovery_drills(parent, arms)
    print("Retained SFT and stochastic RLVR interruptions recovered exactly", flush=True)
    bridge = distributed_bridge(parent, corpus, tokenizers["bpe"])
    print("Two-worker BPE initialization export entered versioned SFT", flush=True)
    return {"controllers": controllers, "recovery": recovery, "distributed_bridge": bridge}


def audit_integration(path, results, arms, corpus_path):
    from .distributed_jobs import audit_job
    from .language_reproduce import compare
    from .symbolic_loop import audit_symbolic_run
    path = Path(path)
    for row in results["controllers"]["cases"]:
        child = path/row["run"]
        require(file_hash(child/"state.json") == row["state_sha256"], "Controller state changed")
        audited = audit_numerical(child) if row["domain"] == "numerical" else audit_symbolic_run(child)
        require(audited == row["audit"], "Controller replay changed")
        trace_path = child/"researcher-output.json"
        if row["trace"] is not None:
            require(read_json(trace_path) == row["trace"], "generation trace changed")
    for variants in results["controllers"]["models"].values():
        for model in variants.values():
            require(file_hash(path/model["path"]/"checkpoint.pt") == model["checkpoint_sha256"], "adapter checkpoint changed")
    for stage, drill in results["recovery"].items():
        require(file_hash(path/drill["failed_path"]/"failure.txt") == drill["failure_sha256"], "failed recovery evidence changed")
        require(compare(path/arms["bpe"]["stages"][stage]["full"]/"checkpoint.pt", path/drill["resumed"]/"checkpoint.pt") == drill["comparison"], "recovery comparison changed")
    bridge = results["distributed_bridge"]
    require(audit_job(path/bridge["job"], corpus_path) == bridge["distributed_audit"], "distributed bridge replay changed")
    require(file_hash(path/bridge["sft"]/"checkpoint.pt") == bridge["checkpoint_sha256"], "bridge checkpoint changed")
    return {"controller_cases": len(results["controllers"]["cases"]), "recovery_drills": len(results["recovery"]), "bridge_replayed": True}
