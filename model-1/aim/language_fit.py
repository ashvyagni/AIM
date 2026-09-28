"""Reproducible, bounded SFT fitting study; no test/OOD model selection."""
import argparse
from itertools import product
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .corpus import Corpus, intake, read_json, require
from .language_diagnostics import audit_measurement, freeze_probes, measure, measure_checkpoint, score_text, validate_probes
from .tracking import ROOT, Run, digest, file_hash, write_json


def validate_config(config):
    require(isinstance(config, dict) and set(config) == {"schema", "seeds", "tokenizers", "steps", "probe_count", "max_new_tokens", "training"},
            "fitting configuration fields differ")
    require(config["schema"] == "aim-language-fit-study-v1", "unsupported fitting study")
    seeds, steps, tokenizers = config["seeds"], config["steps"], config["tokenizers"]
    require(isinstance(seeds, list) and 1 <= len(seeds) <= 3 and all(type(x) is int and 0 <= x < 2**32 for x in seeds)
            and len(set(seeds)) == len(seeds), "invalid bounded fitting seeds")
    require(isinstance(steps, list) and 1 <= len(steps) <= 4 and all(type(x) is int and 1 <= x <= 1000 for x in steps)
            and sorted(set(steps)) == steps, "steps must be increasing unique positive budgets")
    require(isinstance(tokenizers, list) and tokenizers and all(x in {"byte", "bpe32"} for x in tokenizers)
            and len(set(tokenizers)) == len(tokenizers), "unsupported/duplicate fitting tokenizer")
    require(type(config["probe_count"]) is int and 1 <= config["probe_count"] <= 4, "study supports 1..4 probes per split")
    require(type(config["max_new_tokens"]) is int and 1 <= config["max_new_tokens"] <= 128, "study generation budget exceeds 128")
    from .language_training import validate_config as training_config
    require(set(config["training"]) == {"batch_size", "learning_rate", "weight_decay", "checkpoint_every", "eval_batch_size", "model"},
            "explicit SFT training fields required")
    training_config({**config["training"], "stage": "sft", "task": "symbolic", "seed": seeds[0], "steps": steps[-1]})


def structured_fixture(run):
    """Explicit fitting worlds only. Do not construct or open a holdout artifact."""
    from .memory import Memory
    from .research_data import make_record, world_id
    from .research_format import VERSION
    coefficients = {"train": [(0, 1, 0), (1, 2, 0), (-1, -2, 0), (2, 0, 1), (0, -1, 1), (-2, 2, -1), (3, -3, 2), (1, 1, -2)],
                    "validation": [(2, 1, 0), (-3, -1, 0), (1, 0, 2), (-1, 2, -2)]}
    memory = Memory(run.path/"fitting-source-memory")
    try:
        rows = {s: [make_record(world_id(c), c, "linear" if c[2] == 0 else "quadratic", s, memory) for c in cs]
                for s, cs in coefficients.items()}
    finally:
        memory.close()
    path = run.path/"structured-train-validation.json"
    write_json(path, {"schema": "aim-research-sft-v1", "version": VERSION, **rows})
    return path


def controller_probes(parent, task, backend, probes, checkpoint=None, budget=128):
    from .controller import Controller
    from .language_integration import audit_numerical
    from .researcher import PolynomialResearcher, TransformerResearcher
    from .symbolic_data import symbolic_case
    from .symbolic_loop import BinomialResearcher, SymbolicController, SymbolicTransformerResearcher, audit_symbolic_run
    researcher = (TransformerResearcher(checkpoint, budget) if checkpoint else PolynomialResearcher()) if task == "structured" else (
        SymbolicTransformerResearcher(checkpoint, budget) if checkpoint else BinomialResearcher())
    results = []
    for split in ("train", "validation"):
        for row in [r for r in probes["rows"] if r["split"] == split][:2]:
            case = row["case"] if task == "structured" else symbolic_case(row["lhs"], row["id"])
            with Run(parent.path/"controllers", "fitting-"+task, {"backend": backend, "id": row["id"], "split": split,
                     "scope": "exposed fitting probes"}, [checkpoint] if checkpoint else []) as child:
                if task == "structured":
                    state = Controller(researcher=researcher, max_actions=8).run(case, child)
                    audit = audit_numerical(child.path)
                else:
                    state = SymbolicController(researcher=researcher).run(case, child)
                    audit = audit_symbolic_run(child.path)
                trace = getattr(researcher, "last_trace", None)
                if task == "structured" and trace is not None:
                    write_json(child.path/"researcher-output.json", trace)
                results.append({"task": task, "backend": backend, "id": row["id"], "split": split,
                    "run": str(child.path.relative_to(parent.path)), "audit": audit, "state_sha256": file_hash(child.path/"state.json"),
                    "hypotheses": len(state.hypotheses), "unknowns": state.unknowns,
                    "trace_sha256": file_hash(child.path/"researcher-output.json") if trace is not None else None})
    if checkpoint is None:
        require(all(r["audit"]["verified"] >= 1 for r in results), "positive reference Controller failed")
    return results


def run_study(config, runs, regression=False):
    from .checkpoint_bundle import tree_hash
    from .corpus_fixture import create_fixture
    from .language_contract import model_config, read_source
    from .language_data import prepare
    from .language_training import train
    from .neural import CausalLM
    from .tokenization import ByteTokenizer, fit_bpe
    from .training import seed_all
    with Run(Path(runs), "phase-3e-language-fitting", {"study": config, "regression": regression}) as run:
        validate_config(config)
        if regression:
            tested = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (run.path/"tests.log").write_text(tested.stdout)
            require(tested.returncode == 0 and re.search(r"OK \(skipped=", tested.stdout) is None, "regression failed/skipped; log retained")
            print("Full regression passed", flush=True)
        data_path = structured_fixture(run)
        data = {"structured": prepare("structured", "sft", data_path), "symbolic": prepare("symbolic", "sft")}
        probes = {task: freeze_probes(values, config["probe_count"]) for task, values in data.items()}
        corpus = Corpus(intake(create_fixture(run.path/"tokenizer-fixture"), run.path/"intake"))
        tokenizers = {name: ByteTokenizer() if name == "byte" else fit_bpe(corpus, 32) for name in config["tokenizers"]}
        for task, values in data.items():
            write_json(run.path/(task+"-prepared.json"), values)
            write_json(run.path/(task+"-probes.json"), probes[task])
            require(all(score_text(task, row, row["response"])["check"]["outcome"] == "PASS" for row in probes[task]["rows"]),
                    "reference diagnostic responses must pass their scoped checker")
        for name, tokenizer in tokenizers.items():
            write_json(run.path/(name+"-tokenizer.json"), tokenizer.specification())
        arms = []
        controls = [c for task in data for c in controller_probes(run, task, "reference", probes[task])]
        for task, name, seed in product(data, tokenizers, config["seeds"]):
            tokenizer = tokenizers[name]
            base = {**config["training"], "stage": "sft", "task": task, "seed": seed}
            if task == "structured":
                base["dataset_path"] = str(data_path)
            seed_all(seed)
            initial = CausalLM(model_config(base["model"], tokenizer))
            arm = {"task": task, "tokenizer": name, "seed": seed, "parameters": initial.cfg.parameter_estimate(),
                   "probe_hash": probes[task]["probe_hash"], "dataset_hash": data[task]["dataset_hash"], "measurements": []}
            folder = run.path/"diagnostics"/f"{task}-{name}-{seed}"
            folder.mkdir(parents=True)
            first = measure(initial, tokenizer, probes[task], config["max_new_tokens"])
            write_json(folder/"step-000000.json", first)
            arm["measurements"].append({"step": 0, "checkpoint": None, "path": str((folder/"step-000000.json").relative_to(run.path)),
                                        "sha256": file_hash(folder/"step-000000.json"), "summary": first["summary"]})
            parent = None
            for steps in config["steps"]:
                trained = train({**base, "steps": steps}, run.path/"training", tokenizer, resume=parent)
                parent = trained/"checkpoint.pt"
                record = read_source(parent)
                require(tree_hash(record["reference"]) == first["model_state_hash"], "fitting chain lost its initial frozen reference")
                measured = measure_checkpoint(parent, probes[task], config["max_new_tokens"])
                path = folder/f"step-{steps:06d}.json"
                write_json(path, measured)
                arm["measurements"].append({"step": steps, "checkpoint": str(parent.relative_to(run.path)),
                    "checkpoint_sha256": file_hash(parent), "path": str(path.relative_to(run.path)), "sha256": file_hash(path),
                    "summary": measured["summary"], "scored_tokens": record["scored_tokens"]})
                print(f"{task}/{name}/seed{seed}/step{steps}: "+str({s: measured["summary"][s]["scoped_check_pass"] for s in ("train", "validation")}), flush=True)
            arm["controllers"] = controller_probes(run, task, f"{name}-seed{seed}", probes[task], parent, config["max_new_tokens"])
            arms.append(arm)
        write_json(run.path/"results.json", {"schema": "aim-language-fit-results-v1", "config": config, "arms": arms, "controls": controls,
                   "scope": "predeclared fitting diagnostics, no test/OOD cases; no checkpoint selection or promotion",
                   "tokenizer_note": "BPE32 fitted on existing authored corpus training split, not supervised validation probes; no tokenizer superiority study"})
    return run.path


def contained(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "artifact path must be relative")
    path = root/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "artifact escaped run or is missing")
    return path


def audit_study(path):
    """Read-only artifact/checker/replay audit, not independent neural inference."""
    from .checkpoint_bundle import tree_hash
    from .language_contract import read_source, validate_record
    from .language_integration import audit_numerical
    from .neural import CausalLM
    from .language_contract import model_config
    from .symbolic_loop import audit_symbolic_run
    from .tokenization import tokenizer_from_spec
    from .training import seed_all
    path = Path(path)
    require(read_json(path/"status.json")["status"] == "COMPLETED", "only completed fitting runs can be audited")
    results = read_json(path/"results.json")
    config = results["config"]
    validate_config(config)
    manifest = read_json(path/"manifest.json")
    require(config == manifest["configuration"]["study"], "study configuration differs from run manifest")
    expected = set(product(("structured", "symbolic"), config["tokenizers"], config["seeds"]))
    require(len(results["arms"]) == len(expected) and {(a["task"], a["tokenizer"], a["seed"]) for a in results["arms"]} == expected,
            "fitting arms differ from plan")
    measurements, cases = 0, list(results["controls"])
    expected_controls = set()
    for task in ("structured", "symbolic"):
        probes = read_json(path/(task+"-probes.json"))
        data = read_json(path/(task+"-prepared.json"), 10*1024*1024)
        require(freeze_probes(data, config["probe_count"]) == probes, "saved probes differ from prepared fitting data")
        for split in ("train", "validation"):
            for row in [r for r in probes["rows"] if r["split"] == split][:2]:
                expected_controls.add((task, "reference", row["id"], split))
    require(len(results["controls"]) == len(expected_controls) and
            {(c["task"], c["backend"], c["id"], c["split"]) for c in results["controls"]} == expected_controls,
            "positive control cases differ from plan")
    for arm in results["arms"]:
        probes = read_json(path/(arm["task"]+"-probes.json"))
        validate_probes(probes)
        require(probes["probe_hash"] == arm["probe_hash"] and probes["dataset_hash"] == arm["dataset_hash"], "arm probe/data identity changed")
        tokenizer = tokenizer_from_spec(read_json(path/(arm["tokenizer"]+"-tokenizer.json")))
        require([m["step"] for m in arm["measurements"]] == [0]+config["steps"], "fitting budget rows changed")
        for item in arm["measurements"]:
            report_path = contained(path, item["path"])
            require(file_hash(report_path) == item["sha256"], "diagnostic artifact changed")
            report = read_json(report_path)
            require(report["probe_hash"] == probes["probe_hash"] and report["dataset_hash"] == probes["dataset_hash"] and
                    report["tokenizer_hash"] == digest(tokenizer.specification()), "diagnostic identities differ")
            if item["step"]:
                checkpoint = contained(path, item["checkpoint"])
                require(file_hash(checkpoint) == item["checkpoint_sha256"] == report["checkpoint_sha256"], "checkpoint identity changed")
                source = read_source(checkpoint)
                validate_record(source)
                require(tree_hash(source["model"]) == report["model_state_hash"] and source["step"] == report["step"] == item["step"] and
                        source["dataset_hash"] == probes["dataset_hash"] and source["task"] == arm["task"] and
                        source["tokenizer"] == tokenizer.specification() and source["stable_config"]["seed"] == arm["seed"] and
                        tree_hash(source["reference"]) == initial_hash, "checkpoint model/data/step/reference changed")
            else:
                seed_all(arm["seed"])
                initial = CausalLM(model_config(config["training"]["model"], tokenizer))
                require(tree_hash(initial.state_dict()) == report["model_state_hash"], "random baseline initialization changed")
                initial_hash = report["model_state_hash"]
            audit_measurement(report, probes, tokenizer, config["training"]["model"]["context"], config["max_new_tokens"])
            require(report["summary"] == item["summary"], "diagnostic aggregate changed")
            measurements += 1
        expected_cases = {(task, f'{arm["tokenizer"]}-seed{arm["seed"]}', identifier, split)
                          for task, _, identifier, split in expected_controls if task == arm["task"]}
        require(len(arm["controllers"]) == len(expected_cases) and
                {(c["task"], c["backend"], c["id"], c["split"]) for c in arm["controllers"]} == expected_cases,
                "neural Controller cases differ from plan")
        final_rows = {row["id"]: row for row in report["rows"]}
        for case in arm["controllers"]:
            trace = read_json(contained(path, case["run"]+"/researcher-output.json"))
            require(trace.get("generation") == final_rows[case["id"]]["generation"], "Controller and diagnostic generation differ")
        cases += arm["controllers"]
    for case in cases:
        state_path = contained(path, case["run"]+"/state.json")
        require(file_hash(state_path) == case["state_sha256"], "Controller state changed")
        audited = audit_numerical(state_path.parent) if case["task"] == "structured" else audit_symbolic_run(state_path.parent)
        require(audited == case["audit"], "Controller replay differs")
        if case["trace_sha256"] is not None:
            require(file_hash(contained(path, case["run"]+"/researcher-output.json")) == case["trace_sha256"], "Controller trace changed")
    return {"arms": len(expected), "measurements": measurements, "controller_replays": len(cases),
            "scope": "artifact identities, deterministic initialization, candidate checker and ledger replay; no independent neural likelihood recomputation"}


def export(path, destination):
    path, destination = Path(path), Path(destination)
    audit = audit_study(path)
    results = read_json(path/"results.json")
    destination.mkdir(parents=True, exist_ok=False)
    files = ["results.json", "structured-probes.json", "symbolic-probes.json"]+[n+"-tokenizer.json" for n in results["config"]["tokenizers"]]
    files += [m["path"] for arm in results["arms"] for m in arm["measurements"]]
    for name in files:
        (destination/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(path, name), destination/name)
    redactions = None
    if (path/"tests.log").exists():
        from .pretrain_reproduce import portable_traceback
        log = portable_traceback((path/"tests.log").read_text())
        log = re.sub(r'(?:/private)?/var/folders/[^/]+/[^/]+/T/(tmp[^/\s"\']+)', r'<temporary>/\1', log)
        (destination/"tests.log").write_text(log)
        files.append("tests.log")
        redactions = {"original_sha256": file_hash(path/"tests.log"), "scope": "only local/runtime path prefixes; original retained"}
    manifest = read_json(path/"manifest.json")
    write_json(destination/"manifest.json", {"run_id": path.name, "git_commit": manifest["environment"]["git_commit"],
        "code_hash": manifest["code_hash"], "code_hashes": manifest["code_hashes"], "configuration": manifest["configuration"],
        "environment": {k: manifest["environment"][k] for k in ("python", "platform", "machine", "logical_cpus", "packages")},
        "status": read_json(path/"status.json"), "audit": audit, "log_redactions": redactions,
        "files": {name: file_hash(destination/name) for name in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/language-fitting.json")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    path = run_study(read_json(args.config), args.runs, args.regression)
    if args.export:
        export(path, args.export)
    print(path)


if __name__ == "__main__":
    main()
