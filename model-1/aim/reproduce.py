"""One command to produce a fresh, retained miniature engineering evidence bundle."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .tracking import ROOT, Run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    import torch  # Fail explicitly when the optional training environment is missing.
    import numpy
    from .training import train
    from .evaluation import evaluate
    from .controller import Controller
    from .judge import CalibratedJudge
    from .researcher import TransformerResearcher
    from .benchmark import benchmark
    with Run(args.runs, "reproduction", {"protocol":"miniature-phase-1-v1","seed":17}) as bundle:
        tests = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                               cwd=ROOT,text=True,capture_output=True,timeout=180)
        (bundle.path/"tests.log").write_text(tests.stdout+tests.stderr)
        write_json(bundle.path/"tests.json", {"exit_code":tests.returncode,"command":"python -m unittest discover -s tests -v"})
        if tests.returncode:
            raise RuntimeError("Test suite failed; retained tests.log contains details")
        paths = {}
        experiment_dir = bundle.path/"experiments"
        for stage in ("sft","preference","rlvr","judge"):
            cfg = json.loads((ROOT/f"configs/{stage}.json").read_text())
            parent = paths.get("sft") if stage=="preference" else paths.get("preference") if stage=="rlvr" else None
            paths[stage] = train(cfg,experiment_dir,initialize=parent/"checkpoint.pt" if parent else None)
        paths["rlvr_from_sft"] = train(json.loads((ROOT/"configs/rlvr.json").read_text()),experiment_dir,
                                      initialize=paths["sft"]/"checkpoint.pt")
        for label, checkpoint in (("rule_judge",None),("trained_judge",paths["judge"]/"checkpoint.pt")):
            path, metrics = evaluate(ROOT/"eval/suite-v1.json", experiment_dir, checkpoint)
            paths["evaluation_"+label] = path
            if metrics["passed"] != metrics["total"]:
                raise RuntimeError(f"Regression cases failed for {label}; see {path}")
        case = json.loads((ROOT/"data/demo.json").read_text())
        with Run(experiment_dir,"integrated-loop", {"case":"demo","judge_checkpoint":str(paths["judge"])},
                 [ROOT/"data/demo.json",paths["judge"]/"checkpoint.pt"]) as loop:
            Controller(judge=CalibratedJudge(paths["judge"]/"checkpoint.pt")).run(case,loop)
        paths["integrated_loop"] = loop.path
        with Run(experiment_dir,"neural-researcher-diagnostic", {"expected":"unresolved; arithmetic SFT is not structured research training"},
                 [paths["sft"]/"checkpoint.pt"]) as diagnostic:
            state = Controller(researcher=TransformerResearcher(paths["sft"]/"checkpoint.pt",max_new_tokens=80)).run(case,diagnostic)
        paths["neural_researcher_diagnostic"] = diagnostic.path
        # This is a recorded diagnostic, not a success assertion or synthetic fallback.
        write_json(bundle.path/"neural-diagnostic.json", {"claims":len(state.claims),"unknowns":state.unknowns,
                    "note":"A valid response, if any, still needs independent verification; no hidden reference-researcher fallback"})
        paths["benchmark"], _ = benchmark(json.loads((ROOT/"configs/benchmark.json").read_text()),experiment_dir)
        write_json(bundle.path/"index.json", {k:str(v.resolve()) for k,v in paths.items()})
        write_json(bundle.path/"summary.json", {"status":"engineering checks completed","experiments":len(paths),
                    "limitations":["single seed miniature fixtures","synthetic preferences","reference researcher for working loop",
                                   "Judge fails under distribution shift","no VIT measurements","no integrated policy training"]})
    print(bundle.path)


if __name__ == "__main__":
    main()
