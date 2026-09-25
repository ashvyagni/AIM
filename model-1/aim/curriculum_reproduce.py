"""One-command registered curriculum study with retained tests and stage records."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .curriculum_data import build_dataset
from .curriculum_train import train
from .curriculum_eval import evaluate
from .tracking import ROOT, Run, file_hash, write_json


def reproduce(config_path,runs):
    config=json.loads(Path(config_path).read_text())
    with Run(Path(runs),"curriculum-reproduction",config,[Path(config_path)]) as run:
        log=run.path/"tests.log"
        with log.open('x') as stream:
            tests=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=ROOT,
                                 stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        if tests.returncode or 'skipped=' in log.read_text(): raise RuntimeError("Preflight failed or skipped tests; log retained")
        write_json(run.path/"test-result.json",{"returncode":tests.returncode,"sha256":file_hash(log)})
        dataset=build_dataset(run.path/"datasets")
        write_json(run.path/"dataset-stage.json",{"path":str(dataset.resolve())})
        trained=train(config,dataset/"train-validation.json",run.path/"training")
        write_json(run.path/"training-stage.json",{"path":str(trained.resolve()),
                   "selection_sha256":file_hash(trained/"selected-models.json")})
        evaluated,audited,metrics=evaluate(trained/"selected-models.json",dataset/"holdout.json",run.path/"evaluation")
        write_json(run.path/"summary.json",{"dataset":str(dataset.resolve()),"training":str(trained.resolve()),
            "evaluation":str(evaluated.resolve()),"observation_audit":str(audited.resolve()),"metrics":metrics,
            "observation_metrics":json.loads((audited/"metrics.json").read_text())})
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=Path("configs/arithmetic-curriculum.json"))
    parser.add_argument("--runs",type=Path,default=Path("runs"));args=parser.parse_args()
    print(reproduce(args.config,args.runs))
