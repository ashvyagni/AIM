"""Bounded Phase 2A.1 reproduction with tests and retained stage records."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .comparison_data import build_dataset
from .comparison_train import train
from .comparison_eval import evaluate
from .tracking import ROOT, Run, file_hash, write_json


def reproduce(config_path,runs):
    config=json.loads(Path(config_path).read_text())
    with Run(Path(runs),"comparison-reproduction",config,[Path(config_path)]) as run:
        log=run.path/"tests.log"
        with log.open('x') as stream:
            result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                                  cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode: raise RuntimeError("Comparison preflight tests failed; log retained")
        if 'skipped=' in log.read_text(): raise RuntimeError("Full reproduction requires zero skipped tests")
        write_json(run.path/"test-result.json",{"returncode":result.returncode,"sha256":file_hash(log)})
        dataset=build_dataset(run.path/"datasets")
        write_json(run.path/"dataset-stage.json",{"path":str(dataset.resolve())})
        trained=train(config,dataset/"train-validation.json",run.path/"training")
        write_json(run.path/"training-stage.json",{"path":str(trained.resolve()),
                   "selection_sha256":file_hash(trained/"selected-models.json")})
        evaluated,report=evaluate(trained/"selected-models.json",dataset/"holdout.json",run.path/"evaluation")
        write_json(run.path/"summary.json",{"dataset":str(dataset.resolve()),"training":str(trained.resolve()),
                   "evaluation":str(evaluated.resolve()),"metrics":report})
    return run.path


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("configs/finite-difference-comparison.json"))
    parser.add_argument("--runs",type=Path,default=Path("runs"));args=parser.parse_args()
    print(reproduce(args.config,args.runs))
