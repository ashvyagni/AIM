"""Bounded full Judge experiment; never modifies historical runs or defaults."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .judge_shift_data import build_dataset
from .judge_shift_eval import evaluate
from .judge_shift_train import train
from .tracking import ROOT, Run, file_hash, write_json


def reproduce(config_path, runs):
    config=json.loads(Path(config_path).read_text())
    with Run(Path(runs),'judge-shift-reproduction',config,inputs=[config_path,ROOT/'docs/experiments/phase-2b-protocol.md']) as run:
        with (run.path/'tests.log').open('x') as log:
            test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=180)
        text=(run.path/'tests.log').read_text()
        write_json(run.path/'test-result.json',{'returncode':test.returncode})
        if test.returncode or 'skipped=' in text or not re.search(r'Ran \d+ tests',text):
            raise RuntimeError('Complete passing test suite required')
        dataset=build_dataset(run.path/'data')
        write_json(run.path/'dataset-stage.json',{'path':str(dataset.resolve())})
        trained=train(dataset/'train-calibration.json',config,run.path/'training')
        write_json(run.path/'training-stage.json',{'path':str(trained.resolve()),'frozen_sha256':file_hash(trained/'frozen.json')})
        evaluated=evaluate(trained/'frozen.json',dataset/'holdout.json',run.path/'evaluation')
        summary={'dataset':str(dataset.resolve()),'training':str(trained.resolve()),'evaluation':str(evaluated.resolve()),
                 'results':json.loads((evaluated/'metrics.json').read_text())}
        write_json(run.path/'summary.json',summary)
    return run.path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/judge-shift.json')
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    args=parser.parse_args()
    print(reproduce(args.config,args.runs))
