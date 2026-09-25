"""Complete paid-evidence episodes with frozen historical Judges and retained traces."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .contracts import ContractError
from .judge_shift_train import ShiftJudge
from .paid_evidence import VERSION, PaidEvidenceController, SharedPolicy
from .paid_evidence_data import build_dataset
from .paid_evidence_eval import summarize
from .tracking import ROOT, Run, file_hash, write_json


def load_forecasters(evidence_path,seeds):
    data=json.loads(Path(evidence_path).read_text())
    forecasters={};records={}
    for seed in seeds:
        row=data['frozen']['models'][f'rich-log-seed{seed}']
        if row['mode']!='rich' or row['loss']!='log' or row['seed']!=seed: raise ContractError('Wrong frozen Judge')
        checkpoint=Path(row['checkpoint'])
        if not checkpoint.is_absolute(): checkpoint=ROOT.parent/checkpoint
        if file_hash(checkpoint)!=row['sha256']: raise ContractError('Frozen Judge changed')
        judge=ShiftJudge(checkpoint,calibrated=True)
        if judge.temperature!=row['temperature']: raise ContractError('Frozen temperature mismatch')
        forecasters[seed]=judge;records[str(seed)]={**row,'checkpoint':str(checkpoint.resolve())}
    return forecasters,records


def policy_specs(seeds):
    return [(name,name,None) for name in ('verify_all','abstain_all','fit_direct','acquire_fit')]+[
        (f'{kind}-seed{seed}',kind,seed) for kind in ('judge_direct','acquire_judge','selective_judge') for seed in seeds]


def evaluate(dataset,config,runs,forecasters,records,evidence_path):
    data=json.loads(Path(dataset).read_text())
    if data['version']!=VERSION: raise ContractError('Wrong acquisition dataset')
    with Run(Path(runs),'paid-evidence-evaluation',config,inputs=[dataset,evidence_path,ROOT/'docs/experiments/phase-2b1-protocol.md']) as run:
        specs=policy_specs(config['seeds'])
        write_json(run.path/'frozen.json',{'version':VERSION,'configuration':config,'judges':records,
            'judge_evidence_sha256':file_hash(evidence_path),'dataset_sha256':file_hash(dataset),'policies':specs})
        episodes={}
        for name,kind,seed in specs:
            rows=[]
            for row in data['rows']:
                policy=SharedPolicy(kind,config,forecasters.get(seed))
                with Run(run.path/'cases',name,{'policy':name,'world':row['id'],'dataset_sha256':file_hash(dataset)}) as case_run:
                    _,result=PaidEvidenceController(policy,max_actions=config['max_actions']).run(row['case'],case_run)
                rows.append({k:row[k] for k in ('id','group','family')}|{
                    'run':str(case_run.path.resolve()),'summary':result,'memory_sha256':file_hash(case_run.path/'memory/memory.sqlite')})
            episodes[name]=rows
            write_json(run.path/(name+'-episodes.json'),rows)
            print(json.dumps({'policy':name,'episodes':len(rows),'successes':sum(r['summary']['success'] for r in rows)}),flush=True)
        result=summarize(episodes,config)
        result['unbacked_verified_claims']=sum(r['summary']['verified_claims']-r['summary']['verified_with_passing_checks']
            for rows in episodes.values() for r in rows)
        result['evidence_gate_passed']=result['utility_gate_passed'] and result['unbacked_verified_claims']==0
        write_json(run.path/'metrics.json',result)
        write_json(run.path/'episode-index.json',{name:str((run.path/(name+'-episodes.json')).resolve()) for name in episodes})
    return run.path


def reproduce(config_path,runs):
    config=json.loads(Path(config_path).read_text())
    if config['version']!=VERSION: raise ContractError('Wrong experiment configuration')
    import torch
    torch.set_num_threads(1)
    evidence=ROOT/config['judge_evidence']
    with Run(Path(runs),'paid-evidence-reproduction',config,inputs=[config_path,evidence,ROOT/'docs/experiments/phase-2b1-protocol.md']) as run:
        forecasters,records=load_forecasters(evidence,config['seeds'])
        with (run.path/'tests.log').open('x') as log:
            completed=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=180)
        log=(run.path/'tests.log').read_text();write_json(run.path/'test-result.json',{'returncode':completed.returncode})
        if completed.returncode or 'skipped=' in log or not re.search(r'Ran \d+ tests',log):
            raise RuntimeError('Full passing preflight without skips required')
        dataset=build_dataset(run.path/'data')
        write_json(run.path/'dataset-stage.json',{'path':str(dataset.resolve())})
        evaluation=evaluate(dataset/'worlds.json',config,run.path/'evaluation',forecasters,records,evidence)
        write_json(run.path/'summary.json',{'dataset':str(dataset.resolve()),'evaluation':str(evaluation.resolve()),
            'results':json.loads((evaluation/'metrics.json').read_text())})
    return run.path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'configs/paid-evidence.json')
    parser.add_argument('--runs',type=Path,default=Path('runs'));args=parser.parse_args()
    print(reproduce(args.config,args.runs))
