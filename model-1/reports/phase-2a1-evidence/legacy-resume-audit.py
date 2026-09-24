import json
from pathlib import Path
import torch
from aim.neural import read_checkpoint
from aim.research_train import train_seed
from aim.research_data import load_training_data
from aim.tracking import ROOT, Run, file_hash, write_json

data=ROOT/'runs/20260924T143247-research-dataset-7003fc09/train-validation.json'
original=ROOT/'runs/20260924T143332-research-training-122c26fb/seeds/20260924T143332-research-sft-seed17-599c5be2'
resume=original/'checkpoint-step1200.pt';reference=original/'checkpoint-step1800.pt'
config=json.loads((ROOT/'configs/structured-researcher.json').read_text())
protocol=ROOT/'docs/experiments/phase-2a-protocol.md'

def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

with Run(ROOT/'runs','legacy-research-resume-audit',{'seed':17,'from_step':1200,'to_step':1800},[data,resume,reference,protocol]) as run:
    result=train_seed(config,load_training_data(data),data,run.path/'continuation',17,protocol,resume=resume)
    resumed=Path(result['evaluations'][-1]['checkpoint'])
    a=read_checkpoint(reference);b=read_checkpoint(resumed)
    checks={key:equal(a[key],b[key]) for key in ('model','optimizer','torch_rng','sampler_rng','step')}
    old=json.loads((original/'metrics.json').read_text())
    checks['original_initial_sha256']=old['initial']['sha256']==result['initial']['sha256']
    checks['selected_step']=old['selected']['step']==result['selected']['step']
    checks['validation_metrics']=all(old['selected'][k]==result['selected'][k] for k in ('valid_rate','prediction_agreement_rate','response_token_nll'))
    write_json(run.path/'metrics.json',{'checks':checks,'passed':all(checks.values()),'result':result,
        'reference_sha256':file_hash(reference),'resumed_sha256':file_hash(resumed),
        'note':'Compare tensors and selection outcomes, not archive bytes containing different provenance metadata.'})
    if not all(checks.values()): raise AssertionError('Legacy continuation differs; retained audit result')
print(run.path)
