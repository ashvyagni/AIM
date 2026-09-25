"""Separate log/Brier Judge fits; holdout is never opened by this module."""
import json
from pathlib import Path

import torch
from torch import nn

from .contracts import ContractError, Decision
from .judge import TARGET
from .judge_shift_features import FEATURE_VERSION, VERSION, extract, payload
from .neural import read_checkpoint
from .objectives import proper_loss
from .tracking import ROOT, Run, digest, file_hash, write_json


class ShiftNetwork(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in {"five","rich"}: raise ContractError("Unknown feature mode")
        self.layers = nn.Sequential(nn.Linear(5 if mode=="five" else 7,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,x):
        return self.layers(x).squeeze(-1)


def validate_rows(rows, split):
    allowed = {"id","group","world","split","family","observations_count","input","features","label",
               "label_origin","source","verification_id"}
    if not isinstance(rows,list) or not rows: raise ContractError("Empty Judge split")
    seen = set()
    for row in rows:
        if set(row)!=allowed or row["split"]!=split or row["id"] in seen:
            raise ContractError("Judge row schema, split or identity mismatch")
        seen.add(row["id"])
        if type(row["label"]) is not int or row["label"] not in (0,1) or row["label_origin"]!="programmatic-verifier":
            raise ContractError("Invalid resolved label")
        if row["observations_count"] != len(row["input"]["observations"]):
            raise ContractError("Observation count mismatch")
        if row["features"]!={m:extract(row["input"],m) for m in ("five","rich")}:
            raise ContractError("Judge feature mismatch")
    return rows


def load_training(path):
    record=json.loads(Path(path).read_text())
    if set(record)!={"version","train","calibration"} or record["version"]!=VERSION:
        raise ContractError("Only training and calibration data are allowed")
    for split in ("train","calibration"): validate_rows(record[split],split)
    if {r['group'] for r in record['train']} & {r['group'] for r in record['calibration']}:
        raise ContractError("Grouped split leakage")
    return record


def train(data_path, config, runs):
    if config['version']!=VERSION or config['hidden']!=32 or not 1<=config['steps']<=600:
        raise ContractError("Unsupported Judge experiment configuration")
    torch.set_num_threads(1)
    data=load_training(data_path)
    metadata=json.loads(Path(data_path).with_name('manifest-data.json').read_text())
    if (set(metadata)!={'version','counts','group_counts','train_sha256','holdout_sha256'}
        or metadata['version']!=VERSION or metadata['train_sha256']!=file_hash(data_path)):
        raise ContractError('Dataset metadata mismatch')
    with Run(Path(runs),"judge-shift-training",config,inputs=[data_path,ROOT/'docs/experiments/phase-2b-protocol.md']) as run:
        models={}
        for mode in config['features']:
            for loss in config['losses']:
                for seed in config['seeds']:
                    name=f'{mode}-{loss}-seed{seed}'
                    with Run(run.path/name,"judge-fit",{**config,"mode":mode,"loss":loss,"seed":seed},inputs=[data_path]) as fit:
                        torch.manual_seed(seed)
                        model=ShiftNetwork(mode)
                        initial_hash=digest({k:v.tolist() for k,v in model.state_dict().items()})
                        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
                        x=torch.tensor([r['features'][mode] for r in data['train']],dtype=torch.float32)
                        y=torch.tensor([r['label'] for r in data['train']],dtype=torch.float32)
                        sampler=torch.Generator().manual_seed(seed)
                        for step in range(config['steps']):
                            indices=torch.randint(len(x),(config['batch_size'],),generator=sampler)
                            optimizer.zero_grad()
                            objective=proper_loss(model(x[indices]),y[indices],loss)
                            if not torch.isfinite(objective): raise FloatingPointError("Non-finite Judge loss")
                            objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                            optimizer.step()
                            if (step+1)%100==0 or step==config['steps']-1:
                                fit.metric(step+1,loss=objective.item())
                        model.eval()
                        with torch.no_grad():
                            cx=torch.tensor([r['features'][mode] for r in data['calibration']],dtype=torch.float32)
                            cy=torch.tensor([r['label'] for r in data['calibration']],dtype=torch.float32)
                            logits=model(cx)
                            scores=[{'temperature':t,'log_loss':proper_loss(logits/t,cy,'log').item()} for t in config['temperatures']]
                            temperature=min(scores,key=lambda r:r['log_loss'])['temperature']
                        checkpoint=fit.path/'checkpoint.pt'
                        torch.save({'origin':'aim-random-init-v1','kind':VERSION,'feature_version':FEATURE_VERSION,
                            'mode':mode,'target':TARGET,'temperature':temperature,'model':model.state_dict(),
                            'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                            'sampler_rng':sampler.get_state(),'step':config['steps'],'configuration':config,
                            'seed':seed,'loss':loss,'dataset_hash':file_hash(data_path),'initial_hash':initial_hash},checkpoint)
                        write_json(checkpoint.with_name(checkpoint.name+'.sha256.json'),{'sha256':file_hash(checkpoint)})
                        models[name]={'checkpoint':str(checkpoint.resolve()),'sha256':file_hash(checkpoint),
                            'temperature':temperature,'calibration_scores':scores,'mode':mode,'loss':loss,'seed':seed,
                            'initial_hash':initial_hash,'parameters':sum(p.numel() for p in model.parameters())}
                        write_json(fit.path/'result.json',models[name])
        write_json(run.path/'frozen.json',{'version':VERSION,'models':models,'configuration':config,
            'holdout_sha256':metadata['holdout_sha256'],
            'training_sha256':file_hash(data_path),'training_base_rate':sum(r['label'] for r in data['train'])/len(data['train']),
            'training_groups':sorted({r['group'] for r in data['train']}),
            'calibration_groups':sorted({r['group'] for r in data['calibration']})})
    return run.path


class ShiftJudge:
    """Opt-in Controller adapter. Extractor cannot see claims, outcomes or labels."""
    def __init__(self, checkpoint, *, calibrated=True, cost=0.5):
        if type(cost) not in (int,float) or not 0<=cost<=1: raise ContractError("Invalid cost")
        record=read_checkpoint(checkpoint)
        if (record.get('kind')!=VERSION or record.get('feature_version')!=FEATURE_VERSION or record.get('target')!=TARGET
            or record.get('temperature') not in (0.5,0.75,1.,1.25,1.5,2.,3.,5.)):
            raise ContractError("Incompatible shift Judge checkpoint")
        self.model=ShiftNetwork(record['mode']);self.model.load_state_dict(record['model']);self.model.eval()
        self.mode=record['mode'];self.temperature=record['temperature'] if calibrated else 1.
        self.cost=cost;self.model_id='judge-shift-'+file_hash(checkpoint)

    def forecast(self, pre):
        with torch.no_grad():
            return (self.model(torch.tensor([extract(pre,self.mode)],dtype=torch.float32))/self.temperature).sigmoid().item()

    def decide(self,state,hypothesis,claim):
        from .researcher import predict
        if (claim.hypothesis_id!=hypothesis.id or claim.target_x!=state.target_x
            or set(claim.evidence_ids)!=set(hypothesis.evidence_ids)
            or abs(claim.predicted-predict(hypothesis.coefficients,state.target_x))>1e-8):
            raise ContractError('Judge claim does not match candidate and target')
        p=self.forecast(payload(state,hypothesis))
        return Decision(claim.id,p,TARGET,'VERIFY' if p>=self.cost else 'ABSTAIN',self.model_id,
                        'Synthetic validation only; shift performance reported separately')
