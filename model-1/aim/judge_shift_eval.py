"""Frozen forecast and one-step policy evaluation, with whole-group uncertainty."""
import json
from pathlib import Path

import numpy as np

from .contracts import ContractError
from .judge_shift_features import VERSION
from .judge_shift_train import ShiftJudge, validate_rows
from .metrics import calibration_metrics
from .tracking import Run, file_hash, write_json


def policy_metrics(actions, labels, cost):
    n=len(labels)
    if len(actions)!=n or not n: raise ValueError('Policy shape mismatch')
    selected=sum(actions);passed=sum(a*y for a,y in zip(actions,labels))
    return {'utility':sum(a*(y-cost) for a,y in zip(actions,labels))/n,
        'verification_fraction':selected/n,'verified_successes':passed,'forgone_successes':sum(labels)-passed,
        'failure_risk':1-passed/selected if selected else None}


def scores(p, rows, costs):
    y=[r['label'] for r in rows]
    result=calibration_metrics(p,y)
    result['high_confidence_wrong']=sum(prob>=.95 and label==0 for prob,label in zip(p,y))
    result['high_confidence_wrong_fraction']=result['high_confidence_wrong']/len(rows)
    result['policies']={str(c):policy_metrics([prob>=c for prob in p],y,c) for c in costs}
    return result


def paired_interval(rows, left, right, config):
    grouped={}
    for row,a,b in zip(rows,left,right):
        y=row['label'];grouped.setdefault(row['group'],[]).append(
            ((a-y)**2-(b-y)**2, (a>=.5)*(y-.5)-(b>=.5)*(y-.5)))
    means=np.array([np.mean(grouped[g],axis=0) for g in sorted(grouped)])
    rng=np.random.default_rng(config['bootstrap_seed'])
    boot=means[rng.integers(len(means),size=(config['bootstrap_draws'],len(means)))].mean(axis=1)
    return {name:{'difference':float(means[:,i].mean()),'percentile95':np.quantile(boot[:,i],[.025,.975]).tolist()}
            for i,name in enumerate(('brier','utility_at_0.5'))}


def evaluate(frozen_path, holdout_path, runs):
    frozen_path=Path(frozen_path);holdout_path=Path(holdout_path)
    frozen=json.loads(frozen_path.read_text())
    if frozen['version']!=VERSION: raise ContractError('Wrong frozen study')
    if file_hash(holdout_path)!=frozen['holdout_sha256']: raise ContractError('Holdout hash changed')
    # Validate every frozen model before opening any holdout records.
    judges={}
    for name,item in frozen['models'].items():
        if file_hash(item['checkpoint'])!=item['sha256']: raise ContractError('Frozen checkpoint changed')
        for calibrated in (False,True):
            key=name+('-temperature' if calibrated else '-raw')
            judges[key]=ShiftJudge(item['checkpoint'],calibrated=calibrated)
            if calibrated and judges[key].temperature!=item['temperature']: raise ContractError('Frozen temperature mismatch')
    data=json.loads(holdout_path.read_text())
    if set(data)!={'version','test','ood'} or data['version']!=VERSION: raise ContractError('Wrong holdout schema')
    known=set(frozen['training_groups']+frozen['calibration_groups'])
    for split in ('test','ood'):
        validate_rows(data[split],split)
        groups={r['group'] for r in data[split]}
        if known & groups: raise ContractError('Holdout group leakage')
        known |= groups
    config=frozen['configuration']
    with Run(Path(runs),'judge-shift-evaluation',config,inputs=[frozen_path,holdout_path]) as run:
        all_predictions={};metrics={};contrasts={};gates={}
        for split in ('test','ood'):
            rows=data[split];y=[r['label'] for r in rows]
            predictions={name:[judge.forecast(r['input']) for r in rows] for name,judge in judges.items()}
            predictions.update(constant_half=[.5]*len(rows),base_rate=[frozen['training_base_rate']]*len(rows))
            metrics[split]={}
            for name,p in predictions.items():
                result=scores(p,rows,config['costs'])
                result['slices']={}
                for field in ('family','observations_count'):
                    for value in sorted({r[field] for r in rows}):
                        ix=[i for i,r in enumerate(rows) if r[field]==value]
                        result['slices'][f'{field}={value}']=scores([p[i] for i in ix],[rows[i] for i in ix],config['costs'])
                metrics[split][name]=result
            policies={}
            for name,actions in {'verify_all':[True]*len(rows),'abstain_all':[False]*len(rows),
                                 'observed_fit':[bool(r['features']['rich'][5]) for r in rows]}.items():
                policies[name]={str(c):policy_metrics(actions,y,c) for c in config['costs']}
            metrics[split]['deterministic_policies']=policies
            all_predictions[split]={'rows':[{k:r[k] for k in ('id','group','world','family','observations_count','label')} for r in rows],
                                    'probabilities':predictions}
            contrasts[split]={str(seed):paired_interval(rows,predictions[f'rich-log-seed{seed}-temperature'],
                                    predictions[f'five-log-seed{seed}-temperature'],config) for seed in config['seeds']}
        for seed in config['seeds']:
            key=f'rich-log-seed{seed}-temperature';t=metrics['test'][key];o=metrics['ood'][key]
            gates[str(seed)]={'familiar_brier':t['brier']<=metrics['test']['base_rate']['brier']-.02,
                'ood_brier':o['brier']<=metrics['ood']['base_rate']['brier']+.02,
                'familiar_utility':t['policies']['0.5']['utility']>=max(metrics['test']['base_rate']['policies']['0.5']['utility'],
                    metrics['test']['deterministic_policies']['verify_all']['0.5']['utility']),
                'ood_utility':o['policies']['0.5']['utility']>=0,
                'familiar_coverage':t['policies']['0.5']['verification_fraction']>=.1,
                'ood_high_confidence_error':o['high_confidence_wrong_fraction']<=.05}
        write_json(run.path/'predictions.json',all_predictions)
        write_json(run.path/'metrics.json',{'version':VERSION,'metrics':metrics,'paired_contrasts':contrasts,'gates':gates,
            'evidence_gate_passed':all(all(g.values()) for g in gates.values()),
            'frozen_sha256':file_hash(frozen_path),'holdout_sha256':file_hash(holdout_path),
            'scope':'Synthetic complete-outcome per-candidate utility; not measured shared-tool savings or sequential RL'})
    return run.path
