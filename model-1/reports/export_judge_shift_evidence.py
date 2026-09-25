"""Audit a completed Phase 2B run and export inspectable, non-overwriting evidence."""
import argparse
import json
import re
from pathlib import Path

from aim.contracts import Claim, Evidence, Outcome, ToolResult
from aim.datasets import world_value
from aim.judge_shift_eval import scores, policy_metrics, paired_interval
from aim.judge_shift_features import VERSION
from aim.judge_shift_train import validate_rows
from aim.memory import Memory
from aim.tracking import ROOT, file_hash, write_json
from aim.verifiers import MeasurementVerifier


def read(path): return json.loads(Path(path).read_text())


def portable(value):
    if isinstance(value,str): return value.replace(str(ROOT.resolve())+'/', 'model-1/')
    if isinstance(value,list): return [portable(v) for v in value]
    if isinstance(value,dict):
        result={portable(k):portable(v) for k,v in value.items()}
        if isinstance(result.get('git_status'),str):
            result['git_status']='\n'.join(line for line in result['git_status'].splitlines() if not line.startswith('?? '))
            result['git_status_note']='Untracked filenames omitted; original manifest retained locally.'
        return result
    return value


def completed(path):
    status=read(Path(path)/'status.json')
    if status['status']!='COMPLETED': raise ValueError('Incomplete stage')
    return status


def export(run,output,development_failure=None):
    run=Path(run).resolve();output=Path(output)
    status=completed(run);summary=read(run/'summary.json')
    dataset=Path(summary['dataset']);training=Path(summary['training']);evaluation=Path(summary['evaluation'])
    stages={k:completed(summary[k]) for k in ('dataset','training','evaluation')}
    frozen=read(training/'frozen.json');metrics=read(evaluation/'metrics.json');predictions=read(evaluation/'predictions.json')
    if metrics!=summary['results'] or metrics['version']!=VERSION: raise ValueError('Summary mismatch')
    if file_hash(training/'frozen.json')!=metrics['frozen_sha256']: raise ValueError('Frozen identity changed')
    if file_hash(dataset/'holdout.json')!=metrics['holdout_sha256'] or metrics['holdout_sha256']!=frozen['holdout_sha256']:
        raise ValueError('Holdout identity changed')
    if file_hash(dataset/'train-calibration.json')!=frozen['training_sha256']: raise ValueError('Training data changed')
    fits={};hashes=set()
    for name,record in frozen['models'].items():
        checkpoint=Path(record['checkpoint'])
        if file_hash(checkpoint)!=record['sha256']: raise ValueError('Checkpoint changed')
        manifest=read(checkpoint.parent/'manifest.json');hashes.add(manifest['code_hash'])
        fits[name]={'manifest':manifest,'status':completed(checkpoint.parent)}
    if len(hashes)!=1: raise ValueError('Fit source snapshots differ')
    splits={**{k:v for k,v in read(dataset/'train-calibration.json').items() if k!='version'},
            **{k:v for k,v in read(dataset/'holdout.json').items() if k!='version'}}
    rows_by_id={};groups=set()
    for split,rows in splits.items():
        validate_rows(rows,split)
        current={r['group'] for r in rows}
        if current & groups: raise ValueError('Group leakage')
        groups |= current
        for row in rows:
            if row['id'] in rows_by_id: raise ValueError('Duplicate row identity')
            rows_by_id[row['id']]=row
    audit=read(dataset/'label-audit.json')['rows']
    if len(audit)!=len(rows_by_id) or {a['id'] for a in audit}!=set(rows_by_id): raise ValueError('Incomplete label audit')
    db=dataset/'memory/memory.sqlite';before=file_hash(db);memory=Memory(dataset/'memory',read_only=True)
    try:
        for a in audit:
            row=rows_by_id[a['id']]
            source=Evidence(**row['source']);mev=Evidence(**a['measurement_evidence'])
            if not memory.validate(source) or not memory.validate(mev): raise ValueError('Broken source provenance')
            if json.loads(source.quote)['observations']!=row['input']['observations']: raise ValueError('Source/input mismatch')
            m=a['measurement'].copy();m['outcome']=Outcome(m['outcome'])
            measurement=json.loads(mev.quote)
            if (measurement!={'x':row['input']['target_x'],'value':m['value']}
                or world_value(a['coefficients'],row['input']['target_x'])!=m['value']):
                raise ValueError('Stored measurement/environment mismatch')
            check=MeasurementVerifier().verify(Claim(**a['claim']),ToolResult(**m),mev)
            if check.id!=row['verification_id'] or int(check.outcome==Outcome.PASS)!=row['label']:
                raise ValueError('Independent label audit failed')
    finally: memory.close()
    if file_hash(db)!=before: raise ValueError('Audit changed source database')
    config=frozen['configuration']
    for split in ('test','ood'):
        rows=splits[split];p=predictions[split]
        if [r['id'] for r in rows]!=[r['id'] for r in p['rows']]: raise ValueError('Forecast pairing changed')
        for name,values in p['probabilities'].items():
            measured=scores(values,rows,config['costs'])
            if any(metrics['metrics'][split][name][key]!=value for key,value in measured.items()):
                raise ValueError('Forecast metrics mismatch')
        for seed in config['seeds']:
            contrast=paired_interval(rows,p['probabilities'][f'rich-log-seed{seed}-temperature'],
                p['probabilities'][f'five-log-seed{seed}-temperature'],config)
            if contrast!=metrics['paired_contrasts'][split][str(seed)]: raise ValueError('Bootstrap mismatch')
        for name,actions in {'verify_all':[True]*len(rows),'abstain_all':[False]*len(rows),
                             'observed_fit':[bool(r['features']['rich'][5]) for r in rows]}.items():
            for cost in config['costs']:
                if policy_metrics(actions,[r['label'] for r in rows],cost)!=metrics['metrics'][split]['deterministic_policies'][name][str(cost)]:
                    raise ValueError('Policy metric mismatch')
    log=(run/'tests.log').read_text();match=re.search(r'Ran (\d+) tests',log)
    if not match or 'skipped=' in log or read(run/'test-result.json')['returncode']: raise ValueError('Preflight failure')
    output.mkdir(parents=True,exist_ok=False)
    source_paths={'summary':run/'summary.json','frozen':training/'frozen.json','predictions':evaluation/'predictions.json',
        'training':dataset/'train-calibration.json','holdout':dataset/'holdout.json','label_audit':dataset/'label-audit.json'}
    report={'summary':summary,'frozen':frozen,'fits':fits,'reproduction_manifest':read(run/'manifest.json'),
        'status':status,'stage_statuses':stages,'dataset_manifest':read(dataset/'manifest-data.json'),
        'source_hashes':{k:file_hash(v) for k,v in source_paths.items()},
        'audit':{'passed':True,'label_rows':len(rows_by_id),'base_groups':len(groups),'tests':int(match.group(1)),
                 'fit_count':len(fits),'code_hashes':sorted(hashes),'memory_sha256':before}}
    write_json(output/'results.json',portable(report));write_json(output/'predictions.json',portable(predictions))
    for name,text in [('tests.log',log)]+([('development-failure.log',Path(development_failure).read_text())] if development_failure else []):
        text=portable(text)
        text=re.sub(r'/var/folders/[^\s\"\']+/T/tmp[^/\s\"\']+','<temporary-test-directory>',text)
        (output/name).write_text(text)
    write_json(output/'export-manifest.json',{'source_run':portable(str(run)),'exporter_sha256':file_hash(__file__),
        'note':'Path normalization and untracked-filename omission affect exported copies only; original runs unchanged.',
        'files':{p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(report['audit']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--development-failure',type=Path)
    args=parser.parse_args();export(args.run,args.output,args.development_failure)
