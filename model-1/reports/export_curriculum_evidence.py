"""Export a completed curriculum run without modifying its original artifacts.

From model-1: PYTHONPATH=. .venv/bin/python reports/export_curriculum_evidence.py
  --run runs/<curriculum-reproduction> --output reports/<new-evidence-directory>
"""
import argparse
import json
import re
from pathlib import Path

from aim.curriculum_tasks import EXPERIMENT
from aim.tracking import ROOT, file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text())


def portable(value):
    if isinstance(value,str):
        return value.replace(str(ROOT.resolve())+'/', 'model-1/')
    if isinstance(value,list):
        return [portable(v) for v in value]
    if isinstance(value,dict):
        result={portable(k):portable(v) for k,v in value.items()}
        if isinstance(result.get('git_status'),str):
            lines=result['git_status'].splitlines()
            result['git_status']='\n'.join(line for line in lines if not line.startswith('?? '))
            if any(line.startswith('?? ') for line in lines):
                result['git_status']+='\n[untracked path names omitted; original status retained locally]'
        return result
    return value


def completed(path):
    status=read(path/'status.json')
    if status['status']!='COMPLETED':
        raise ValueError(f'Incomplete experiment stage: {path}')
    return status


def export(run,output):
    run=Path(run).resolve();output=Path(output)
    summary=read(run/'summary.json');root_status=completed(run)
    training=Path(summary['training']);evaluation=Path(summary['evaluation'])
    audit=Path(summary['observation_audit']);dataset=Path(summary['dataset'])
    stages={name:completed(path) for name,path in
            [('training',training),('evaluation',evaluation),('observation_audit',audit),('dataset',dataset)]}
    frozen=read(training/'selected-models.json')
    if frozen['version']!=EXPERIMENT: raise ValueError('Wrong experiment version')
    if file_hash(dataset/'holdout.json')!=frozen['holdout_sha256']:
        raise ValueError('Holdout hash changed')
    cases=read(evaluation/'case-results.json');checks=read(audit/'case-checks.json')
    metrics=summary['metrics'];n_cases=0
    for label,rows in cases.items():
        score=metrics['backends'][label];n_cases+=len(rows)
        if len(rows)!=score['n'] or sum(r['verified'] for r in rows)!=score['verified_worlds']:
            raise ValueError('Case metrics mismatch')
        if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate evaluation world')
        if [r['id'] for r in rows]!=[r['id'] for r in checks[label]]: raise ValueError('Audit pairing mismatch')
        observed=summary['observation_metrics']['backends'][label]
        if (sum(c['observation_consistent'] for c in checks[label])!=observed['observation_passes'] or
            sum(c['target_and_observations_pass'] for c in checks[label])!=observed['target_and_observations_pass']):
            raise ValueError('Observation aggregate mismatch')
        for row,check in zip(rows,checks[label]):
            path=Path(row['run']);completed(path)
            if file_hash(path/'state.json')!=check['state_sha256'] or file_hash(path/'memory/memory.sqlite')!=check['memory_sha256']:
                raise ValueError('Audited source artifacts changed')
    seeds={};selected_fit={};code_hashes=set()
    for arm,details in frozen['arms'].items():
        for selection in details['selections']:
            key=f"{arm}-seed{selection['seed']}"
            config=frozen['configuration']
            if (selection['processed_positions']!=config['steps']*config['batch_size']*config['pad_to'] or
                sum(selection['task_counts'].values())!=config['steps']*config['batch_size']):
                raise ValueError('Training budget or task accounting mismatch')
            for phase in ('initial','selected'):
                checkpoint=selection[phase]
                if file_hash(checkpoint['checkpoint'])!=checkpoint['sha256']:
                    raise ValueError('Frozen checkpoint changed')
            directory=Path(selection['initial']['checkpoint']).parent
            manifest=read(directory/'manifest.json');code_hashes.add(manifest['code_hash'])
            curves=[]
            for point in selection['evaluations']:
                fit_path=directory/f"fit-step{point['step']:04d}.json"
                fit=read(fit_path)
                curves.append({'step':point['step'],'fit_sha256':file_hash(fit_path),
                    'research_train':{k:v for k,v in fit['research_training_sample'].items() if k!='outputs'},
                    'research_validation':{k:point[k] for k in ('n','prediction_agreement_rate','valid_rate','response_token_nll')},
                    'auxiliary':{k:{a:b for a,b in v.items() if a!='outputs'} for k,v in fit['auxiliary'].items()},
                    'pool_counts':fit['pool_counts']})
                if point['step']==selection['selected']['step']: selected_fit[key]=fit
            seeds[key]={'directory':str(directory),'status':completed(directory),'manifest':manifest,'fit_curves':curves}
    if len(code_hashes)!=1: raise ValueError('Seed implementation snapshots differ')
    for left,right in zip(frozen['arms']['worked']['selections'],frozen['arms']['curriculum']['selections']):
        if left['sampling_digest']!=right['sampling_digest']: raise ValueError('Paired world draws differ')
    testlog=(run/'tests.log').read_text()
    match=re.search(r'Ran (\d+) tests',testlog)
    if not match or 'skipped=' in testlog or read(run/'test-result.json')['returncode']!=0:
        raise ValueError('Missing complete passing preflight')
    source_hashes={name:file_hash(path) for name,path in {
        'summary':run/'summary.json','selection':training/'selected-models.json',
        'case_results':evaluation/'case-results.json','observation_checks':audit/'case-checks.json',
        'test_log':run/'tests.log','split_audit':dataset/'split-audit.json'}.items()}
    # Refuse overwrites; inspect all original evidence before creating the export.
    output.mkdir(parents=True,exist_ok=False)
    results={'summary':summary,'frozen_selection':frozen,'seed_runs':seeds,'reproduction_status':root_status,
             'stage_statuses':stages,'reproduction_manifest':read(run/'manifest.json'),
             'dataset_manifest':read(dataset/'split-manifest.json'),'source_artifact_hashes':source_hashes,
             'artifact_audit':{'passed':True,'case_runs':n_cases,'tests':int(match.group(1)),'code_hashes':sorted(code_hashes)}}
    for name,value in [('results.json',results),('case-results.json',cases),
                       ('observation-checks.json',checks),('selected-fit-diagnostics.json',selected_fit)]:
        write_json(output/name,portable(value))
    testlog=re.sub(r'/var/folders/[^\s\"\']+/T/tmp[^/\s\"\']+', '<temporary-test-directory>',testlog)
    (output/'tests.log').write_text(testlog)
    write_json(output/'export-manifest.json',{'source_run':portable(str(run)),
        'exporter_sha256':file_hash(__file__),
        'note':'Workspace paths made repository-relative; test temporary paths and untracked filenames redacted. Original experiments unchanged.',
        'files':{p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({'output':str(output),'case_runs':n_cases,'tests':int(match.group(1))}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    export(args.run,args.output)
