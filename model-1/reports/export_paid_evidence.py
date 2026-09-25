"""Read-only episode, timing, provenance and accounting audit for Phase 2B.1."""
import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from aim.contracts import Claim, Evidence, Verification
from aim.controller import Controller
from aim.judge_shift_features import extract
from aim.memory import Memory
from aim.paid_evidence_eval import summarize
from aim.tracking import ROOT, canonical, file_hash, write_json


def read(path): return json.loads(Path(path).read_text())


def portable(value):
    if isinstance(value,str): return value.replace(str(ROOT.resolve())+'/', 'model-1/')
    if isinstance(value,list): return [portable(v) for v in value]
    if isinstance(value,dict):
        result={portable(k):portable(v) for k,v in value.items()}
        if isinstance(result.get('git_status'),str):
            result['git_status']='\n'.join(line for line in result['git_status'].splitlines() if not line.startswith('?? '))
            result['git_status_note']='Untracked path names omitted from export; original retained.'
        return result
    return value


def completed(path):
    status=read(Path(path)/'status.json')
    if status['status']!='COMPLETED': raise ValueError('Incomplete run')
    return status


def audit_case(row):
    path=Path(row['run']);completed(path);s=row['summary']
    if read(path/'policy-summary.json')!=s: raise ValueError('Summary mismatch')
    if file_hash(path/'state.json')!=s['state_sha256']: raise ValueError('State changed')
    db=path/'memory/memory.sqlite'
    if file_hash(db)!=row['memory_sha256']: raise ValueError('Memory changed')
    state=read(path/'state.json')
    memory=Memory(path/'memory',read_only=True)
    try:
        if canonical(memory.replay())!=canonical(state): raise ValueError('Replay mismatch')
        typed=SimpleNamespace(evidence=[Evidence(**e) for e in state['evidence']],
            claims=[Claim(**c) for c in state['claims']],verifications=[Verification(**v) for v in state['verifications']])
        Controller._validate_final(typed,memory)
        events=[json.loads(r[0]) for r in memory.db.execute('SELECT record FROM events ORDER BY seq')]
    finally: memory.close()
    if file_hash(db)!=row['memory_sha256']: raise ValueError('Audit mutated memory')
    actions=[e['payload'] for e in events if e['kind']=='ACTION']
    results=[e['payload'] for e in events if e['kind']=='RESULT']
    if actions!=[d['action'] for d in s['dispatches']] or results!=[d['result'] for d in s['dispatches']]:
        raise ValueError('Dispatch ledger mismatch')
    decisions=[e['payload'] for e in events if e['kind'] in ('ACQUISITION_DECISION','TARGET_DECISION')]
    if decisions!=s['decisions']: raise ValueError('Decision ledger mismatch')
    attempts={'acquisition':0,'target':0,'calculation':0}
    for i,event in enumerate(events):
        if event['kind']!='ACTION': continue
        action=event['payload']
        if action['name']=='CALCULATE': attempts['calculation']+=1;continue
        kind='target' if action['arguments']['x']==state['target_x'] else 'acquisition'
        attempts[kind]+=1
        required='TARGET_DECISION' if kind=='target' else 'ACQUISITION_DECISION'
        prior=[e['payload'] for e in events[:i] if e['kind']==required]
        if not prior or not prior[-1]['measure' if kind=='target' else 'purchase']:
            raise ValueError('Measurement lacks preceding affirmative decision')
    if attempts!=s['attempts'] or attempts['target']>1 or attempts['acquisition']>1:
        raise ValueError('Cost counts mismatch')
    for decision in decisions:
        for pre in decision['inputs']:
            extract(pre,'rich')
            if decision['kind']=='ACQUISITION_DECISION' and max(x for x,_ in pre['observations'])>2:
                raise ValueError('Acquisition decision leaked its outcome')
    success=any(c['status']=='VERIFIED' for c in state['claims'])
    verified=sum(c['status']=='VERIFIED' for c in state['claims'])
    if s['success']!=success or s['verified_claims']!=verified or s['verified_with_passing_checks']!=verified:
        raise ValueError('Verified credit mismatch')
    if s['utility']!=int(success)-s['target_cost']*attempts['target']-s['acquisition_cost']*attempts['acquisition']:
        raise ValueError('Question utility mismatch')
    if s['tool_dispatches']!=len(actions) or read(path/'loop-summary.json')['action_count']!=len(actions):
        raise ValueError('Tool dispatch count mismatch')
    return read(path/'manifest.json')['code_hash']


def export(run,output,development_failure=None):
    run=Path(run).resolve();output=Path(output);status=completed(run)
    summary=read(run/'summary.json');evaluation=Path(summary['evaluation']);dataset=Path(summary['dataset'])
    stages={'evaluation':completed(evaluation),'dataset':completed(dataset)}
    frozen=read(evaluation/'frozen.json');config=frozen['configuration']
    if file_hash(dataset/'worlds.json')!=frozen['dataset_sha256']: raise ValueError('Dataset changed')
    for record in frozen['judges'].values():
        if file_hash(record['checkpoint'])!=record['sha256']: raise ValueError('Parent checkpoint changed')
    evidence=ROOT/config['judge_evidence']
    if file_hash(evidence)!=frozen['judge_evidence_sha256']: raise ValueError('Parent evidence changed')
    index=read(evaluation/'episode-index.json');episodes={name:read(path) for name,path in index.items()}
    specifications={name:kind for name,kind,_ in frozen['policies']}
    if set(episodes)!=set(specifications): raise ValueError('Frozen policy list changed')
    worlds=read(dataset/'worlds.json')['rows'];world_index={r['id']:r for r in worlds}
    if len(world_index)!=len(worlds): raise ValueError('Duplicate world')
    hashes=set();count=0
    for name,rows in episodes.items():
        if len(rows)!=len(worlds) or {r['id'] for r in rows}!=set(world_index): raise ValueError('Missing policy worlds')
        for row in rows:
            original=world_index[row['id']]
            if any(row[k]!=original[k] for k in ('group','family')): raise ValueError('World attribution mismatch')
            if (row['summary']['policy']!=specifications[name] or
                any(row['summary'][key]!=config[key] for key in ('target_cost','acquisition_cost'))):
                raise ValueError('Frozen policy or utility configuration changed')
            hashes.add(audit_case(row));count+=1
    if len(hashes)!=1: raise ValueError('Episode implementations differ')
    if hashes!={read(run/'manifest.json')['code_hash']} or hashes!={read(evaluation/'manifest.json')['code_hash']}:
        raise ValueError('Root and episode source snapshots differ')
    measured=summarize(episodes,config);metrics=read(evaluation/'metrics.json')
    if any(metrics[k]!=v for k,v in measured.items()) or metrics!=summary['results']: raise ValueError('Aggregate mismatch')
    if metrics['unbacked_verified_claims']!=0 or metrics['evidence_gate_passed']!=measured['utility_gate_passed']:
        raise ValueError('Evidence gate mismatch')
    log=(run/'tests.log').read_text();match=re.search(r'Ran (\d+) tests',log)
    if not match or 'skipped=' in log or read(run/'test-result.json')['returncode']: raise ValueError('Incomplete preflight')
    output.mkdir(parents=True,exist_ok=False)
    report={'summary':summary,'frozen':frozen,'status':status,'stage_statuses':stages,
        'reproduction_manifest':read(run/'manifest.json'),'evaluation_manifest':read(evaluation/'manifest.json'),
        'audit':{'passed':True,'episodes':count,'worlds':len(worlds),'groups':len({w['group'] for w in worlds}),
                 'tests':int(match.group(1)),'code_hashes':sorted(hashes)},
        'source_hashes':{'summary':file_hash(run/'summary.json'),'frozen':file_hash(evaluation/'frozen.json'),
                         'dataset':file_hash(dataset/'worlds.json'),'episode_files':{n:file_hash(p) for n,p in index.items()}}}
    write_json(output/'results.json',portable(report));write_json(output/'episodes.json',portable(episodes))
    for name,text in [('tests.log',log)]+([('development-failure.log',Path(development_failure).read_text())] if development_failure else []):
        text=re.sub(r'/var/folders/[^\s\"\']+/T/tmp[^/\s\"\']+','<temporary-test-directory>',portable(text))
        (output/name).write_text(text)
    write_json(output/'export-manifest.json',{'source_run':portable(str(run)),'exporter_sha256':file_hash(__file__),
        'note':'Original episodes untouched; exported paths normalized and untracked filenames omitted.',
        'files':{p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(report['audit']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--development-failure',type=Path)
    args=parser.parse_args();export(args.run,args.output,args.development_failure)
