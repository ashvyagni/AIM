"""Whole-question accounting and paired group uncertainty; no policy fitting."""
import numpy as np


def weighted(rows,family,prior,acquisition_cost=None):
    groups={}
    for row in rows:
        if row['family'] not in ('base',family): continue
        bucket=groups.setdefault(row['group'],{})
        if row['family'] in bucket: raise ValueError('Duplicate group-family episode')
        s=row['summary'];a=s['attempts']['acquisition'];m=s['attempts']['target']
        cost=s['acquisition_cost'] if acquisition_cost is None else acquisition_cost
        bucket[row['family']]={'success':float(s['success']),'utility':float(s['success'])-s['target_cost']*m-cost*a,
            'acquisition_rate':a,'target_rate':m,'tool_dispatches':s['tool_dispatches'],
            'unknown_question_rate':float(bool(s['unknown_reasons']) and not s['success']),
            'tool_seconds':s['tool_elapsed_seconds']}
    result={}
    for group,pair in groups.items():
        if set(pair)!={'base',family}: raise ValueError('Incomplete factorial pair')
        result[group]={k:prior*pair['base'][k]+(1-prior)*pair[family][k] for k in pair['base']}
    if not result: raise ValueError('Empty evaluation')
    return result


def summarize(episodes,config):
    metrics={};contrasts={}
    for family in ('cubic','quintic'):
        for prior in config['priors']:
            scenario=f'{family}-prior{prior}'
            by_policy={name:weighted(rows,family,prior) for name,rows in episodes.items()}
            group_sets=[set(v) for v in by_policy.values()]
            if any(g!=group_sets[0] for g in group_sets): raise ValueError('Policy groups differ')
            metrics[scenario]={}
            for name,groups in by_policy.items():
                means={key:float(np.mean([r[key] for r in groups.values()])) for key in next(iter(groups.values()))}
                means['acquisition_cost_sensitivity']={str(cost):float(np.mean([v['utility'] for v in
                    weighted(episodes[name],family,prior,cost).values()])) for cost in config['sensitivity_acquisition_costs']}
                metrics[scenario][name]=means
            contrasts[scenario]={}
            for seed in config['seeds']:
                left=by_policy[f'acquire_judge-seed{seed}'];right=by_policy[f'judge_direct-seed{seed}']
                differences=np.array([left[g]['utility']-right[g]['utility'] for g in sorted(left)])
                rng=np.random.default_rng(config['bootstrap_seed'])
                boot=differences[rng.integers(len(differences),size=(config['bootstrap_draws'],len(differences)))].mean(axis=1)
                contrasts[scenario][str(seed)]={'utility_difference':float(differences.mean()),
                    'percentile95':np.quantile(boot,[.025,.975]).tolist(),'groups':len(left)}
    gates={str(seed):{
        'known_balanced_improvement':contrasts['cubic-prior0.5'][str(seed)]['utility_difference']>=.02,
        'quintic_balanced_utility':metrics['quintic-prior0.5'][f'acquire_judge-seed{seed}']['utility']>=0,
        'quintic_low_prior_utility':metrics['quintic-prior0.25'][f'acquire_judge-seed{seed}']['utility']>=0}
        for seed in config['seeds']}
    return {'metrics':metrics,'contrasts':contrasts,'gates':gates,
        'utility_gate_passed':all(all(v.values()) for v in gates.values()),
        'scope':'Shared actual episodes, weighted family/prior scenarios; unchanged actions in cost sensitivity'}
