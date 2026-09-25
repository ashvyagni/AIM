import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from aim.contracts import ContractError, Outcome, ResearchState, ToolResult
from aim.datasets import polynomial_case, world_value
from aim.memory import Memory
from aim.paid_evidence import PaidEvidenceController, SharedPolicy, aggregate
from aim.paid_evidence_data import bases
from aim.researcher import PolynomialResearcher
from aim.tools import ToolRunner
from aim.tracking import ROOT, Run, canonical, file_hash, write_json


class FixedForecast:
    model_id='test-only-fixed'
    def __init__(self,p): self.p=p;self.inputs=[]
    def forecast(self,pre):
        assert set(pre)=={'observations','coefficients','target_x'}
        self.inputs.append(copy.deepcopy(pre));return self.p


class FailedAcquisition:
    def __init__(self,outcome): self.outcome=outcome
    def execute(self,action):
        if action.name=='MEASURE' and action.arguments['x']==3:
            return ToolResult(action.id,self.outcome,None,'Injected test failure',0.,'test-only')
        return ToolRunner().execute(action)


class PaidEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=json.loads((ROOT/'configs/paid-evidence.json').read_text())

    def tearDown(self): self.temp.cleanup()

    def case(self,coefs=(300,2,1)):
        case=polynomial_case('paid-unit',list(coefs),target=4)
        case['acquisition']={'x':3,'available':True,'observations':[[3,world_value(coefs,3)]]}
        return case

    def execute(self,kind,case=None,forecaster=None,**kwargs):
        policy=SharedPolicy(kind,self.config,forecaster)
        with Run(self.root,'paid-unit',{'kind':kind}) as run:
            state,summary=PaidEvidenceController(policy,**kwargs).run(case or self.case(),run)
        return state,summary,run.path

    def test_aggregate_deduplicates_and_caps(self):
        self.assertEqual(aggregate([4,4,4],[.6,.6,.6])[0],.6)
        self.assertEqual(aggregate([4,5],[.6,.6])[0],1.)
        self.assertEqual(aggregate([4,4],[.3,.6])[0],.6)
        for p in (True,-1,2,float('nan')):
            with self.assertRaises(ContractError): aggregate([4],[p])
        with self.assertRaises(ContractError): aggregate([4],[])

    def test_candidate_duplicates_never_multiply_question_reward(self):
        state,s,_=self.execute('verify_all',self.case((300,2,0)))
        self.assertEqual(s['verified_claims'],2);self.assertTrue(s['success'])
        self.assertEqual(s['attempts']['target'],1);self.assertEqual(s['utility'],.5)
        self.assertEqual(s['verified_with_passing_checks'],2)

    def test_abstention_dispatches_no_measurement(self):
        _,s,_=self.execute('abstain_all')
        self.assertEqual(s['attempts'],{'acquisition':0,'target':0,'calculation':2})
        self.assertEqual(s['utility'],0);self.assertFalse(s['success'])

    def test_acquisition_is_persisted_and_charged(self):
        state,s,path=self.execute('acquire_fit')
        self.assertEqual(len(state.observations),4)
        self.assertEqual(s['attempts'],{'acquisition':1,'target':1,'calculation':2})
        self.assertAlmostEqual(s['utility'],.4)
        memory=Memory(path/'memory',read_only=True)
        try:
            self.assertTrue(all(memory.validate(e) for e in state.evidence))
            self.assertEqual(canonical(memory.replay()),canonical(state.to_dict()))
        finally: memory.close()
        self.assertEqual(len(s['decisions'][0]['inputs'][0]['observations']),3)
        self.assertEqual(len(s['decisions'][1]['inputs'][0]['observations']),4)

    def test_extra_point_separates_cubic_but_not_quintic(self):
        _,c,_=self.execute('acquire_fit',self.case((300,4,-2,1)))
        self.assertEqual(c['attempts']['target'],0);self.assertAlmostEqual(c['utility'],-.1)
        _,q,_=self.execute('acquire_fit',self.case((300,-4,6,5,-5,1)))
        self.assertEqual(q['attempts']['target'],1);self.assertFalse(q['success'])
        self.assertAlmostEqual(q['utility'],-.6)

    def test_unavailable_acquisition_stops_without_free_target_fallback(self):
        case=self.case();case['acquisition']['available']=False
        state,s,_=self.execute('acquire_fit',case)
        self.assertFalse(state.claims);self.assertEqual(s['attempts']['target'],0)
        self.assertEqual(s['attempts']['acquisition'],1);self.assertAlmostEqual(s['utility'],-.1)

    def test_error_and_timeout_are_paid_and_unresolved(self):
        for outcome in (Outcome.ERROR,Outcome.TIMEOUT):
            state,s,_=self.execute('acquire_fit',tools=FailedAcquisition(outcome))
            self.assertFalse(state.claims);self.assertAlmostEqual(s['utility'],-.1)
            self.assertEqual(s['attempts']['target'],0)

    def test_budget_only_charges_dispatched_work(self):
        _,s,_=self.execute('acquire_fit',max_actions=0)
        self.assertEqual(s['tool_dispatches'],0);self.assertEqual(s['utility'],0)
        _,s,_=self.execute('acquire_fit',max_actions=1)
        self.assertEqual(s['attempts']['acquisition'],1);self.assertAlmostEqual(s['utility'],-.1)
        self.assertEqual(s['attempts']['target'],0)

    def test_conflicting_source_prevents_acquisition(self):
        case=polynomial_case('paid-conflict',[300,2,1],conflicting=True)
        case['acquisition']=self.case()['acquisition']
        state,s,_=self.execute('acquire_fit',case)
        self.assertTrue(state.contradictions);self.assertEqual(s['tool_dispatches'],0)

    def test_missing_acquisition_fails_closed(self):
        case=self.case();del case['acquisition']
        state,s,_=self.execute('acquire_fit',case)
        self.assertFalse(state.claims);self.assertEqual(s['tool_dispatches'],0)

    def test_forecaster_sees_only_pre_measurement_inputs(self):
        forecast=FixedForecast(.3)
        state,s,path=self.execute('selective_judge',forecaster=forecast)
        self.assertEqual(s['attempts']['acquisition'],1)
        self.assertTrue(all(max(x for x,_ in p['observations'])<p['target_x'] for p in forecast.inputs))
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        self.assertLess(next(i for i,e in enumerate(events) if e['kind']=='ACQUISITION_DECISION'),
                        next(i for i,e in enumerate(events) if e['kind']=='ACQUISITION_RESULT'))

    def test_selective_band_boundaries_and_input_validation(self):
        for probability,expected in ((.2,False),(.8,False),(.5,True)):
            policy=SharedPolicy('selective_judge',self.config,FixedForecast(probability))
            self.assertEqual(policy.acquire({'aggregate':probability}),expected)
        bad={**self.config,'target_cost':True}
        with self.assertRaises(ContractError): SharedPolicy('verify_all',bad)
        with self.assertRaises(ContractError): SharedPolicy('judge_direct',self.config)
        _,s,_=self.execute('judge_direct',forecaster=FixedForecast(float('nan')))
        self.assertFalse(s['success']);self.assertEqual(s['attempts']['target'],0)

    def test_fresh_groups_and_invisible_family(self):
        b=bases();self.assertEqual(len(b),32);self.assertEqual(len({g for g,_ in b}),32)
        self.assertEqual(sum(q[2]==0 for _,q in b),16)
        for _,(a,b,c) in b:
            self.assertGreaterEqual(a,120)
            for x in (0,1,2,3):
                self.assertEqual(world_value((a,b,c),x),world_value((a,b-6,c+5,5,-5,1),x))

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'numpy required')
    def test_factorial_weighting_and_cost_sensitivity(self):
        from aim.paid_evidence_eval import weighted
        rows=[]
        for family,success in [('base',True),('quintic',False)]:
            rows.append({'group':'g','family':family,'summary':{'success':success,'target_cost':.5,'acquisition_cost':.1,
                'attempts':{'acquisition':1,'target':1},'tool_dispatches':4,'unknown_reasons':[],'tool_elapsed_seconds':0}})
        self.assertAlmostEqual(weighted(rows,'quintic',.5)['g']['utility'],-.1)
        self.assertAlmostEqual(weighted(rows,'quintic',.25)['g']['utility'],-.35)
        self.assertAlmostEqual(weighted(rows,'quintic',.5,0.)['g']['utility'],0.)
        with self.assertRaises(ValueError): weighted(rows[:1],'quintic',.5)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
    def test_frozen_judge_loader_checks_identity(self):
        import torch
        from aim.judge import TARGET
        from aim.judge_shift_features import VERSION,FEATURE_VERSION
        from aim.judge_shift_train import ShiftNetwork
        from aim.paid_evidence_reproduce import load_forecasters
        checkpoint=self.root/'unit-judge.pt'
        torch.save({'origin':'aim-random-init-v1','kind':VERSION,'feature_version':FEATURE_VERSION,
            'mode':'rich','target':TARGET,'temperature':1.,'model':ShiftNetwork('rich').state_dict()},checkpoint)
        write_json(checkpoint.with_name(checkpoint.name+'.sha256.json'),{'sha256':file_hash(checkpoint)})
        path=self.root/'evidence.json';record={'frozen':{'models':{'rich-log-seed17':{
            'checkpoint':str(checkpoint),'sha256':file_hash(checkpoint),'mode':'rich','loss':'log','seed':17,'temperature':1.}}}}
        path.write_text(json.dumps(record));judges,_=load_forecasters(path,[17]);self.assertEqual(judges[17].mode,'rich')
        record['frozen']['models']['rich-log-seed17']['sha256']='0'*64;path.write_text(json.dumps(record))
        with self.assertRaises(ContractError): load_forecasters(path,[17])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
    def test_complete_small_factorial_evaluation(self):
        from aim.paid_evidence_data import build_dataset
        from aim.paid_evidence_reproduce import evaluate
        from aim.tracking import digest
        with patch('aim.paid_evidence_data.bases',return_value=[(digest('unit-paid-a'),(300,2,0)),
                                                             (digest('unit-paid-b'),(301,2,1))]):
            dataset=build_dataset(self.root/'fixture')
        config={**self.config,'seeds':[17],'bootstrap_draws':20}
        evidence=self.root/'test-only-forecaster.json';evidence.write_text('{"test_only":true}')
        output=evaluate(dataset/'worlds.json',config,self.root/'evaluation',{17:FixedForecast(.3)},
            {'17':{'test_only':True}},evidence)
        results=json.loads((output/'metrics.json').read_text())
        self.assertFalse(results['evidence_gate_passed']);self.assertEqual(results['unbacked_verified_claims'],0)
        self.assertEqual(len(json.loads((output/'episode-index.json').read_text())),7)


if __name__=='__main__': unittest.main()
