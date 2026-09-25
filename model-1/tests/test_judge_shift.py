import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aim.contracts import Claim, ContractError, Hypothesis, ResearchState
from aim.judge_shift_data import groups, variants, build_dataset
from aim.judge_shift_features import extract, payload
from aim.tracking import ROOT, Run, digest


def tiny_groups():
    return {s:[(digest(['unit-judge',s,j]),(200+i*10+j,2,j%2),3+j%2) for j in range(4)]
            for i,s in enumerate(('train','calibration','test','ood'))}


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.pre={'observations':[[0,1],[1,2],[2,5]],'coefficients':[1,0,1],'target_x':4}

    def test_exact_fit_and_dimensions(self):
        self.assertEqual(len(extract(self.pre,'five')),5)
        self.assertEqual(extract(self.pre)[5:],[1.,0.])
        self.pre['coefficients'][0]+=1
        self.assertEqual(extract(self.pre)[5],0)
        self.assertGreater(extract(self.pre)[6],0)

    def test_forbidden_inputs(self):
        for field in ('label','measurement','family','world_coefficients','verifications'):
            with self.assertRaises(ContractError): extract({**self.pre,field:1})
        for point in ([4,99],[1,2],[True,2],[2,float('nan')]):
            pre=copy.deepcopy(self.pre);pre['observations'].append(point)
            with self.assertRaises(ContractError): extract(pre)

    def test_payload_ignores_outcomes(self):
        h=Hypothesis('h',(1,0,1),('e',),'')
        state=ResearchState('1','','',4,observations=self.pre['observations'])
        before=extract(payload(state,h))
        state.final_response='label is false';state.unknowns=['hidden'];state.claims=[Claim('c','h','',17,4,['e'])]
        self.assertEqual(before,extract(payload(state,h)))

    def test_group_design_and_invisible_perturbations(self):
        g=groups();all_groups=[x[0] for rows in g.values() for x in rows]
        self.assertEqual(len(all_groups),320);self.assertEqual(len(set(all_groups)),320)
        for split,rows in g.items():
            self.assertEqual(sum(n==3 for _,_,n in rows),len(rows)//2)
            self.assertEqual(sum(q[2]==0 for _,q,_ in rows),len(rows)//2)
            for group,q,_ in rows:
                self.assertGreaterEqual(q[0],30)
                for coefficients,_ in variants(q,group,split):
                    for x in (0,1,2):
                        self.assertEqual(sum(c*x**i for i,c in enumerate(q)),sum(c*x**i for i,c in enumerate(coefficients)))


@unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aim.judge_shift_train import train
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        with patch('aim.judge_shift_data.groups',tiny_groups):
            cls.data=build_dataset(cls.root/'data')
        cls.config=json.loads((ROOT/'configs/judge-shift.json').read_text())
        cls.config.update(steps=8,seeds=[17],bootstrap_draws=20)
        read=Path.read_text
        def guarded(path,*args,**kw):
            if path.name in ('holdout.json','label-audit.json'): raise AssertionError('Training crossed holdout boundary')
            return read(path,*args,**kw)
        with patch.object(Path,'read_text',guarded):
            cls.training=train(cls.data/'train-calibration.json',cls.config,cls.root/'training')

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_independent_labels_and_provenance(self):
        from aim.memory import Memory
        from aim.contracts import Evidence, Outcome, ToolResult
        from aim.verifiers import MeasurementVerifier
        audit=json.loads((self.data/'label-audit.json').read_text())
        with_memory=Memory(self.data/'memory',read_only=True)
        try:
            for row in audit['rows']:
                ev=Evidence(**row['measurement_evidence']);self.assertTrue(with_memory.validate(ev))
                m=row['measurement'];m['outcome']=Outcome(m['outcome'])
                check=MeasurementVerifier().verify(Claim(**row['claim']),ToolResult(**m),ev)
                self.assertEqual(check.outcome.value,row['verification']['outcome'])
        finally: with_memory.close()

    def test_tampered_features_and_metadata(self):
        from aim.judge_shift_train import validate_rows,load_training
        rows=json.loads((self.data/'train-calibration.json').read_text())['train']
        rows[0]['features']['rich'][5]=.5
        with self.assertRaises(ContractError): validate_rows(rows,'train')
        wrong=self.root/'bad.json'
        wrong.write_text(json.dumps({'version':'aim-judge-shift-v1','train':rows,'calibration':rows,'holdout':[]}))
        with self.assertRaises(ContractError): load_training(wrong)

    def test_frozen_evaluation_and_group_intervals(self):
        from aim.judge_shift_eval import evaluate,paired_interval
        output=evaluate(self.training/'frozen.json',self.data/'holdout.json',self.root/'evaluation')
        metrics=json.loads((output/'metrics.json').read_text())
        self.assertEqual(metrics['version'],'aim-judge-shift-v1')
        rows=[{'group':str(i//2),'label':i%2} for i in range(8)]
        result=paired_interval(rows,[.5]*8,[.5]*8,self.config)
        self.assertEqual(result['brier']['percentile95'],[0.,0.])
        changed=self.root/'changed-holdout.json';changed.write_text('{}')
        with self.assertRaises(ContractError): evaluate(self.training/'frozen.json',changed,self.root/'bad-eval')

    def test_policy_math(self):
        from aim.judge_shift_eval import policy_metrics
        self.assertEqual(policy_metrics([True,False],[1,0],.5)['utility'],.25)
        self.assertEqual(policy_metrics([False,False],[1,0],.5)['forgone_successes'],1)
        self.assertIsNone(policy_metrics([False,False],[1,0],.5)['failure_risk'])

    def test_initialization_pairing_and_parameters(self):
        frozen=json.loads((self.training/'frozen.json').read_text())
        for mode,count in (('five',225),('rich',289)):
            a=frozen['models'][f'{mode}-log-seed17'];b=frozen['models'][f'{mode}-brier-seed17']
            self.assertEqual(a['initial_hash'],b['initial_hash']);self.assertEqual(a['parameters'],count)

    def test_adapter_controller_verification_and_abstention(self):
        from aim.judge_shift_train import ShiftJudge
        from aim.controller import Controller
        from aim.datasets import polynomial_case
        frozen=json.loads((self.training/'frozen.json').read_text())
        checkpoint=frozen['models']['rich-log-seed17']['checkpoint']
        for cost in (0.,1.):
            judge=ShiftJudge(checkpoint,cost=cost)
            case=polynomial_case('judge-shift-unit-'+str(cost),[200,2,1])
            with Run(self.root/'loops','judge-adapter-test',{'cost':cost}) as run:
                Controller(judge=judge).run(case,run)
            state=json.loads((run.path/'state.json').read_text())
            self.assertTrue(state['claims'])
            self.assertTrue(all(c['forecast']['action']==('VERIFY' if cost==0 else 'ABSTAIN') for c in state['claims']))
            self.assertEqual(sum(c['status']=='VERIFIED' for c in state['claims']),1 if cost==0 else 0)
        with Run(self.root/'loops','judge-conflict-test',{}) as run:
            Controller(judge=ShiftJudge(checkpoint)).run(polynomial_case('conflict',[200,2,1],conflicting=True),run)
        self.assertFalse(json.loads((run.path/'state.json').read_text())['claims'])


if __name__=='__main__': unittest.main()
