import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aim.curriculum_data import partitions, build_dataset
from aim.curriculum_tasks import EXPERIMENT, auxiliary_rows, batch_for, diagnostic_pools, score_auxiliary
from aim.finite_difference import CONTRACT, worked_response
from aim.contracts import ContractError
from aim.tracking import file_hash, write_json

HAS_TORCH=importlib.util.find_spec('torch') is not None


class CurriculumTaskTests(unittest.TestCase):
    def row(self,group='train',split='train'):
        row={'group':group,'split':split,'prompt':json.dumps({'evidence':{'E0':[[0,-2],[1,-4],[2,0]]}})}
        row['auxiliary']=auxiliary_rows(row)
        return row

    def test_exact_signed_teachers_and_strict_scoring(self):
        rows=self.row()['auxiliary']
        self.assertEqual([json.loads(r['response']) for r in rows],
                         [{'value':-2},{'value':4},{'value':6},{'coefficients':[[-2,-5,3]]}])
        for row in rows: self.assertTrue(score_auxiliary(row['response'],row)['correct'])
        for text in ('{"value":true}','{"value":NaN}','{"value":-2,"value":-2}','{"extra":-2}'):
            self.assertFalse(score_auxiliary(text,rows[0])['valid'])
        self.assertFalse(score_auxiliary('{"value":2}',rows[0])['correct'])

    def test_fixed_schedule_and_paired_world_provenance(self):
        rows=[self.row(str(i)) for i in range(16)]
        for step,kind,count in ((1,'subtract',8),(600,'subtract',8),(601,'second_difference',4),
                                (1200,'second_difference',4),(1201,'reconstruct',4),(1800,'reconstruct',4)):
            batch,roles=batch_for({'arm':'curriculum'},rows,step)
            self.assertEqual(roles.count(kind),count);self.assertEqual(roles.count('research'),16-count)
            self.assertEqual([r['world_group'] for r in batch[:count]],[r['group'] for r in rows[:count]])
        self.assertEqual(batch_for({'arm':'worked'},rows,1)[0],rows)

    def test_diagnostic_pool_removes_training_prompt_overlap(self):
        data={'train':[self.row()],'validation':[self.row('different-world','validation')]}
        pools,counts=diagnostic_pools(data)
        for kind in ('subtract','second_difference','reconstruct'):
            self.assertEqual(pools['validation-'+kind],[])
            self.assertGreater(counts['validation-'+kind]['excluded_training_overlap'],0)

    def test_all_prior_world_vectors_excluded(self):
        splits,excluded=partitions()
        self.assertEqual(len(excluded),1408)
        worlds=[tuple(c) for rows in splits.values() for _,c,_ in rows]
        self.assertEqual(len(set(worlds)),704);self.assertFalse(set(worlds)&set(excluded))
        self.assertEqual({s:len(v) for s,v in splits.items()},{'train':512,'validation':64,'test':64,'ood':64})

    def test_new_metadata_and_auxiliary_split_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            path=build_dataset(Path(temp));meta=json.loads((path/'split-manifest.json').read_text())
            self.assertNotIn('world_coefficients',meta);self.assertNotIn('excluded_coefficients',meta)
            data=json.loads((path/'train-validation.json').read_text())
            for split in ('train','validation'):
                for row in data[split]:
                    for aux in row['auxiliary']:
                        self.assertEqual(aux['world_group'],row['group']);self.assertEqual(aux['split'],split)


@unittest.skipUnless(HAS_TORCH,'Optional torch environment missing')
class CurriculumTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        from aim.memory import Memory
        from aim.research_data import make_record
        memory=Memory(self.path/'memory')
        try:
            self.rows={s:make_record(str(i)*64,[i,1,1] if s!='ood' else [i,1,1,1],
                       'quadratic' if s!='ood' else 'cubic',s,memory,version=EXPERIMENT)
                       for i,s in enumerate(('train','validation','test','ood'),1)}
        finally: memory.close()
        for split in ('train','validation'):
            row=self.rows[split];row['research_contract']=CONTRACT
            row['response']=worked_response(row['world_coefficients']);row['auxiliary']=auxiliary_rows(row)
        self.data={'schema':'aim-curriculum-sft-v1','version':EXPERIMENT,'train':[self.rows['train']],'validation':[self.rows['validation']]}
        write_json(self.path/'train-validation.json',self.data)
        write_json(self.path/'holdout.json',{'schema':'aim-curriculum-holdout-v1','version':EXPERIMENT,
                   'test':[self.rows['test']],'ood':[self.rows['ood']]})
        write_json(self.path/'split-manifest.json',{'version':EXPERIMENT,'train_validation_sha256':file_hash(self.path/'train-validation.json'),
                                                  'holdout_sha256':file_hash(self.path/'holdout.json')})
        self.config={'protocol':EXPERIMENT,'seeds':[17],'steps':1,'evaluation_steps':[1],'batch_size':4,
                     'learning_rate':.001,'weight_decay':.01,'max_new_tokens':8,'deadline_seconds':60,'pad_to':256,
                     'model':{'width':16,'layers':1,'heads':2,'kv_heads':1,'ffn_width':32,'context':256}}

    def test_two_arm_training_isolation_and_real_observation_audit(self):
        from aim.curriculum_train import train
        from aim.curriculum_eval import evaluate
        original_read=Path.read_text
        def guarded(path,*args,**kwargs):
            if path.name in ('holdout.json','split-audit.json'): raise AssertionError('Trainer read heldout labels')
            return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',guarded):
            trained=train(self.config,self.path/'train-validation.json',self.path/'runs')
        frozen=json.loads((trained/'selected-models.json').read_text())
        self.assertTrue(frozen['paired_world_draws_equal'])
        self.assertEqual(frozen['arms']['curriculum']['selections'][0]['task_counts'],{'research':2,'subtract':2})
        compared,audited,metrics=evaluate(trained/'selected-models.json',self.path/'holdout.json',self.path/'runs')
        checks=json.loads((audited/'metrics.json').read_text())['backends']
        self.assertEqual(checks['reference-test']['target_and_observations_pass'],1)
        self.assertEqual(checks['reference-ood']['observation_passes'],1)
        self.assertEqual(checks['reference-ood']['target_and_observations_pass'],0)
        self.assertFalse(metrics['curriculum_promotion_gate'])

    def test_changed_auxiliary_label_or_metadata_rejected(self):
        from aim.curriculum_train import load_data
        data=copy.deepcopy(self.data);data['train'][0]['auxiliary'][0]['response']='{"value":999}'
        path=self.path/'train-validation.json';path.write_text(json.dumps(data))
        meta=json.loads((self.path/'split-manifest.json').read_text());meta['train_validation_sha256']=file_hash(path)
        (self.path/'split-manifest.json').write_text(json.dumps(meta))
        with self.assertRaises(ContractError): load_data(path)
        meta['world_coefficients']={'test':[[1,2,3]]}
        (self.path/'split-manifest.json').write_text(json.dumps(meta))
        with self.assertRaises(ContractError): load_data(path)

    def test_curriculum_resume_preserves_weights_and_task_accounting(self):
        import torch
        from aim.neural import read_checkpoint
        from aim.research_train import train_seed
        from aim.tracking import ROOT
        config={**self.config,'arm':'curriculum','research_contract':CONTRACT,'steps':4,'evaluation_steps':[2,4]}
        args=(self.data,self.path/'train-validation.json')
        protocol=ROOT/'docs/experiments/phase-2a2-protocol.md'
        full=train_seed(config,*args,self.path/'full',17,protocol)
        part=train_seed({**config,'steps':2,'evaluation_steps':[2]},*args,self.path/'partial',17,protocol)
        resumed=train_seed(config,*args,self.path/'resume',17,protocol,resume=part['evaluations'][-1]['checkpoint'])
        left=read_checkpoint(full['evaluations'][-1]['checkpoint']);right=read_checkpoint(resumed['evaluations'][-1]['checkpoint'])
        for key in left['model']: self.assertTrue(torch.equal(left['model'][key],right['model'][key]))
        self.assertTrue(torch.equal(left['sampler_rng'],right['sampler_rng']))
        self.assertEqual(full['task_counts'],resumed['task_counts']);self.assertEqual(full['sampling_digest'],resumed['sampling_digest'])
        self.assertEqual(resumed['initial'],part['initial'])


if __name__=='__main__': unittest.main()
