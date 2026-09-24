import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from aim.contracts import ContractError
from aim.finite_difference import EXPERIMENT, CONTRACT, worked_response, parse_output, diagnostics
from aim.research_format import VERSION, target_response
from aim.comparison_data import partitions
from aim.research_data import world_partitions

HAS_TORCH=importlib.util.find_spec('torch') is not None


class FiniteDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.prompt=json.dumps({"evidence":{"E0":[[0,1],[1,4],[2,9]]},"x":3})
        self.aliases={"E0":"actual-source-span"}

    def test_worked_numbers_and_citations(self):
        text=worked_response([1,2,1])
        self.assertEqual(parse_output(text,self.aliases,CONTRACT)[0].evidence_ids,('actual-source-span',))
        self.assertTrue(diagnostics(text,self.prompt,self.aliases,CONTRACT)["all_steps_correct"])

    def test_wrong_step_is_diagnostic_failure_not_silent_repair(self):
        text=worked_response([1,2,1]).replace('"d2":2','"d2":4')
        self.assertEqual(parse_output(text,self.aliases,CONTRACT)[0].coefficients,(1,2,1))
        score=diagnostics(text,self.prompt,self.aliases,CONTRACT)
        self.assertFalse(score["d2_correct"]);self.assertFalse(score["all_steps_correct"])
        self.assertEqual(score["coefficient_observation_agreement"],[True]*3)

    def test_worked_rejects_duplicate_nonfinite_and_invented_alias(self):
        original=worked_response([1,2,1])
        for text in (original.replace('"d1":3','"d1":3,"d1":3'),original.replace('"d1":3','"d1":NaN'),
                     original.replace('E0','E9'),target_response([1,2,1])):
            with self.subTest(text=text),self.assertRaises(ContractError): parse_output(text,self.aliases,CONTRACT)

    def test_plain_contract_unchanged_and_invalid_counts_as_failure(self):
        result=diagnostics(target_response([1,2,1]),self.prompt,self.aliases,VERSION)
        self.assertIsNone(result["all_steps_correct"])
        self.assertEqual(diagnostics('',self.prompt,self.aliases,CONTRACT)["coefficient_observation_agreement"],[False]*3)

    def test_worlds_exclude_all_old_splits_by_coefficient_vector(self):
        splits,excluded=partitions()
        old={tuple(c) for rows in world_partitions().values() for _,c,_ in rows}
        fresh=[tuple(c) for rows in splits.values() for _,c,_ in rows]
        self.assertEqual(old,{tuple(c) for c in excluded})
        self.assertFalse(old&set(fresh));self.assertEqual(len(set(fresh)),704)
        self.assertEqual({s:len(rows) for s,rows in splits.items()},{"train":512,"validation":64,"test":64,"ood":64})
        self.assertEqual(splits,partitions()[0])


@unittest.skipUnless(HAS_TORCH,'Optional torch environment missing')
class ComparisonIntegrationTests(unittest.TestCase):
    def test_dataset_metadata_contains_no_holdout_coefficients(self):
        from aim.comparison_data import build_dataset
        with tempfile.TemporaryDirectory() as temp:
            path=build_dataset(Path(temp))
            metadata=json.loads((path/'split-manifest.json').read_text())
            self.assertNotIn('world_coefficients',metadata)
            self.assertNotIn('test',metadata)
            audit=json.loads((path/'split-audit.json').read_text())
            self.assertEqual(len(audit['world_coefficients']['test']),64)
            self.assertEqual(len(audit['world_coefficients']['ood']),64)

    def test_padding_preserves_supervised_tokens_and_scores(self):
        import torch
        from aim.neural import CausalLM, ModelConfig, ByteTokenizer
        from aim.training import sequence_scores, seed_all
        seed_all(17)
        model=CausalLM(ModelConfig(width=16,layers=1,heads=2,kv_heads=1,ffn_width=32,context=256))
        pairs=[('prompt','answer'),('p','a')]
        a,ac=sequence_scores(model,ByteTokenizer(),pairs)
        b,bc=sequence_scores(model,ByteTokenizer(),pairs,pad_to=256)
        self.assertTrue(torch.equal(ac,bc));self.assertTrue(torch.allclose(a,b,atol=1e-5))
        with self.assertRaises(ContractError): sequence_scores(model,ByteTokenizer(),pairs,pad_to=2)

    def test_two_arm_freeze_and_actual_controller_evaluation(self):
        from aim.comparison_train import train, arm_data
        from aim.comparison_eval import evaluate
        from aim.research_data import make_record
        from aim.memory import Memory
        from aim.tracking import write_json, file_hash
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);memory=Memory(path/'memory')
            try:
                rows={s:make_record(str(i)*64,[i,1,1] if s!='ood' else [i,1,1,1],
                      'quadratic' if s!='ood' else 'cubic',s,memory,version=EXPERIMENT)
                      for i,s in enumerate(('train','validation','test','ood'),1)}
            finally: memory.close()
            data={"schema":"aim-research-comparison-sft-v1","version":EXPERIMENT,
                  "train":[rows['train']],"validation":[rows['validation']]}
            plain,_=arm_data(data,'plain');worked,_=arm_data(data,'worked')
            self.assertEqual(plain['train'][0]['prompt'],worked['train'][0]['prompt'])
            self.assertNotEqual(plain['train'][0]['response'],worked['train'][0]['response'])
            write_json(path/'train-validation.json',data)
            write_json(path/'holdout.json',{"schema":"aim-research-comparison-holdout-v1","version":EXPERIMENT,
                                         "test":[rows['test']],"ood":[rows['ood']]})
            write_json(path/'split-manifest.json',{"train_validation_sha256":file_hash(path/'train-validation.json'),
                                                  "holdout_sha256":file_hash(path/'holdout.json')})
            config={"protocol":EXPERIMENT,"seeds":[17],"steps":1,"evaluation_steps":[1],"batch_size":1,
                    "learning_rate":.001,"weight_decay":.01,"max_new_tokens":8,"deadline_seconds":60,"pad_to":256,
                    "model":{"width":16,"layers":1,"heads":2,"kv_heads":1,"ffn_width":32,"context":256}}
            original_read=Path.read_text
            def isolated_read(file,*args,**kwargs):
                if file.name in ('holdout.json','split-audit.json'): raise AssertionError('Trainer opened label-bearing holdout/audit file')
                return original_read(file,*args,**kwargs)
            with patch.object(Path,'read_text',isolated_read):
                trained=train(config,path/'train-validation.json',path/'runs')
            selected=json.loads((trained/'selected-models.json').read_text())
            self.assertTrue(selected['paired_initial_weights_equal'])
            for arm in selected['arms'].values(): self.assertEqual(arm['selections'][0]['processed_positions'],256)
            evaluated,report=evaluate(trained/'selected-models.json',path/'holdout.json',path/'runs')
            self.assertEqual(report['backends']['reference-test']['verified_worlds'],1)
            self.assertEqual(report['unbacked_verified_claims'],0)
            self.assertFalse(report['worked_promotion_gate'])
            metadata=json.loads((path/'split-manifest.json').read_text());metadata['world_coefficients']={'test':[[3,1,1]]}
            (path/'split-manifest.json').write_text(json.dumps(metadata))
            with self.assertRaises(ContractError): train(config,path/'train-validation.json',path/'runs')
            (path/'holdout.json').write_text('{}')
            with self.assertRaises(ContractError): evaluate(trained/'selected-models.json',path/'holdout.json',path/'runs')


if __name__=='__main__': unittest.main()
