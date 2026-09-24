import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from aim.neural import load_lm
    from aim.research_data import make_record, world_id
    from aim.memory import Memory
    from aim.research_format import target_response, VERSION
    from aim.research_train import score_generation, selection_key, train_experiment, validate_config
    from aim.tracking import file_hash, write_json


@unittest.skipUnless(HAS_TORCH,"Optional torch environment missing")
class ResearchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.config={"protocol":VERSION,"seeds":[17],"steps":1,"evaluation_steps":[1],"batch_size":1,
                     "learning_rate":0.001,"weight_decay":0.01,"max_new_tokens":8,"deadline_seconds":60,
                     "model":{"width":16,"layers":1,"heads":2,"kv_heads":1,"ffn_width":32,"context":256}}

    def data(self):
        memory=Memory(self.path/"memory")
        try:
            train=make_record(world_id([1,2,1]),[1,2,1],"quadratic","train",memory)
            validation=make_record(world_id([2,1,0]),[2,1,0],"linear","validation",memory)
        finally: memory.close()
        path=self.path/"train-validation.json"
        write_json(path,{"schema":"aim-research-sft-v1","version":VERSION,"train":[train],"validation":[validation]})
        write_json(self.path/"split-manifest.json",{"train_validation_sha256":file_hash(path),"holdout_sha256":"unopened-holdout-test"})
        return path,train,validation

    def test_score_uses_actual_next_measurement(self):
        _,row,_=self.data()
        self.assertTrue(score_generation(target_response([1,2,1]),row)["agrees"])
        self.assertFalse(score_generation(target_response([1,2,0]),row)["agrees"])
        self.assertFalse(score_generation('',row)["valid"])

    def test_selection_priorities_and_earliest_tie(self):
        base={"prediction_agreement_rate":0.5,"valid_rate":0.9,"response_token_nll":0.3}
        self.assertGreater(selection_key(base,300),selection_key(base,600))
        self.assertGreater(selection_key({**base,"prediction_agreement_rate":0.6},600),selection_key(base,300))

    def test_training_freezes_checkpoints_without_opening_holdout(self):
        path,_,_=self.data()
        result=train_experiment(self.config,path,self.path/"runs")
        freeze=json.loads((result/"selected-models.json").read_text())
        self.assertEqual(freeze["holdout_sha256"],"unopened-holdout-test")
        self.assertEqual(len(freeze["selections"]),1)
        initial=freeze["selections"][0]["initial"]
        model,_,record=load_lm(initial["checkpoint"])
        self.assertEqual(record["research_contract"],VERSION)
        self.assertEqual(record["step"],0)
        self.assertEqual(file_hash(initial["checkpoint"]),initial["sha256"])

    def test_tampered_training_file_rejected(self):
        path,_,_=self.data()
        path.write_text(path.read_text()+"\n")
        from aim.contracts import ContractError
        with self.assertRaises(ContractError): train_experiment(self.config,path,self.path/"runs")

    def test_update_budget_rejected(self):
        config=copy.deepcopy(self.config);config["steps"]=2000
        from aim.contracts import ContractError
        with self.assertRaises(ContractError): validate_config(config)


if __name__=='__main__': unittest.main()
