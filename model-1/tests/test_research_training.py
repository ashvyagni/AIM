import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from aim.contracts import ContractError
    from aim.neural import read_checkpoint
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

    def selection(self, path):
        return json.loads((path/"selected-models.json").read_text())["selections"][0]

    def assert_tree_equal(self, left, right):
        if isinstance(left,torch.Tensor): self.assertTrue(torch.equal(left,right))
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for key in left: self.assert_tree_equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right): self.assert_tree_equal(a,b)
        else: self.assertEqual(left,right)

    def test_resume_matches_uninterrupted_optimizer_rng_and_selection(self):
        path,_,_=self.data()
        full={**self.config,"steps":4,"evaluation_steps":[2,4]}
        partial={**full,"steps":2,"evaluation_steps":[2]}
        a=self.selection(train_experiment(full,path,self.path/"full"))
        b=self.selection(train_experiment(partial,path,self.path/"partial"))
        c=self.selection(train_experiment(full,path,self.path/"resumed",resume=b["evaluations"][-1]["checkpoint"]))
        before=read_checkpoint(a["evaluations"][-1]["checkpoint"])
        after=read_checkpoint(c["evaluations"][-1]["checkpoint"])
        for key in ("model","optimizer","torch_rng","sampler_rng","step"):
            self.assert_tree_equal(before[key],after[key])
        self.assertEqual(c["initial"],b["initial"])
        self.assertEqual([r["step"] for r in c["evaluations"]],[0,2,4])
        self.assertEqual(a["selected"]["step"],c["selected"]["step"])
        for x,y in zip(a["evaluations"],c["evaluations"]):
            for key in ("response_token_nll","valid_rate","prediction_agreement_rate"):
                self.assertEqual(x[key],y[key])

    def test_resume_rejects_changed_optimizer_seed_schedule_or_dataset(self):
        path,_,_=self.data()
        base=self.selection(train_experiment(self.config,path,self.path/"partial"))
        checkpoint=base["evaluations"][-1]["checkpoint"]
        config={**self.config,"steps":2,"evaluation_steps":[1,2]}
        for changed in ({"learning_rate":0.002},{"seeds":[23]},{"evaluation_steps":[2]},{"steps":1,"evaluation_steps":[1]}):
            with self.subTest(changed=changed),self.assertRaises(ContractError):
                train_experiment({**config,**changed},path,self.path/"bad",resume=checkpoint)
        data=json.loads(path.read_text());data["train"][0]["response"]+=' '
        path.write_text(json.dumps(data))
        (path.parent/"split-manifest.json").write_text(json.dumps({"train_validation_sha256":file_hash(path)}))
        with self.assertRaises(ContractError): train_experiment(config,path,self.path/"bad-data",resume=checkpoint)

    def test_resume_rejects_tampered_selection_history(self):
        path,_,_=self.data()
        base=self.selection(train_experiment(self.config,path,self.path/"partial"))
        Path(base["initial"]["validation_path"]).write_text('{}')
        with self.assertRaises(ContractError):
            train_experiment({**self.config,"steps":2,"evaluation_steps":[1,2]},path,self.path/"bad",
                             resume=base["evaluations"][-1]["checkpoint"])

    def test_legacy_checkpoint_reconstructs_original_history(self):
        path,_,_=self.data()
        base=self.selection(train_experiment(self.config,path,self.path/"partial"))
        checkpoint=Path(base["evaluations"][-1]["checkpoint"])
        record=read_checkpoint(checkpoint);record.pop("evaluation_history")
        legacy=checkpoint.parent/"legacy.pt"
        torch.save(record,legacy)
        write_json(Path(str(legacy)+".sha256.json"),{"sha256":file_hash(legacy)})
        resumed=self.selection(train_experiment({**self.config,"steps":2,"evaluation_steps":[1,2]},path,
                                               self.path/"resumed",resume=legacy))
        self.assertEqual(resumed["initial"]["sha256"],base["initial"]["sha256"])
        self.assertEqual([r["step"] for r in resumed["evaluations"]],[0,1,2])

    def test_deadline_checkpoint_preserves_completed_boundary(self):
        path,_,_=self.data()
        config={**self.config,"steps":4,"evaluation_steps":[2,4],"deadline_seconds":1}
        with patch("aim.research_train.time.monotonic",side_effect=[0,0,0,2]),self.assertRaises(TimeoutError):
            train_experiment(config,path,self.path/"interrupted")
        emergency=list((self.path/"interrupted").rglob("emergency-deadline.pt"))[0]
        self.assertEqual(read_checkpoint(emergency)["step"],2)
        result=self.selection(train_experiment({**config,"deadline_seconds":60},path,self.path/"resume",resume=emergency))
        self.assertEqual([r["step"] for r in result["evaluations"]],[0,2,4])
        self.assertTrue(any(json.loads(p.read_text())["status"]=="FAILED" for p in (self.path/"interrupted").rglob("status.json")))


if __name__=='__main__': unittest.main()
