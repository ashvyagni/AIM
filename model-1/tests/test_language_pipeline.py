"""Phase 3D candidate tests. Written during build; execution deferred by owner."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from aim.contracts import ContractError
from aim.corpus import Corpus, intake
from aim.corpus_fixture import create_fixture
from aim.language_contract import ENCODING, tokenizer_for
from aim.language_data import prepare
from aim.response_encoding import encode_pair
from aim.tokenization import ByteTokenizer, BytePairTokenizer, fit_bpe
from aim.tracking import ROOT, digest, write_json

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None


class ResponseContractTests(unittest.TestCase):
    def test_boundary_merge_is_explicitly_separate(self):
        tokenizer=BytePairTokenizer([[97,98]],{"corpus_hash":"0"*64,"train_hashes":["1"*64],
            "algorithm":"global-pair-count; lexical-ID-ties; document-boundaries; v1"})
        self.assertEqual(tokenizer.encode("ab"),[259])
        encoded=encode_pair(tokenizer,"a","b",8)
        self.assertEqual(encoded["inputs"],[256,97,98])
        self.assertEqual(encoded["targets"],[97,98,257])
        self.assertEqual(encoded["mask"],[False,True,True])

    def test_empty_prompt_unicode_and_context_rejection(self):
        encoded=encode_pair(ByteTokenizer(),"","α",8)
        self.assertEqual(encoded["response_tokens"],3)
        self.assertEqual(encoded["response_bytes"],2)
        self.assertTrue(all(encoded["mask"]))
        with self.assertRaises(ContractError):encode_pair(ByteTokenizer(),"abcd","efgh",4)

    def test_tokenizer_identity_requires_full_spec_not_size(self):
        provenance={"corpus_hash":"0"*64,"train_hashes":["1"*64],"algorithm":"global-pair-count; lexical-ID-ties; document-boundaries; v1"}
        a,b=BytePairTokenizer([[97,98]],provenance),BytePairTokenizer([[98,99]],provenance)
        self.assertEqual(a.vocab_size,b.vocab_size)
        with self.assertRaises(ContractError):tokenizer_for({"tokenizer":a.specification()},b)

    def test_external_dataset_rejects_holdouts_and_unattributed_humans(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.json"
            row={"id":"a","group":"a","prompt":"Q?","response":"A","rights":"fixture", "label_origin":"human"}
            data={"schema":"aim-language-data-v1","version":"fixture","train":[row],
                  "validation":[{**row,"id":"b","group":"b","prompt":"R?"}]}
            path.write_text(json.dumps(data))
            with self.assertRaises(ContractError):prepare("external","sft",path)
            for split in ("train","validation"):data[split][0]["annotation_batch"]="fixture-attribution"
            data["test"]=[];path.write_text(json.dumps(data))
            with self.assertRaises(ContractError):prepare("external","sft",path)


@unittest.skipUnless(HAS_TORCH,"torch/numpy required")
class LanguageTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.config=json.loads((ROOT/"configs/language-sft.json").read_text())
        self.config.update(steps=4)
        self.config["model"]={"width":16,"layers":1,"heads":2,"kv_heads":1,"ffn_width":32,"context":128}
        corpus=Corpus(intake(create_fixture(self.root/"fixture"),self.root/"intake"))
        self.bpe=fit_bpe(corpus,8)

    def test_padding_and_response_mask_independence(self):
        import torch
        from aim.neural import CausalLM
        from aim.language_contract import model_config
        from aim.response_encoding import scores
        from aim.training import seed_all
        seed_all(17);model=CausalLM(model_config(self.config["model"],self.bpe))
        a,counts,_=scores(model,self.bpe,[("2+3=","5")])
        b,_,_=scores(model,self.bpe,[("2+3=","5"),("A longer prompt","response")])
        self.assertTrue(torch.allclose(a[0],b[0],atol=1e-6))
        self.assertEqual(int(counts[0]),len(self.bpe.encode("5"))+1)

    def test_bpe_pretraining_bridge_and_loader(self):
        from aim.pretrain import pretrain
        from aim.language_training import train
        from aim.neural import load_lm
        corpus=Corpus(intake(create_fixture(self.root/"bridge"),self.root/"intake"))
        bpe=fit_bpe(corpus,8)
        config={k:self.config[k] for k in ("seed","steps","batch_size","learning_rate","weight_decay","checkpoint_every","model")}
        config.update(steps=2,eval_batches=1)
        source=pretrain(config,corpus.path,self.root/"pretrain",bpe)
        run=train({**self.config,"steps":2},self.root/"language",initialize=source/"checkpoint.pt")
        model,tokenizer,record=load_lm(run/"checkpoint.pt")
        self.assertEqual(tokenizer.specification(),bpe.specification())
        self.assertEqual(model.cfg.vocab_size,bpe.vocab_size)
        self.assertEqual(record["encoding"],ENCODING)
        self.assertEqual(record["kind"],"tokenized_lm")

    def test_all_separate_stages_exact_resume(self):
        from aim.language_training import train
        from aim.language_contract import read_source
        from aim.checkpoint_bundle import tree_hash
        parent=None
        for stage in ("sft","preference","rlvr"):
            config={**self.config,"stage":stage}
            full=train(config,self.root/"runs",self.bpe,initialize=parent)
            partial=train({**config,"steps":2},self.root/"runs",self.bpe,initialize=parent)
            resumed=train(config,self.root/"runs",self.bpe,resume=partial/"checkpoint.pt")
            a,b=read_source(full/"checkpoint.pt"),read_source(resumed/"checkpoint.pt")
            for key in ("model","reference","optimizer","torch_rng","step","scored_tokens","tokenizer","encoding_manifest_hash","dataset_hash"):
                self.assertEqual(tree_hash(a[key]),tree_hash(b[key]),(stage,key))
            self.assertTrue((full/"checkpoint-step-000002.pt").exists())
            parent=full/"checkpoint.pt"

    def test_stage_references_are_frozen(self):
        from aim.language_training import train
        from aim.language_contract import read_source
        from aim.checkpoint_bundle import tree_hash
        sft=train({**self.config,"steps":2},self.root/"runs",self.bpe)
        parent=read_source(sft/"checkpoint.pt")
        preference=train({**self.config,"stage":"preference"},self.root/"runs",initialize=sft/"checkpoint.pt")
        child=read_source(preference/"checkpoint.pt")
        self.assertEqual(tree_hash(parent["model"]),tree_hash(child["reference"]))
        self.assertNotEqual(tree_hash(child["model"]),tree_hash(child["reference"]))

    def test_resume_rejects_objective_tokenizer_and_config_changes(self):
        from aim.language_training import train
        run=train({**self.config,"steps":2},self.root/"runs",self.bpe)
        with self.assertRaises(ContractError):train({**self.config,"stage":"preference"},self.root/"runs",resume=run/"checkpoint.pt")
        with self.assertRaises(ContractError):train(self.config,self.root/"runs",ByteTokenizer(),resume=run/"checkpoint.pt")
        with self.assertRaises(ContractError):train({**self.config,"learning_rate":.2},self.root/"runs",resume=run/"checkpoint.pt")

    def test_symbolic_adapter_and_wrong_numerical_adapter(self):
        from aim.language_training import train
        from aim.symbolic_loop import SymbolicTransformerResearcher
        from aim.researcher import TransformerResearcher
        config={**self.config,"task":"symbolic","steps":1,"model":{**self.config["model"],"context":256}}
        run=train(config,self.root/"runs",self.bpe)
        self.assertEqual(SymbolicTransformerResearcher(run/"checkpoint.pt").tokenizer.specification(),self.bpe.specification())
        with self.assertRaises(ContractError):TransformerResearcher(run/"checkpoint.pt")

    def test_legacy_byte_training_checkpoint_import(self):
        from aim.training import train as legacy_train
        from aim.language_training import train
        from aim.language_contract import read_source
        config={k:v for k,v in self.config.items() if k not in {"checkpoint_every","eval_batch_size"}}
        source=legacy_train({**config,"steps":2},self.root/"legacy")
        run=train({**self.config,"steps":2},self.root/"runs",initialize=source/"checkpoint.pt")
        self.assertEqual(read_source(run/"checkpoint.pt")["tokenizer"],ByteTokenizer().specification())

    def test_strict_generation_special_utf8_and_eos(self):
        import torch
        from types import SimpleNamespace
        from aim.generation import generate
        class Scripted(torch.nn.Module):
            def __init__(self,ids):
                super().__init__();self.ids=iter(ids);self.cfg=SimpleNamespace(context=32,vocab_size=259)
            def forward(self,x):
                value=next(self.ids);logits=torch.full((1,x.shape[1],259),-10.)
                logits[0,-1,value]=10.;return logits
        for ids,termination,error in (([65,257],"eos",None),([256],"invalid_special_token",True),([195,257],"invalid_utf8",True),([65,66],"token_budget",None)):
            model=Scripted(ids)
            result=generate(model,ByteTokenizer(),"Q",2)
            self.assertEqual(result["termination"],termination)
            self.assertEqual(result["error"] is None,error is None)
            self.assertTrue(model.training)


if __name__=="__main__":unittest.main()
