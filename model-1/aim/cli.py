import argparse
import json
from pathlib import Path

from .tracking import Run


def main():
    parser=argparse.ArgumentParser(description="AIM Model-1 engineering prototype")
    sub=parser.add_subparsers(dest="command",required=True)
    language=sub.add_parser("language-train")
    language.add_argument("--config",type=Path,required=True)
    language.add_argument("--tokenizer",type=Path)
    language.add_argument("--initialize",type=Path)
    language.add_argument("--resume",type=Path)
    language.add_argument("--runs",type=Path,default=Path("runs"))
    generate=sub.add_parser("language-generate")
    generate.add_argument("--checkpoint",type=Path,required=True)
    generate.add_argument("--prompt-file",type=Path,required=True)
    generate.add_argument("--max-new-tokens",type=int,default=64)
    generate.add_argument("--runs",type=Path,default=Path("runs"))
    for name in ("language-inspect","language-stage-plan"):
        item=sub.add_parser(name)
        item.add_argument("--checkpoint",type=Path,required=True)
    node=sub.add_parser("node-audit")
    node.add_argument("--node-id",required=True)
    node.add_argument("--workspace",type=Path,default=Path("."))
    node.add_argument("--runs",type=Path,default=Path("runs"))
    probe=sub.add_parser("storage-probe")
    probe.add_argument("--node",type=Path,required=True)
    probe.add_argument("--mebibytes",type=int,default=4)
    probe.add_argument("--runs",type=Path,default=Path("runs"))
    fleet=sub.add_parser("fleet-plan")
    fleet.add_argument("--nodes",type=Path,nargs="+",required=True)
    fleet.add_argument("--policy",type=Path)
    fleet.add_argument("--config",type=Path,default=Path("configs/distributed-pretrain-smoke.json"))
    fleet.add_argument("--max-age-seconds",type=int,default=86400)
    fleet.add_argument("--runs",type=Path,default=Path("runs"))
    ca=sub.add_parser("corpus-audit")
    ca.add_argument("--corpus",type=Path,required=True)
    ca.add_argument("--config",type=Path)
    ca.add_argument("--runs",type=Path,default=Path("runs"))
    for name in ("corpus-gate","corpus-export-reviewed"):
        cr=sub.add_parser(name)
        cr.add_argument("--corpus",type=Path,required=True)
        cr.add_argument("--audit",type=Path,required=True)
        cr.add_argument("--review",type=Path,required=True)
        cr.add_argument("--runs",type=Path,default=Path("runs"))
    ar=sub.add_parser("corpus-release-audit")
    ar.add_argument("--release",type=Path,required=True)
    ar.add_argument("--parent-corpus",type=Path,required=True)
    ar.add_argument("--audit",type=Path,required=True)
    bound=sub.add_parser("audit-seal")
    bound.add_argument("--input",type=Path,required=True)
    bound.add_argument("--output",type=Path,required=True)
    distributed=sub.add_parser("distributed-pretrain")
    distributed.add_argument("--corpus",type=Path,required=True)
    distributed.add_argument("--config",type=Path,default=Path("configs/distributed-pretrain-smoke.json"))
    distributed.add_argument("--tokenizer",type=Path)
    distributed.add_argument("--ranks",type=int,choices=(1,2,4),default=2)
    distributed.add_argument("--resume",type=Path)
    distributed.add_argument("--runs",type=Path,default=Path("runs"))
    distributed.add_argument("--deadline",type=int,default=180)
    preflight=sub.add_parser("distributed-preflight")
    preflight.add_argument("--corpus",type=Path,required=True)
    preflight.add_argument("--config",type=Path,default=Path("configs/distributed-pretrain-smoke.json"))
    preflight.add_argument("--tokenizer",type=Path)
    preflight.add_argument("--ranks",type=int,default=2)
    preflight.add_argument("--runs",type=Path,default=Path("runs"))
    checkpoint=sub.add_parser("checkpoint-inspect")
    checkpoint.add_argument("path",type=Path)
    checkpoint.add_argument("--corpus",type=Path)
    export=sub.add_parser("distributed-export")
    export.add_argument("--checkpoint",type=Path,required=True)
    export.add_argument("--runs",type=Path,default=Path("runs"))
    corpus=sub.add_parser("corpus-intake")
    corpus.add_argument("--manifest", type=Path, required=True)
    corpus.add_argument("--runs", type=Path, default=Path("runs"))
    fixture=sub.add_parser("corpus-fixture")
    fixture.add_argument("--output", type=Path, required=True)
    tokenizers=sub.add_parser("tokenizer-compare")
    tokenizers.add_argument("--corpus", type=Path, required=True)
    tokenizers.add_argument("--merges", type=int, default=32)
    tokenizers.add_argument("--runs", type=Path, default=Path("runs"))
    pretrain=sub.add_parser("pretrain")
    pretrain.add_argument("--corpus", type=Path, required=True)
    pretrain.add_argument("--config", type=Path, default=Path("configs/pretrain-smoke.json"))
    pretrain.add_argument("--tokenizer", type=Path)
    pretrain.add_argument("--resume", type=Path)
    pretrain.add_argument("--runs", type=Path, default=Path("runs"))
    symbolic=sub.add_parser("symbolic-loop")
    symbolic.add_argument("--case", type=Path)
    symbolic.add_argument("--researcher-checkpoint", type=Path)
    symbolic.add_argument("--judge", type=Path)
    symbolic.add_argument("--check-cost", type=float, default=.2)
    symbolic.add_argument("--max-actions", type=int, default=3)
    symbolic.add_argument("--runs", type=Path, default=Path("runs"))
    audit=sub.add_parser("symbolic-audit")
    audit.add_argument("path", type=Path)
    symbolic_judge=sub.add_parser("symbolic-judge-train")
    symbolic_judge.add_argument("--config",type=Path,default=Path("configs/symbolic-judge.json"))
    symbolic_judge.add_argument("--resume",type=Path)
    symbolic_judge.add_argument("--runs",type=Path,default=Path("runs"))
    symbolic_eval=sub.add_parser("symbolic-evaluate")
    symbolic_eval.add_argument("--suite",type=Path,default=Path("eval/symbolic-v1.json"))
    symbolic_eval.add_argument("--researcher-checkpoint",type=Path)
    symbolic_eval.add_argument("--runs",type=Path,default=Path("runs"))
    loop=sub.add_parser("loop")
    loop.add_argument("--case",type=Path,default=Path("data/demo.json"))
    loop.add_argument("--config",type=Path,default=Path("configs/loop.json"))
    loop.add_argument("--judge",type=Path)
    loop.add_argument("--researcher-checkpoint",type=Path)
    loop.add_argument("--runs",type=Path,default=Path("runs"))
    train=sub.add_parser("train")
    train.add_argument("--config",type=Path,required=True)
    train.add_argument("--initialize",type=Path)
    train.add_argument("--resume",type=Path)
    train.add_argument("--runs",type=Path,default=Path("runs"))
    ev=sub.add_parser("evaluate")
    ev.add_argument("--suite",type=Path,default=Path("eval/suite-v1.json"))
    ev.add_argument("--judge",type=Path)
    ev.add_argument("--runs",type=Path,default=Path("runs"))
    bench=sub.add_parser("benchmark")
    bench.add_argument("--config",type=Path,default=Path("configs/benchmark.json"))
    bench.add_argument("--runs",type=Path,default=Path("runs"))
    args=parser.parse_args()
    if args.command=="language-train":
        from .corpus import read_json
        from .tokenization import tokenizer_from_spec
        from .language_training import train as language_train
        tokenizer=tokenizer_from_spec(read_json(args.tokenizer)) if args.tokenizer else None
        print(language_train(read_json(args.config),args.runs,tokenizer,args.initialize,args.resume))
    elif args.command=="language-generate":
        from .corpus import require
        from .language_artifacts import generate_run
        require(args.prompt_file.is_file() and args.prompt_file.stat().st_size<=262144,"prompt file exceeds byte budget")
        print(generate_run(args.checkpoint,args.prompt_file.read_text(encoding="utf-8"),args.max_new_tokens,args.runs))
    elif args.command in {"language-inspect","language-stage-plan"}:
        from .language_artifacts import describe,stage_plan
        result=describe(args.checkpoint) if args.command=="language-inspect" else stage_plan(args.checkpoint)
        print(json.dumps(result,indent=2))
    elif args.command=="node-audit":
        from .hardware_audit import collect_run
        print(collect_run(args.node_id,args.workspace,args.runs))
    elif args.command=="storage-probe":
        from .hardware_audit import storage_probe
        print(storage_probe(args.node,args.runs,args.mebibytes))
    elif args.command=="fleet-plan":
        from .fleet_plan import plan_run
        print(plan_run(args.nodes,args.config,args.runs,args.policy,args.max_age_seconds))
    elif args.command=="corpus-audit":
        from .corpus import read_json
        from .corpus_review import audit_run
        print(audit_run(args.corpus,args.runs,read_json(args.config) if args.config else None))
    elif args.command in {"corpus-gate","corpus-export-reviewed"}:
        from .corpus import Corpus,read_json
        from .corpus_release import gate,export_subset
        from .tracking import write_json
        if args.command=="corpus-export-reviewed":
            print(export_subset(args.corpus,args.audit,args.review,args.runs))
        else:
            with Run(args.runs,"corpus-release-gate",{},[args.corpus,args.audit,args.review]) as run:
                result=gate(Corpus(args.corpus),read_json(args.audit,16*1024*1024),read_json(args.review))
                write_json(run.path/"gate.json",result)
            print(run.path)
            if result["blockers"]: raise SystemExit(2)
    elif args.command=="corpus-release-audit":
        from .corpus_release import audit_release
        print(json.dumps(audit_release(args.release,args.parent_corpus,args.audit),indent=2))
    elif args.command=="audit-seal":
        from .audit_contracts import seal
        from .corpus import read_json,require
        from .tracking import write_json
        record=read_json(args.input)
        require(isinstance(record,dict) and record.get("schema") in {"aim-lab-policy-v1","aim-corpus-review-decisions-v1"},"only operator policy/review records can be sealed")
        write_json(args.output,seal(record))
        print("Content hash recorded; this does not approve or certify the declarations.")
    elif args.command in {"distributed-pretrain","distributed-preflight"}:
        from .corpus import read_json
        from .tokenization import tokenizer_from_spec
        from .distributed_jobs import launch,preflight
        from .tracking import write_json
        config=read_json(args.config)
        tokenizer=tokenizer_from_spec(read_json(args.tokenizer)) if args.tokenizer else None
        if args.command=="distributed-pretrain":
            print(launch(config,args.corpus,args.runs,args.ranks,tokenizer,args.resume,deadline=args.deadline))
        else:
            with Run(args.runs,"distributed-preflight",{"config":config,"world_size":args.ranks},[args.corpus,args.config]) as run:
                write_json(run.path/"preflight.json",preflight(config,args.corpus,tokenizer,args.ranks))
            print(run.path)
    elif args.command=="checkpoint-inspect":
        from .checkpoint_bundle import load_bundle,scan
        from .distributed_jobs import audit_checkpoint
        if (args.path/"bundle.json").is_file():
            result=audit_checkpoint(args.path,args.corpus) if args.corpus else load_bundle(args.path)[2]
        else:
            result=scan(args.path)
        print(json.dumps(result,indent=2))
    elif args.command=="distributed-export":
        from .distributed_jobs import export_initialization
        print(export_initialization(args.checkpoint,args.runs))
    elif args.command=="corpus-fixture":
        from .corpus_fixture import create_fixture
        print(create_fixture(args.output))
    elif args.command=="corpus-intake":
        from .corpus import intake
        print(intake(args.manifest,args.runs))
    elif args.command=="tokenizer-compare":
        from .corpus import Corpus
        from .tokenization import ByteTokenizer,fit_bpe,compare_tokenizers
        from .tracking import write_json
        with Run(args.runs,"tokenizer-compare",{"merges":args.merges},[args.corpus]) as run:
            corpus=Corpus(args.corpus)
            tokenizers={"byte":ByteTokenizer(),"bpe":fit_bpe(corpus,args.merges)}
            for name,tokenizer in tokenizers.items():
                write_json(run.path/(name+"-tokenizer.json"),tokenizer.specification())
            write_json(run.path/"comparison.json",compare_tokenizers(corpus,tokenizers))
        print(run.path)
    elif args.command=="pretrain":
        from .corpus import read_json
        from .tokenization import tokenizer_from_spec
        from .pretrain import pretrain as train_pretraining
        tokenizer=tokenizer_from_spec(read_json(args.tokenizer)) if args.tokenizer else None
        print(train_pretraining(read_json(args.config),args.corpus,args.runs,tokenizer,args.resume))
    elif args.command=="symbolic-loop":
        from .symbolic_data import symbolic_case
        from .symbolic_loop import SymbolicController, SymbolicTransformerResearcher
        case=json.loads(args.case.read_text()) if args.case else symbolic_case()
        inputs=[p for p in (args.case,args.researcher_checkpoint,args.judge) if p]
        with Run(args.runs,"symbolic-loop",{"case":case,"max_actions":args.max_actions,
                 "researcher_checkpoint":str(args.researcher_checkpoint) if args.researcher_checkpoint else None,
                 "judge_checkpoint":str(args.judge) if args.judge else None,"check_cost":args.check_cost},inputs) as run:
            researcher=SymbolicTransformerResearcher(args.researcher_checkpoint) if args.researcher_checkpoint else None
            from .symbolic_judge import CalibratedSymbolicJudge
            judge=CalibratedSymbolicJudge(args.judge,args.check_cost) if args.judge else None
            state=SymbolicController(researcher=researcher,judge=judge,max_actions=args.max_actions).run(case,run)
            print(state.final_response)
        print(f"Artifacts: {run.path}")
    elif args.command=="symbolic-audit":
        from .symbolic_loop import audit_symbolic_run
        print(json.dumps(audit_symbolic_run(args.path),indent=2))
    elif args.command=="symbolic-judge-train":
        from .symbolic_judge import train_symbolic_judge
        print(train_symbolic_judge(json.loads(args.config.read_text()),args.runs,args.resume))
    elif args.command=="symbolic-evaluate":
        from .symbolic_eval import evaluate_symbolic
        path, metrics=evaluate_symbolic(args.suite,args.runs,args.researcher_checkpoint)
        print(json.dumps({"artifacts":str(path),"metrics":metrics},indent=2))
    elif args.command=="loop":
        from .controller import Controller
        from .judge import CalibratedJudge
        from .researcher import TransformerResearcher
        config=json.loads(args.config.read_text())
        inputs=[args.case,args.config]+[p for p in (args.judge,args.researcher_checkpoint) if p]
        with Run(args.runs,"loop",config,inputs) as run:
            judge=CalibratedJudge(args.judge) if args.judge else None
            researcher=TransformerResearcher(args.researcher_checkpoint,max_new_tokens=80) if args.researcher_checkpoint else None
            state=Controller(researcher=researcher,judge=judge,max_actions=config["max_actions"],timeout=config["timeout_seconds"]).run(json.loads(args.case.read_text()),run)
            print(state.final_response)
        print(f"Artifacts: {run.path}")
    elif args.command=="train":
        from .training import train as train_model
        path=train_model(json.loads(args.config.read_text()),args.runs,initialize=args.initialize,resume=args.resume)
        print(path)
    elif args.command=="evaluate":
        from .evaluation import evaluate
        path,metrics=evaluate(args.suite,args.runs,args.judge)
        print(json.dumps({"artifacts":str(path),"passed":metrics["passed"],"total":metrics["total"]},indent=2))
        if metrics["passed"]!=metrics["total"]: raise SystemExit(1)
    elif args.command=="benchmark":
        from .benchmark import benchmark
        path,metrics=benchmark(json.loads(args.config.read_text()),args.runs)
        print(json.dumps({"artifacts":str(path),"metrics":metrics},indent=2))


if __name__=="__main__": main()
