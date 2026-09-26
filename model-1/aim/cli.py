import argparse
import json
from pathlib import Path

from .tracking import Run


def main():
    parser=argparse.ArgumentParser(description="AIM Model-1 engineering prototype")
    sub=parser.add_subparsers(dest="command",required=True)
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
    if args.command=="symbolic-loop":
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
