import argparse
import json
from pathlib import Path

from .tracking import Run


def main():
    parser=argparse.ArgumentParser(description="AIM Model-1 engineering prototype")
    sub=parser.add_subparsers(dest="command",required=True)
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
    if args.command=="loop":
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
