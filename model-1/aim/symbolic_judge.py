"""Separate small symbolic Judge: proper-score fitting and calibration-only scaling.

Forecasts checker PASS, not universal mathematical truth. Features contain no
checker result or normal form. This syntactic baseline is intentionally weak.
"""
from pathlib import Path

from .contracts import ContractError, Decision, finite_number
from .symbolic_loop import TARGET
from .tracking import Run, digest, file_hash, write_json

FEATURE_VERSION = "symbolic-syntax-five-v1"


def features(lhs, rhs):
    if not isinstance(lhs, str) or not isinstance(rhs, str) or not 1 <= len(lhs) <= 512 or not 1 <= len(rhs) <= 512:
        raise ContractError("symbolic Judge expressions outside length bounds")
    return [len(lhs)/512, len(rhs)/512, rhs.count("x")/512, rhs.count("*")/512, rhs.count("/")/512]


def train_symbolic_judge(config, runs, resume=None):
    import torch
    from .metrics import calibration_metrics
    from .neural import DecisionNetwork, read_checkpoint
    from .objectives import proper_loss
    from .symbolic_data import symbolic_data, symbolic_reward, VERSION
    from .training import save_checkpoint, seed_all, validate_config
    import json
    with Run(Path(runs), "symbolic-judge", {**config, "resume": str(resume) if resume else None},
             [Path(resume)] if resume else []) as run:
        if set(config) - {"seed", "steps", "batch_size", "learning_rate", "weight_decay", "scoring_rule"}:
            raise ContractError("unknown symbolic Judge configuration field")
        validate_config({**config, "stage": "judge", "task": "arithmetic"})
        if config.get("scoring_rule", "log") not in {"log", "brier"}:
            raise ContractError("symbolic Judge requires a proper log or Brier score")
        seed_all(config["seed"])
        # Holdout results are computed after fitting; they never select updates or temperature.
        def partition(split):
            return [{"group": row["group"], "lhs": row["lhs"], "rhs": json.loads(candidate)["rhs"],
                         "features": features(row["lhs"], json.loads(candidate)["rhs"]),
                         "label": symbolic_reward(row, candidate)} for row in symbolic_data(split)
                        for candidate in row["candidates"]]
        data = {s: partition(s) for s in ("train", "validation")}
        tensors = {s: (torch.tensor([r["features"] for r in rows]), torch.tensor([r["label"] for r in rows])) for s, rows in data.items()}
        model = DecisionNetwork()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", .01))
        stable = {k: v for k, v in config.items() if k != "steps"}
        fitting_hash = digest({s: data[s] for s in ("train", "validation")})
        start = 0
        if resume:
            source = read_checkpoint(resume)
            if (source.get("kind"), source.get("feature_version"), source.get("target"), source.get("stable_config"), source.get("dataset_hash")) != ("symbolic_judge", FEATURE_VERSION, TARGET, stable, fitting_hash):
                raise ContractError("symbolic Judge resume contract mismatch")
            start = source["step"]
            if start >= config["steps"]:
                raise ContractError("resume must advance Judge step")
            model.load_state_dict(source["model"])
            optimizer.load_state_dict(source["optimizer"])
            torch.set_rng_state(source["torch_rng"])
        x, y = tensors["train"]
        for step in range(start, config["steps"]):
            indices = [(step*config["batch_size"]+i) % len(x) for i in range(config["batch_size"])]
            optimizer.zero_grad()
            loss = proper_loss(model(x[indices]), y[indices], config.get("scoring_rule", "log"))
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite symbolic Judge loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            run.metric(step+1, loss=float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits = {s: model(x) for s, (x, _) in tensors.items()}
            temperature = min((.5, 1., 2., 4.), key=lambda t: float(proper_loss(logits["validation"]/t, tensors["validation"][1], "log")))
            data["test"] = partition("test")
            tensors["test"] = (torch.tensor([r["features"] for r in data["test"]]), torch.tensor([r["label"] for r in data["test"]]))
            logits["test"] = model(tensors["test"][0])
            metrics = {s: calibration_metrics((z/temperature).sigmoid().tolist(), tensors[s][1].tolist()) for s, z in logits.items()}
            metrics["constant_train_prior_test"] = calibration_metrics([float(y.mean())]*len(tensors["test"][1]), tensors["test"][1].tolist())
        write_json(run.path/"dataset.json", {"version": VERSION, "partitions": data,
                                            "validation_role": "temperature calibration only", "label_target": TARGET})
        write_json(run.path/"metrics.json", {"partitions": metrics, "temperature": temperature, "parameters": sum(p.numel() for p in model.parameters()),
                   "scope": "syntactic toy forecast; no calibration-transfer or sequential decision-learning claim"})
        save_checkpoint(run, {"kind": "symbolic_judge", "origin": "aim-random-init-v1", "feature_version": FEATURE_VERSION,
                              "target": TARGET, "model": model.state_dict(), "hidden": 32, "temperature": temperature,
                              "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(), "step": config["steps"],
                              "stable_config": stable, "dataset_hash": fitting_hash,
                              "parent_sha256": file_hash(resume) if resume else None})
    return run.path


class CalibratedSymbolicJudge:
    def __init__(self, checkpoint, cost=.2):
        import torch
        from .neural import DecisionNetwork, read_checkpoint
        record = read_checkpoint(checkpoint)
        if (record.get("kind"), record.get("feature_version"), record.get("target"), record.get("hidden")) != ("symbolic_judge", FEATURE_VERSION, TARGET, 32):
            raise ContractError("not a compatible symbolic Judge checkpoint")
        self.temperature = finite_number(record["temperature"])
        self.cost = finite_number(cost)
        if not 0 < self.temperature <= 100 or not 0 <= self.cost <= 1:
            raise ContractError("invalid symbolic calibration temperature or check cost")
        self.model = DecisionNetwork()
        self.model.load_state_dict(record["model"])
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ContractError("nonfinite symbolic Judge weights")
        self.model.eval()
        self.model_id = "native-symbolic-judge-"+file_hash(checkpoint)[:16]

    def decide(self, state, claim):
        import torch
        request = claim["request"]
        with torch.no_grad():
            probability = float((self.model(torch.tensor([features(request["lhs"], request["rhs"])]))/self.temperature).sigmoid()[0])
        return Decision(claim["id"], probability, TARGET, "VERIFY" if probability > self.cost else "ABSTAIN", self.model_id,
                        "temperature fit on disjoint binomial offsets only; transfer unvalidated")
