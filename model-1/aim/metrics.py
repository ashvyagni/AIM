import math


def calibration_metrics(probabilities, labels):
    if len(probabilities) != len(labels):
        raise ValueError("Forecast/label length mismatch")
    resolved = [(p,y) for p,y in zip(probabilities,labels) if y is not None]
    if not resolved:
        return {"resolved": 0, "brier": None, "log_loss": None, "accuracy": None, "selective": []}
    if any(not math.isfinite(p) or not 0 <= p <= 1 or y not in (0,1) for p,y in resolved):
        raise ValueError("Invalid forecast or resolved label")
    n = len(resolved)
    brier = sum((p-y)**2 for p,y in resolved)/n
    nll = -sum(y*math.log(max(p,1e-12)) + (1-y)*math.log(max(1-p,1e-12)) for p,y in resolved)/n
    selective = []
    for threshold in (0.0, 0.5, 0.8, 0.95):
        accepted = [y for p,y in resolved if p >= threshold]
        selective.append({"threshold":threshold, "coverage":len(accepted)/n,
                          "prediction_failure_risk":1-sum(accepted)/len(accepted) if accepted else None})
    return {"resolved":n, "brier":brier, "log_loss":nll,
            "accuracy":sum((p>=0.5)==bool(y) for p,y in resolved)/n, "selective":selective}
