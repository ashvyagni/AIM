"""Descriptive fit/transfer diagnostics; never used for checkpoint selection."""
import torch

from .curriculum_tasks import diagnostic_pools, score_auxiliary


@torch.no_grad()
def measure(model,tokenizer,data,max_new_tokens):
    from .research_train import validation_metrics
    training=sorted(data["train"],key=lambda r:r["group"])[:64]
    research=validation_metrics(model,tokenizer,training,max_new_tokens)
    pools,counts=diagnostic_pools(data);results={}
    model.eval()
    for key,rows in pools.items():
        outputs=[]
        for row in rows:
            text=model.generate_text(tokenizer,row["prompt"],max_new_tokens)
            outputs.append({"prompt":row["prompt"],"expected":row["response"],"raw_output":text,
                            "world_group":row["world_group"],**score_auxiliary(text,row)})
        n=len(outputs)
        results[key]={"n":n,"valid_rate":sum(r["valid"] for r in outputs)/n if n else None,
                      "accuracy":sum(r["correct"] for r in outputs)/n if n else None,"outputs":outputs}
    model.train()
    return {"research_training_sample":research,"auxiliary":results,"pool_counts":counts,
            "scope":"fixed diagnostic samples; excluded from checkpoint selection"}
