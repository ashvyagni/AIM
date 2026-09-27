"""Versioned response masking and immutable encoding-accounting records."""
from .corpus import require
from .language_contract import ENCODING
from .tracking import digest


def encode_pair(tokenizer, prompt, response, context):
    require(isinstance(prompt, str) and isinstance(response, str), "prompt and response must be text")
    require(len(prompt.encode("utf-8"))+len(response.encode("utf-8")) <= 262144, "supervised text exceeds byte budget")
    prefix = [tokenizer.bos_id]+tokenizer.encode(prompt)
    answer = tokenizer.encode(response)+[tokenizer.eos_id]
    tokens = prefix+answer
    require(len(tokens)-1 <= context, "supervised sequence exceeds context; truncation is forbidden")
    require(all(type(t) is int and 0 <= t < tokenizer.vocab_size for t in tokens), "encoder produced an invalid ID")
    mask = [False]*(len(prefix)-1)+[True]*len(answer)
    return {"inputs": tokens[:-1], "targets": tokens[1:], "mask": mask,
            "response_tokens": len(answer), "response_bytes": len(response.encode("utf-8"))}


def collate(tokenizer, pairs, context):
    import torch
    require(isinstance(pairs, list) and 1 <= len(pairs) <= 64, "batch must contain 1..64 pairs")
    records = [encode_pair(tokenizer, prompt, response, context) for prompt, response in pairs]
    width = max(len(r["inputs"]) for r in records)
    x = torch.full((len(records), width), tokenizer.pad_id, dtype=torch.long)
    y = torch.full_like(x, tokenizer.pad_id)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i, row in enumerate(records):
        length = len(row["inputs"])
        x[i,:length] = torch.tensor(row["inputs"])
        y[i,:length] = torch.tensor(row["targets"])
        mask[i,:length] = torch.tensor(row["mask"])
    return x, y, mask, records


def scores(model, tokenizer, pairs):
    x, y, mask, records = collate(tokenizer, pairs, model.cfg.context)
    logits = model(x)
    logp = logits.log_softmax(-1).gather(-1, y[:,:,None]).squeeze(-1)
    return (logp*mask).sum(-1), mask.sum(-1), records


def encoding_manifest(splits, tokenizer, context, stage):
    records = []
    for split, rows in splits.items():
        for row in rows:
            variants = {"response": row["response"]}
            if stage == "preference" or (split == "validation" and "chosen" in row and "rejected" in row):
                variants.update(chosen=row["chosen"], rejected=row["rejected"])
            if stage == "rlvr":
                variants.update({"candidate-"+str(i): text for i, text in enumerate(row["candidates"])})
            for name, response in variants.items():
                encoded = encode_pair(tokenizer, row["prompt"], response, context)
                records.append({"id": row["id"], "split": split, "variant": name, "encoding_hash": digest(encoded),
                                "input_tokens": len(encoded["inputs"]), "response_tokens": encoded["response_tokens"],
                                "response_bytes": encoded["response_bytes"]})
    result = {"schema": "aim-supervised-encoding-manifest-v1", "encoding": ENCODING,
              "tokenizer_hash": digest(tokenizer.specification()), "context": context, "stage": stage, "records": records}
    return {**result, "encoding_manifest_hash": digest(result)}
