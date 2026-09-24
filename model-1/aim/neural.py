"""Small randomly initialized dense decoder and independent decision network.

CPU FP32 reference kernels, GQA through explicit KV repetition, RoPE, RMSNorm,
SwiGLU. These implementations establish interfaces, not a final scale choice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ContractError


class ByteTokenizer:
    version = "aim-utf8-byte-v1"
    vocab_size, bos_id, eos_id, pad_id = 259, 256, 257, 258

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens, strict=True):
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="strict" if strict else "replace")

    def specification(self):
        return {"version": self.version, "vocab_size": self.vocab_size, "encoding": "utf-8",
                "bos": self.bos_id, "eos": self.eos_id, "pad": self.pad_id,
                "learned": False, "scope": "temporary engineering tokenizer; no pretrained assets"}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259
    width: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    ffn_width: int = 128
    context: int = 256
    rope_base: float = 10000.0

    def __post_init__(self):
        values = [self.vocab_size, self.width, self.layers, self.heads, self.kv_heads, self.ffn_width, self.context]
        if any(type(x) is not int or x <= 0 for x in values):
            raise ContractError("Model dimensions must be positive integers")
        if self.width % self.heads or self.heads % self.kv_heads or (self.width // self.heads) % 2:
            raise ContractError("Invalid attention head dimensions")
        if not 1 < self.rope_base <= 1e8:
            raise ContractError("Invalid RoPE base")

    def parameter_estimate(self):
        kv_width = self.width // self.heads * self.kv_heads
        return self.vocab_size * self.width + self.width + self.layers * (
            2 * self.width ** 2 + 2 * self.width * kv_width + 3 * self.width * self.ffn_width + 2 * self.width)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


def rope(x, base):
    dim, length = x.shape[-1], x.shape[-2]
    inverse = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angle = torch.outer(torch.arange(length, device=x.device).float(), inverse)
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.width, cfg.width // cfg.heads
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, cfg.kv_heads * hd, bias=False), nn.Linear(d, cfg.kv_heads * hd, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, cfg.ffn_width, bias=False)
        self.up = nn.Linear(d, cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        z = self.norm1(x)
        cfg = self.cfg
        q = self.q(z).view(b, t, cfg.heads, d // cfg.heads).transpose(1, 2)
        k = self.k(z).view(b, t, cfg.kv_heads, d // cfg.heads).transpose(1, 2)
        v = self.v(z).view(b, t, cfg.kv_heads, d // cfg.heads).transpose(1, 2)
        q, k = rope(q, cfg.rope_base), rope(k, cfg.rope_base)
        k = k.repeat_interleave(cfg.heads // cfg.kv_heads, dim=1)
        v = v.repeat_interleave(cfg.heads // cfg.kv_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        x = x + self.out(y.transpose(1, 2).contiguous().view(b, t, d))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class CausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.cfg.context or tokens.shape[1] < 1:
            raise ContractError("Token shape/context outside model contract")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.embedding.weight)

    @torch.no_grad()
    def generate_text(self, tokenizer, prompt, max_new_tokens):
        tokens = [tokenizer.bos_id] + tokenizer.encode(prompt)
        if len(tokens) + max_new_tokens > self.cfg.context:
            raise ContractError("Generation budget exceeds context; provide a smaller explicit request")
        result = []
        for _ in range(max_new_tokens):
            value = int(self(torch.tensor([tokens]))[0, -1].argmax())
            if value == tokenizer.eos_id:
                break
            result.append(value)
            tokens.append(value)
        return tokenizer.decode(result, strict=False)


class DecisionNetwork(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def read_checkpoint(path):
    import json
    from .tracking import file_hash
    path = Path(path)
    sidecar = path.with_name(path.name + ".sha256.json")
    if not sidecar.is_file() or json.loads(sidecar.read_text()).get("sha256") != file_hash(path):
        raise ContractError("Checkpoint integrity sidecar missing or hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("origin") != "aim-random-init-v1":
        raise ContractError("Checkpoint must declare AIM random-initialization lineage")
    return checkpoint


def load_lm(path):
    record = read_checkpoint(path)
    if record.get("kind") != "causal_lm":
        raise ContractError("Checkpoint is not a Researcher LM")
    cfg = ModelConfig(**record["model_config"])
    if cfg.parameter_estimate() > 2_000_000 or record.get("tokenizer") != ByteTokenizer().specification():
        raise ContractError("Checkpoint exceeds miniature budget or tokenizer contract")
    model = CausalLM(cfg)
    model.load_state_dict(record["model"])
    return model, ByteTokenizer(), record


def load_judge(path):
    record = read_checkpoint(path)
    if record.get("kind") != "decision_network" or record.get("feature_version") != "polynomial-features-v1":
        raise ContractError("Checkpoint is not a compatible Judge")
    from .contracts import finite_number
    from .judge import TARGET
    if record.get("hidden") != 32 or record.get("target") != TARGET or not 0 < finite_number(record["temperature"]) <= 100:
        raise ContractError("Invalid Judge dimensions, calibration temperature or target")
    model = DecisionNetwork(record["hidden"])
    model.load_state_dict(record["model"])
    model.eval()
    return model, record["temperature"], record
