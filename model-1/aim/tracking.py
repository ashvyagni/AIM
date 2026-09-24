"""Append-only run records, content hashes and explicit failure retention."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def canonical(value) -> str:
    def encode(obj):
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Unsupported record type: {type(obj).__name__}")
    return json.dumps(value, default=encode,
                      sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value) -> str:
    raw = value if isinstance(value, bytes) else canonical(value).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    # Exclusive writes preserve existing artifacts and failed runs.
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def command(args, cwd=ROOT):
    try:
        r = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment() -> dict:
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "packages": packages, "git_commit": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--porcelain"]),
            "git_note": "null commit means this workspace has no readable commit; code hashes are authoritative"}


class Run:
    def __init__(self, runs: Path, kind: str, config: dict, inputs=()):
        self.id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + kind + "-" + uuid.uuid4().hex[:8]
        self.path = Path(runs) / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        source_files = sorted(p for folder in ("aim", "tests", "scripts", "configs", "data", "eval")
                              for p in (ROOT / folder).rglob("*")
                              if p.is_file() and "__pycache__" not in p.parts)
        source_files += [p for p in (ROOT / "pyproject.toml", ROOT / "requirements-lock.txt") if p.exists()]
        sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files}
        with zipfile.ZipFile(self.path / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for p in source_files:
                archive.write(p, str(p.relative_to(ROOT)))
        self.config = config
        self.started = time.perf_counter()
        write_json(self.path / "manifest.json", {"schema_version": 1, "run_id": self.id,
                   "kind": kind, "configuration": config, "configuration_hash": digest(config),
                   "environment": environment(), "code_hashes": sources, "code_hash": digest(sources),
                   "inputs": {str(p): file_hash(p) for p in inputs}, "created_at_unix": time.time()})
        self.event("RUN_STARTED", {"kind": kind})

    def event(self, kind, payload):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(canonical({"time_unix": time.time(), "kind": kind, "payload": payload}) + "\n")

    def metric(self, step, **values):
        self.event("METRIC", {"step": step, **values})

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        status = "FAILED" if typ else "COMPLETED"
        if typ:
            with (self.path / "failure.txt").open("x") as f:
                f.write("".join(traceback.format_exception(typ, value, tb)))
        write_json(self.path / "status.json", {"status": status, "elapsed_seconds": time.perf_counter() - self.started,
                   "error": str(value) if typ else None})
        self.event("RUN_FINISHED", {"status": status})
        return False
