"""Portable local observations and bounded filesystem probes; no remote access."""
import importlib.metadata
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from .audit_contracts import fields, identifier, number, seal, text, verify
from .corpus import read_json, require
from .tracking import Run, file_hash, write_json

SCHEMA = "aim-node-observation-v1"
PROBE_SCHEMA = "aim-storage-probe-v1"
NUMERIC = {"logical_cpus", "physical_cores", "ram_bytes", "available_ram_bytes", "free_disk_bytes"}
STRING = {"cpu_model", "os", "architecture", "python", "torch"}
MEASUREMENTS = NUMERIC | STRING


def observed(value, method, error=None):
    return {"value": value, "status": "unavailable" if value is None else "observed", "method": method, "error": error}


def sysctl(key):
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def proc_memory(raw):
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        if parts and parts[0] in {"MemTotal:", "MemAvailable:"}:
            require(len(parts) == 3 and parts[2] == "kB" and parts[1].isdigit(), "invalid /proc memory field")
            result[parts[0][:-1]] = int(parts[1])*1024
    return result


def proc_cpu(raw):
    models, cores = set(), set()
    for block in raw.split("\n\n"):
        data = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        data = {k.strip(): v.strip() for k, v in data.items()}
        if data.get("model name"):
            models.add(data["model name"])
        if "physical id" in data and "core id" in data:
            cores.add((data["physical id"], data["core id"]))
    return ("; ".join(sorted(models)) or None), (len(cores) or None)


def collect(node_id, workspace):
    identifier(node_id)
    workspace = Path(workspace)
    require(workspace.is_dir(), "workspace must be an existing directory")
    values = {key: observed(None, "not implemented on this platform", "unavailable") for key in MEASUREMENTS}
    values.update(logical_cpus=observed(os.cpu_count(), "os.cpu_count; host-visible logical CPUs, not an allocation quota"),
                  os=observed(platform.system()+" "+platform.release(), "platform system/release"),
                  architecture=observed(platform.machine() or None, "platform.machine"),
                  python=observed(platform.python_version(), "platform.python_version"))
    try:
        values["free_disk_bytes"] = observed(shutil.disk_usage(workspace).free, "shutil.disk_usage on operator-selected workspace")
    except OSError:
        values["free_disk_bytes"] = observed(None, "shutil.disk_usage", "OS query failed")
    try:
        values["torch"] = observed(importlib.metadata.version("torch"), "installed distribution metadata; import/kernel support not tested")
    except importlib.metadata.PackageNotFoundError:
        values["torch"] = observed(None, "installed distribution metadata", "package not installed")
    if platform.system() == "Darwin":
        for key, query in (("cpu_model", "machdep.cpu.brand_string"), ("physical_cores", "hw.physicalcpu"), ("ram_bytes", "hw.memsize")):
            value = sysctl(query)
            if key in NUMERIC:
                value = int(value) if value and value.isdigit() else None
            values[key] = observed(value, "sysctl "+query, "query unavailable" if value is None else None)
        # Free pages are not an equivalent of Linux MemAvailable; retain unknown.
        values["available_ram_bytes"] = observed(None, "no equivalent portable availability query implemented", "unavailable")
    elif platform.system() == "Linux":
        try:
            memory = proc_memory(Path("/proc/meminfo").read_text())
            for key, name in (("ram_bytes", "MemTotal"), ("available_ram_bytes", "MemAvailable")):
                values[key] = observed(memory.get(name), "/proc/meminfo "+name+"; host-visible, may exceed cgroup quota")
        except (OSError, ValueError):
            pass
        try:
            model, cores = proc_cpu(Path("/proc/cpuinfo").read_text())
            values["cpu_model"] = observed(model, "/proc/cpuinfo model name")
            values["physical_cores"] = observed(cores, "unique /proc/cpuinfo physical-id/core-id pairs")
        except (OSError, ValueError):
            pass
    record = seal({"schema": SCHEMA, "node_id": node_id, "observed_at_unix": time.time(),
                   "measurements": values, "scope": "local OS observations; no NIC/GPU/cluster/availability certification"})
    validate_node(record)
    return record


def validate_node(record):
    verify(record, SCHEMA)
    fields(record, {"schema", "node_id", "observed_at_unix", "measurements", "scope", "record_hash"}, "node")
    identifier(record["node_id"])
    number(record["observed_at_unix"], "observation time", 1)
    text(record["scope"], "scope")
    fields(record["measurements"], MEASUREMENTS, "measurement")
    for key, observation in record["measurements"].items():
        fields(observation, {"value", "status", "method", "error"}, "observation")
        text(observation["method"], "measurement method")
        require(observation["status"] in {"observed", "unavailable"}, "invalid observation status")
        require((observation["value"] is None) == (observation["status"] == "unavailable"), "missing values must be unavailable")
        if observation["error"] is not None:
            text(observation["error"], "observation error")
        if observation["value"] is not None:
            require(observation["error"] is None, "successful observation cannot contain an error")
            if key in NUMERIC:
                number(observation["value"], key, 0 if key in {"free_disk_bytes", "available_ram_bytes"} else 1, integer=True)
            else:
                text(observation["value"], key)
    values = {k: v["value"] for k, v in record["measurements"].items()}
    if values["available_ram_bytes"] is not None and values["ram_bytes"] is not None:
        require(values["available_ram_bytes"] <= values["ram_bytes"], "available RAM exceeds total")
    if values["physical_cores"] is not None and values["logical_cpus"] is not None:
        require(values["physical_cores"] <= values["logical_cpus"], "physical cores exceed logical CPUs")
    return record


def collect_run(node_id, workspace, runs):
    with Run(Path(runs), "node-audit", {"node_id": node_id, "scope": "local observation only"}) as run:
        write_json(run.path/"node.json", collect(node_id, workspace))
    return run.path/"node.json"


def storage_probe(node_path, runs, mebibytes=4):
    node = validate_node(read_json(node_path))
    require(type(mebibytes) is int and 1 <= mebibytes <= 32, "storage probe must be 1..32 MiB")
    with Run(Path(runs), "storage-probe", {"node_hash": node["record_hash"], "mebibytes": mebibytes}, [Path(node_path)]) as run:
        live = collect(node["node_id"], run.path)
        for key in ("os", "architecture", "cpu_model", "logical_cpus", "ram_bytes"):
            require(node["measurements"][key]["value"] == live["measurements"][key]["value"], "probe host differs from supplied observation")
        write_json(run.path/"live-node.json", live)
        path = run.path/"probe.bin"
        block = bytes(range(256))*4096
        began = time.perf_counter()
        with path.open("xb") as stream:
            for _ in range(mebibytes):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        written = time.perf_counter()-began
        began = time.perf_counter()
        with path.open("rb") as stream:
            for _ in range(mebibytes):
                require(stream.read(len(block)) == block, "storage probe readback mismatch")
            require(stream.read(1) == b"", "storage probe unexpected trailing bytes")
        read = time.perf_counter()-began
        result = seal({"schema": PROBE_SCHEMA, "node_hash": node["record_hash"], "live_node_hash": live["record_hash"], "bytes": path.stat().st_size,
                       "write_fsync_seconds": written, "cached_read_seconds": read, "sha256": file_hash(path),
                       "round_trip_equal": True, "scope": "one local file; cached read; fsync returned; no power-loss/shared-filesystem proof"})
        write_json(run.path/"probe.json", result)
    return run.path/"probe.json"
