"""Validate operator observations and prepare blocked-or-ready bounded trial plans."""
import time
from pathlib import Path

from .audit_contracts import fields, number, seal, text, verify
from .corpus import read_json, require
from .hardware_audit import validate_node
from .tracking import Run, write_json

POLICY_SCHEMA = "aim-lab-policy-v1"
PLAN_SCHEMA = "aim-fleet-trial-plan-v1"
PERMISSIONS = {"lab_access", "background_jobs", "network_ports", "shared_storage"}
POLICY_FIELDS = {"schema", "reviewer", "basis", "node_hashes", "permissions", "valid_until_unix",
                 "memory_budget_bytes_per_node", "disk_budget_bytes_per_node", "max_trial_seconds", "record_hash"}


def policy_template(nodes):
    return {"schema": POLICY_SCHEMA, "reviewer": None, "basis": None,
            "node_hashes": sorted(validate_node(n)["record_hash"] for n in nodes),
            "permissions": {key: False for key in sorted(PERMISSIONS)}, "valid_until_unix": None,
            "memory_budget_bytes_per_node": None, "disk_budget_bytes_per_node": None,
            "max_trial_seconds": 180}


def validate_policy(policy, nodes, now):
    verify(policy, POLICY_SCHEMA)
    fields(policy, POLICY_FIELDS, "lab policy")
    text(policy["reviewer"], "reviewer")
    text(policy["basis"], "policy evidence basis")
    require(policy["node_hashes"] == sorted(n["record_hash"] for n in nodes), "policy must bind exactly the reviewed observations")
    fields(policy["permissions"], PERMISSIONS, "permission")
    require(all(type(v) is bool for v in policy["permissions"].values()), "permissions must be explicit booleans")
    number(policy["valid_until_unix"], "policy expiration", 1)
    number(policy["memory_budget_bytes_per_node"], "operator RAM budget", 1, integer=True)
    number(policy["disk_budget_bytes_per_node"], "operator disk budget", 1, integer=True)
    require(type(policy["max_trial_seconds"]) is int and 15 <= policy["max_trial_seconds"] <= 600, "trial deadline must be 15..600 seconds")
    return policy["valid_until_unix"] >= now


def validate_nodes(nodes, now, max_age_seconds):
    require(isinstance(nodes, list) and 1 <= len(nodes) <= 128, "fleet import requires 1..128 observations")
    number(now, "current time", 1)
    require(type(max_age_seconds) is int and 60 <= max_age_seconds <= 604800, "observation age window must be 60 seconds..7 days")
    for node in nodes:
        validate_node(node)
    require(len({n["node_id"] for n in nodes}) == len(nodes), "duplicate node ID; update observations explicitly")
    require(len({n["record_hash"] for n in nodes}) == len(nodes), "duplicate observation")
    return [n["node_id"] for n in nodes if now-n["observed_at_unix"] > max_age_seconds or n["observed_at_unix"] > now+60]


def build_plan(nodes, config, policy=None, now=None, max_age_seconds=86400):
    now = time.time() if now is None else now
    stale = validate_nodes(nodes, now, max_age_seconds)
    # This plan is a fixed, small CPU workload, not permission to change allocation guards.
    from .distributed_pretrain import base_config
    from .pretrain import validate_config
    from .tokenization import ByteTokenizer
    cfg = validate_config(base_config(config), ByteTokenizer())
    params = cfg.parameter_estimate()
    floor = 16*params
    blockers = ["stale_or_future_observation:"+n for n in stale]
    identities = [tuple(n["measurements"][key]["value"] for key in ("os", "architecture", "python", "torch")) for n in nodes]
    if any(None in i for i in identities):
        blockers.append("missing_runtime_identity")
    if len(set(identities)) != 1:
        blockers.append("heterogeneous_runtime_requires_separate_trial")
    if policy is None:
        blockers.append("operator_policy_missing")
    else:
        if not validate_policy(policy, nodes, now):
            blockers.append("operator_policy_expired")
        blockers += ["permission_missing:"+k for k, v in sorted(policy["permissions"].items()) if not v]
        if policy["memory_budget_bytes_per_node"] < floor:
            blockers.append("operator_memory_budget_below_parameter_state_floor")
        # Reserve only an explicit artifact floor; real peak storage remains unmeasured.
        checkpoints = (config["steps"]+config["checkpoint_every"]-1)//config["checkpoint_every"]
        disk_floor = checkpoints*12*params
        if policy["disk_budget_bytes_per_node"] < disk_floor:
            blockers.append("operator_disk_budget_below_checkpoint_planning_floor")
        for node in nodes:
            values = node["measurements"]
            available = values["available_ram_bytes"]["value"]
            free_disk = values["free_disk_bytes"]["value"]
            if available is None:
                blockers.append("available_memory_unknown:"+node["node_id"])
            elif policy["memory_budget_bytes_per_node"] > available:
                blockers.append("memory_budget_exceeds_observed_availability:"+node["node_id"])
            if free_disk is None or policy["disk_budget_bytes_per_node"] > free_disk:
                blockers.append("disk_budget_not_supported_by_observation:"+node["node_id"])
    trials = [{"physical_nodes": n, "processes_per_node": 1, "global_sequences_per_update": n*config["batch_size"]*config["accumulation_steps"],
               "position_budget_per_update": n*config["batch_size"]*config["accumulation_steps"]*cfg.context,
               "selection": "operator must explicitly select distinct physical hosts"} for n in (1, 2, 4) if n <= len(nodes)]
    return seal({"schema": PLAN_SCHEMA, "created_at_unix": now, "observation_hashes": sorted(n["record_hash"] for n in nodes),
                 "node_ids": sorted(n["node_id"] for n in nodes), "policy_hash": policy["record_hash"] if policy else None,
                 "max_trial_seconds": policy["max_trial_seconds"] if policy else None,
                 "configuration": config, "parameters": params, "fp32_state_floor_bytes_per_rank": floor,
                 "status": "BLOCKED" if blockers else "READY_FOR_OPERATOR_TRIAL", "blockers": sorted(blockers),
                 "trials": trials, "unmeasured": ["sustained_throughput", "activation_and_runtime_peak_memory", "network_collectives",
                 "shared_filesystem_visibility", "physical_host_distinctness", "resource_quotas", "thermal_interference", "physical_node_loss_recovery"],
                 "scope": "preparation only; no automatic launch, scale approval, aggregate RAM pooling or physical-host certification"})


def plan_run(node_paths, config_path, runs, policy_path=None, max_age_seconds=86400):
    paths = [Path(p) for p in node_paths]
    with Run(Path(runs), "fleet-plan", {"max_age_seconds": max_age_seconds}, paths+[Path(config_path)]+([Path(policy_path)] if policy_path else [])) as run:
        nodes = [validate_node(read_json(p)) for p in paths]
        write_json(run.path/"observations.json", nodes)
        write_json(run.path/"policy-template.json", policy_template(nodes))
        policy = read_json(policy_path) if policy_path else None
        if policy is not None:
            write_json(run.path/"operator-policy.json", policy)
        plan = build_plan(nodes, read_json(config_path), policy, max_age_seconds=max_age_seconds)
        write_json(run.path/"plan.json", plan)
    return run.path/"plan.json"
