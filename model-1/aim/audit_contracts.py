"""Strict, hash-bound portable audit records. Hashes are not signatures."""
import math
import re

from .corpus import require
from .tracking import digest


def fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), label+" fields differ")


def text(value, label, maximum=2048):
    require(isinstance(value, str) and 0 < len(value) <= maximum and bool(value.strip()), label+" must be nonempty text")
    return value


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) is not None, "use an anonymous alphanumeric node ID, at most 64 characters")
    return value


def number(value, label, minimum=0, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), label+" has invalid numeric type")
    require(math.isfinite(value) and value >= minimum, label+" outside numeric bounds")
    return value


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA-256")
    return value


def seal(record):
    require("record_hash" not in record, "record already sealed")
    return {**record, "record_hash": digest(record)}


def verify(record, schema):
    require(isinstance(record, dict) and record.get("schema") == schema, "unexpected audit schema")
    sha(record.get("record_hash"))
    require(record["record_hash"] == digest({k: v for k, v in record.items() if k != "record_hash"}), "audit record hash mismatch")
