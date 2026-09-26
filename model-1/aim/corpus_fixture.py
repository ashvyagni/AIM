"""Project-authored engineering text only; no scientific measurements or outside data."""
from pathlib import Path

from .corpus import SCHEMA
from .tracking import digest, write_json


def create_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    templates = {
        "prose": "Engineering example {i}. A research claim records its assumptions and sources. A checker may pass, fail, or remain unresolved. Confidence is a forecast; retain the actual check result.\n",
        "code": "# Project-generated Python text example {i}; stored as data, never executed.\ndef identity_{i}(value):\n    return value\n# The document boundary and the source version remain explicit.\n",
        "math": "Algebra example {i}. Over rational polynomials, x + 0 = x and (x + 1)**2 = x**2 + 2*x + 1. An equality check must state its domain. Numerical samples alone do not prove this identity.\n",
        "unicode": "Unicode round-trip example {i}: α + β; हिन्दी; 数学; café; 🧪. Preserve UTF-8 bytes and source provenance. This is constructed tokenizer test text, not a scientific observation.\n",
    }
    documents = []
    i = 0
    for split, count in (("train", 3), ("validation", 1), ("test", 1)):
        for repeat in range(count):
            for kind, template in templates.items():
                text = template.format(i=i)
                name = f"{i:03d}-{kind}.txt"
                (directory/name).write_text(text, encoding="utf-8")
                documents.append({"id": f"document-{i:03d}", "path": name, "sha256": digest(text.encode()),
                                  "split": split, "group": f"fixture-{i}", "source_uri": f"aim://generated/pretrain/{i}",
                                  "source_version": "corpus-engineering-v1", "content_type": kind,
                                  "rights": {"license": "project-generated", "training_allowed": True,
                                             "basis": "Text authored by this fixture generator; no external passages", "reviewer": "fixture-construction-v1"},
                                  "privacy_review": {"status": "approved", "basis": "Constructed examples; no real person records",
                                                     "reviewer": "fixture-construction-v1"}})
                i += 1
    path = directory/"manifest.json"
    write_json(path, {"schema": SCHEMA, "version": "corpus-engineering-v1", "documents": documents})
    return path
