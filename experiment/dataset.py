"""Validate the RNA-seq corpus and construct local SDK items without uploading it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from assays import ASSAY_SCHEMAS

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEMPLATE_IRI = ASSAY_SCHEMAS["rnaseq"]


def _object_file(path: Path) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty JSON object: {path}")
    return value, hashlib.sha256(content).hexdigest()


def load_items(data_dir: Path = DEFAULT_DATA_DIR, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Validate every pair before selecting the first *limit* filenames in sorted order.

    Gold is only in expected_output. Provenance uses corpus-relative paths and
    content hashes, not machine-specific paths or credentials.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    input_dir, gold_dir = data_dir / "rnaseq" / "input", data_dir / "rnaseq" / "gold"
    for directory in (input_dir, gold_dir):
        if not directory.is_dir():
            raise ValueError(f"Missing corpus directory: {directory}")
    inputs = {path.name for path in input_dir.glob("*.json")}
    golds = {path.name for path in gold_dir.glob("*.json")}
    if inputs != golds:
        raise ValueError(
            f"Unpaired filenames: missing gold={sorted(inputs - golds)}, missing input={sorted(golds - inputs)}"
        )
    if not inputs:
        raise ValueError("RNA-seq corpus has no JSON pairs")
    _object_file(data_dir / "schemas" / "rnaseq.json")
    items = []
    for filename in sorted(inputs):
        legacy, input_hash = _object_file(input_dir / filename)
        gold, gold_hash = _object_file(gold_dir / filename)
        items.append(
            {
                "input": {"legacy_metadata": legacy, "template_iri": TEMPLATE_IRI},
                "expected_output": gold,
                "metadata": {
                    "assay": "rnaseq",
                    "record_id": Path(filename).stem,
                    "provenance": {
                        "input": f"rnaseq/input/{filename}",
                        "gold": f"rnaseq/gold/{filename}",
                        "input_sha256": input_hash,
                        "gold_sha256": gold_hash,
                    },
                },
            }
        )
    return items[:limit] if limit is not None else items
