"""Which role each field plays in a CEDAR template.

The only code in this package that opens the template JSON.  Whether a field is
ontology-constrained decides how every metric splits its results, since the whole
comparison turns on whether the model had to pick a term from a vocabulary.

Only top-level ``children`` are read.  Every template in this corpus is flat; a
template that nested fields inside an ``element`` would need the recursive walk
:mod:`conditions.prompt_only.template_spec` does.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _get_permissible_values(schema_path: Path) -> dict[str, list[str]]:
    """Field name -> the values its template enumerates, for the fields that enumerate any.

    A field whose ``permissible_values`` carry no ``options`` is absent from the result
    rather than present with an empty list: "this template does not say" and "this template
    allows nothing" are different, and only the first happens.  Returned raw, since what
    counts as the same value is :mod:`analysis.data_analysis.error_taxonomy`'s decision and
    not this module's.
    """
    with open(schema_path) as f:
        schema = json.load(f)
    fields: dict[str, list[str]] = {}
    for child in schema.get("children", []):
        options = [option for pv in child.get("permissible_values", []) for option in pv.get("options", [])]
        if options:
            fields[child["name"]] = options
    return fields


def _get_ontology_constrained_fields(schema_path: Path) -> list[str]:
    """Return field names constrained by ontology/branch permissible values."""
    with open(schema_path) as f:
        schema = json.load(f)
    fields: list[str] = []
    for child in schema.get("children", []):
        for pv in child.get("permissible_values", []):
            if pv.get("type") in ("branch", "ontology"):
                fields.append(child["name"])
                break
    return fields
