"""Audit populated gold values against the saved, simplified CEDAR templates.

An objectively invalid value excludes its entire field within that assay. The audit
uses gold and template files only, independently of predictions or disagreements.
It does not reassess curation, infer rules from descriptions, or query an ontology.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, iter_records, load_record
from analysis.metrics import _is_missing

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


FIELD_COLUMNS = [
    "assay",
    "field",
    "field_type",
    "status",
    "n_invalid_records",
    "n_gold_records",
    "n_excluded_comparisons",
    "constraints",
    "example_values",
]
VIOLATION_COLUMNS = ["assay", "field", "constraint", "expected", "gold_value", "n_records", "records"]
COVERAGE_COLUMNS = [
    "assay",
    "n_gold_records",
    "n_excluded_fields",
    "n_excluded_comparisons",
    "unchecked_fields",
]


@dataclass
class ReferenceConstraintAudit:
    """Assay-keyed exclusions, field summaries, and complete reproducible evidence.

    ``violations`` groups identical (field, value, constraint) witnesses, retaining
    every gold filename. ``fields`` counts invalid records once even when a value
    violates several constraints. Excluded comparisons include *all* instances of
    that field in gold, including valid and blank ones. ``coverage`` includes assays
    with zero exclusions and names gold fields absent from the saved template.
    """

    excluded_fields_by_assay: dict[str, frozenset[str]]
    fields: pd.DataFrame
    violations: pd.DataFrame
    coverage: pd.DataFrame


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _type_matches(value: Any, field_type: str) -> bool:
    """Check the JSON representation used by this corpus without coercing values.

    Link and temporal fields are strings in the simplified CEDAR representation;
    format restrictions are checked only when an explicit pattern is supplied.
    JSON numbers with integral values satisfy integer constraints. Booleans are
    never numbers, and quoted numbers do not satisfy numeric constraints.
    """
    if field_type in ("string", "link", "date", "datetime", "time"):
        return isinstance(value, str)
    if field_type == "boolean":
        return isinstance(value, bool)
    if field_type in ("integer", "decimal"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return field_type == "decimal" or isinstance(value, int) or value.is_integer()
    raise ValueError(f"Unsupported CEDAR field type: {field_type!r}")


def _value_violations(value: Any, field: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return explicit constraint violations, without normalizing the gold value."""
    if _is_missing(value):
        return []

    multivalued = field.get("multivalued", False)
    if isinstance(value, list) != multivalued:
        return [("multivalued", multivalued)]
    values = value if multivalued else [value]
    violations: list[tuple[str, Any]] = []
    field_type = field.get("type")
    if field_type and any(not _type_matches(item, field_type) for item in values):
        violations.append(("type", field_type))

    pattern = field.get("pattern")
    if pattern is not None:
        # JSON Schema patterns use a search, not an implicit full match. ASCII
        # character classes mirror the JavaScript regexes in the saved templates.
        compiled = re.compile(pattern, re.ASCII)
        if any(isinstance(item, str) and compiled.search(item) is None for item in values):
            violations.append(("pattern", pattern))

    permissible = field.get("permissible_values") or []
    # Multiple permissible-value specifications are alternatives. If any is open
    # (an ontology without an enumerated list, for example), nonmembership in the
    # other lists cannot establish a violation. Empty lists are unspecified in
    # the simplified templates, as in metrics.field_roles.
    if permissible and all(pv.get("options") for pv in permissible):
        options = [option for pv in permissible for option in pv["options"]]
        if any(item not in options for item in values):
            violations.append(("permissible_values", options))
    return violations


def audit_reference_constraints(data_root: str | Path) -> ReferenceConstraintAudit:
    """Find assay-wide exclusions from *every* gold record, including unpredicted ones.

    Only populated values can trigger exclusion. Null, empty, whitespace-only,
    and absent values retain the evaluator's existing unknown/blank semantics,
    even for required fields. Presence constraints are therefore not assessed.
    Checks cover explicit type, multiplicity, regex, and closed permissible-value
    lists in the flat simplified CEDAR templates used by this corpus. A gold field
    absent from the template is reported as unchecked, rather than invented as a
    constraint violation. Defaults and descriptive examples are not constraints.
    """
    import pandas as pd

    exclusions: dict[str, frozenset[str]] = {}
    field_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for assay in iter_assays(data_root):
        if not assay.gold_dir.is_dir():
            continue
        if not assay.schema_path.is_file():
            raise FileNotFoundError(f"Missing template for gold assay {assay.key}: {assay.schema_path}")
        schema = load_record(assay.schema_path)
        if "children" not in schema:
            raise ValueError(f"Expected a simplified CEDAR template: {assay.schema_path}")
        definitions = {child["name"]: child for child in schema["children"]}
        if any(child.get("type") == "element" for child in definitions.values()):
            raise ValueError(f"Reference constraint audit requires a flat template: {assay.schema_path}")

        gold_records = list(iter_records(assay.gold_dir))
        witnesses: dict[tuple[str, str, str, str], list[str]] = {}
        invalid_records: dict[str, set[str]] = {}
        occurrences: dict[str, int] = {}
        unchecked: set[str] = set()
        for path, gold in gold_records:
            for name, value in gold.items():
                occurrences[name] = occurrences.get(name, 0) + 1
                if name not in definitions:
                    unchecked.add(name)
                    continue
                for constraint, expected in _value_violations(value, definitions[name]):
                    key = (name, constraint, _json(expected), _json(value))
                    witnesses.setdefault(key, []).append(path.name)
                    invalid_records.setdefault(name, set()).add(path.name)

        exclusions[assay.key] = frozenset(invalid_records)
        ontology_fields = assay.ontology_fields()
        for name in sorted(invalid_records):
            keys = [key for key in witnesses if key[0] == name]
            field_rows.append(
                {
                    "assay": assay.key,
                    "field": name,
                    "field_type": "ontology" if name in ontology_fields else "non_ontology",
                    "status": "excluded",
                    "n_invalid_records": len(invalid_records[name]),
                    "n_gold_records": len(gold_records),
                    "n_excluded_comparisons": occurrences[name],
                    "constraints": ", ".join(sorted({key[1] for key in keys})),
                    "example_values": _json([json.loads(value) for value in sorted({key[3] for key in keys})[:3]]),
                }
            )
        for (name, constraint, expected, value), records in sorted(witnesses.items()):
            violation_rows.append(
                {
                    "assay": assay.key,
                    "field": name,
                    "constraint": constraint,
                    "expected": expected,
                    "gold_value": value,
                    "n_records": len(records),
                    "records": _json(records),
                }
            )
        coverage_rows.append(
            {
                "assay": assay.key,
                "n_gold_records": len(gold_records),
                "n_excluded_fields": len(invalid_records),
                "n_excluded_comparisons": sum(occurrences[name] for name in invalid_records),
                "unchecked_fields": _json(sorted(unchecked)),
            }
        )

    return ReferenceConstraintAudit(
        exclusions,
        pd.DataFrame(field_rows, columns=FIELD_COLUMNS),
        pd.DataFrame(violation_rows, columns=VIOLATION_COLUMNS),
        pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS),
    )
