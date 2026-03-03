"""Evaluation metrics for comparing agent output against gold standard.

Metrics:
    - Overall accuracy: overall record-level agreement across all fields in the
      gold standard.  Both-null counts as a match; any difference in value or
      presence counts as a mismatch.
"""

from __future__ import annotations

import json
from typing import Any


def compute_overall_accuracy(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> float:
    """Compute accuracy of *predicted* metadata against *gold*.

    Accuracy measures overall record-level agreement: the fraction of gold
    fields where both records agree.  Two fields agree when:

    * both values are missing (``None``), or
    * both values are non-missing and match via ``_values_match()``.

    The denominator is all keys present in *gold*.

    Returns 0.0 when *gold* has no fields.
    """
    if not gold:
        return 0.0
    matches = 0
    for k in gold:
        gold_val = gold[k]
        pred_val = predicted.get(k)
        gold_missing = _is_missing(gold_val)
        pred_missing = _is_missing(pred_val)
        if (gold_missing and pred_missing) or (
            not gold_missing
            and not pred_missing
            and _values_match(
                pred_val, gold_val, match_case=match_case, match_whole_word=match_whole_word, field_name=k
            )
        ):
            matches += 1
    return matches / len(gold)


def compute_ontology_constrained_field_accuracy(
    predicted: dict[str, object],
    gold: dict[str, object],
    schema_path: str,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> float:
    """Compute correctness restricted to ontology-constrained fields.

    Only gold fields whose names appear in the schema's ontology/branch
    permissible-value list are evaluated.  Both-null counts as a match.
    Returns the fraction of those fields where the predicted value matches.

    Returns 0.0 when no ontology-constrained fields exist.
    """
    ontology_fields = set(_get_ontology_constrained_fields(schema_path))
    filtered_gold = {k: v for k, v in gold.items() if k in ontology_fields}
    if not filtered_gold:
        return 0.0
    matches = 0
    for k in filtered_gold:
        gold_val = filtered_gold[k]
        pred_val = predicted.get(k)
        gold_missing = _is_missing(gold_val)
        pred_missing = _is_missing(pred_val)
        if (gold_missing and pred_missing) or (
            not gold_missing
            and not pred_missing
            and _values_match(
                pred_val, gold_val, match_case=match_case, match_whole_word=match_whole_word, field_name=k
            )
        ):
            matches += 1
    return matches / len(filtered_gold)


def compute_non_ontology_constrained_field_accuracy(
    predicted: dict[str, object],
    gold: dict[str, object],
    schema_path: str,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> float:
    """Compute correctness restricted to non-ontology-constrained fields.

    Only gold fields whose names do **not** appear in the schema's
    ontology/branch permissible-value list are evaluated.  Both-null counts
    as a match.  Returns the fraction of those fields where the predicted
    value matches.

    Returns 0.0 when no qualifying fields exist.
    """
    ontology_fields = set(_get_ontology_constrained_fields(schema_path))
    filtered_gold = {k: v for k, v in gold.items() if k not in ontology_fields}
    if not filtered_gold:
        return 0.0
    matches = 0
    for k in filtered_gold:
        gold_val = filtered_gold[k]
        pred_val = predicted.get(k)
        gold_missing = _is_missing(gold_val)
        pred_missing = _is_missing(pred_val)
        if (gold_missing and pred_missing) or (
            not gold_missing
            and not pred_missing
            and _values_match(
                pred_val, gold_val, match_case=match_case, match_whole_word=match_whole_word, field_name=k
            )
        ):
            matches += 1
    return matches / len(filtered_gold)


def _normalize_doi(value: str) -> str:
    """Normalize DOI URLs so that doi.org, dx.doi.org, and bare DOIs are equivalent."""
    return value.replace("https://doi.org/", "https://dx.doi.org/")


def _values_match(
    predicted_val: Any,
    gold_val: Any,
    *,
    match_case: bool,
    match_whole_word: bool,
    field_name: str = "",
) -> bool:
    """Return ``True`` if *predicted_val* matches *gold_val* under the given flags.

    When either value is not a ``str``, exact equality is used regardless of
    flags.  For string values:

    * *match_case=False* lowercases both strings before comparison.
    * *match_whole_word=False* checks whether the gold value is a **substring
      of** the predicted value (rather than requiring equality).
    * When *field_name* ends with ``_doi``, DOI URLs are normalised so that
      ``doi.org`` and ``dx.doi.org`` are treated as equivalent.
    """
    if not isinstance(predicted_val, str) or not isinstance(gold_val, str):
        return predicted_val == gold_val

    predicted_str = predicted_val if match_case else predicted_val.lower()
    gold_str = gold_val if match_case else gold_val.lower()

    if field_name.endswith("_doi"):
        predicted_str = _normalize_doi(predicted_str)
        gold_str = _normalize_doi(gold_str)

    if match_whole_word:
        return predicted_str == gold_str
    return gold_str in predicted_str


def _is_missing(value: Any) -> bool:
    """Return ``True`` if *value* is considered missing.

    A field value is missing when it is ``None``.  Empty strings, empty lists,
    and other falsy-but-not-None values are considered present.
    """
    return value is None


def _get_ontology_constrained_fields(schema_path: str) -> list[str]:
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
