"""Record-level accuracy: the fraction of gold fields the prediction agrees on.

Both-null counts as a match here; any difference in value or presence counts as a
mismatch.  Every function below decides agreement through
:func:`~analysis.metrics.matching._is_field_correct`, so the rule is stated once.
For the asserted-value view that gives no credit for agreeing a field is empty, see
:mod:`analysis.metrics.confusion`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from analysis.metrics.field_roles import _get_ontology_constrained_fields
from analysis.metrics.matching import _is_field_correct

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _accuracy(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    field_names: Iterable[str],
    *,
    match_case: bool,
    match_whole_word: bool,
) -> float:
    """Return the fraction of *field_names* the prediction agrees with gold on.

    Returns 0.0 for an empty selection, so a category the template does not use
    reports zero rather than raising.
    """
    names = list(field_names)
    if not names:
        return 0.0
    matches = sum(
        _is_field_correct(predicted, gold, name, match_case=match_case, match_whole_word=match_whole_word)
        for name in names
    )
    return matches / len(names)


def compute_all_field_accuracy(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> float:
    """Compute accuracy of *predicted* metadata against *gold*.

    Accuracy measures all-field record-level agreement: the fraction of gold
    fields where both records agree.  Two fields agree when:

    * both values are missing (``None``), or
    * both values are non-missing and match via ``_values_match()``.

    The denominator is all keys present in *gold*.

    Returns 0.0 when *gold* has no fields.
    """
    return _accuracy(predicted, gold, gold, match_case=match_case, match_whole_word=match_whole_word)


def compute_ontology_constrained_field_accuracy(
    predicted: dict[str, object],
    gold: dict[str, object],
    schema_path: Path,
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
    selected = [name for name in gold if name in ontology_fields]
    return _accuracy(predicted, gold, selected, match_case=match_case, match_whole_word=match_whole_word)


def compute_non_ontology_constrained_field_accuracy(
    predicted: dict[str, object],
    gold: dict[str, object],
    schema_path: Path,
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
    selected = [name for name in gold if name not in ontology_fields]
    return _accuracy(predicted, gold, selected, match_case=match_case, match_whole_word=match_whole_word)


def compute_overall_accuracy(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    schema_path: Path,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> dict[str, float]:
    """Compute accuracy for a single predicted/gold record pair.

    Returns a dict with keys ``ontology_constrained_accuracy``,
    ``non_ontology_constrained_accuracy``, and ``all_field_accuracy``.
    Any metric whose denominator is zero is reported as ``0.0``.
    """
    counts = _compute_field_counts(
        predicted, gold, schema_path, match_case=match_case, match_whole_word=match_whole_word
    )

    ontology_total = counts["ontology_total"]
    non_ontology_total = counts["non_ontology_total"]
    all_correct = counts["ontology_correct"] + counts["non_ontology_correct"]
    all_total = ontology_total + non_ontology_total

    return {
        "ontology_constrained_accuracy": counts["ontology_correct"] / ontology_total if ontology_total else 0.0,
        "non_ontology_constrained_accuracy": (
            counts["non_ontology_correct"] / non_ontology_total if non_ontology_total else 0.0
        ),
        "all_field_accuracy": all_correct / all_total if all_total else 0.0,
    }


def compute_field_results(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    schema_path: Path,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> list[tuple[str, str, bool]]:
    """Return the per-field correctness of one predicted/gold record pair.

    Unlike :func:`compute_overall_accuracy`, which returns only aggregate ratios,
    this function preserves the outcome of *every* gold field.  Each element is a
    ``(field_name, field_type, is_correct)`` tuple where ``field_type`` is either
    ``"ontology"`` or ``"non_ontology"``.  A field is correct when both values are
    missing or both are present and match via
    :func:`~analysis.metrics.matching._values_match`.

    This per-field detail is what paired, field-level significance tests (e.g.
    McNemar's test) require, since they compare baseline vs. agent correctness on
    the same field of the same record.
    """
    ontology_fields = set(_get_ontology_constrained_fields(schema_path))
    return [
        (
            name,
            "ontology" if name in ontology_fields else "non_ontology",
            _is_field_correct(predicted, gold, name, match_case=match_case, match_whole_word=match_whole_word),
        )
        for name in gold
    ]


def _compute_field_counts(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    schema_path: Path,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> dict[str, int]:
    """Return raw correct/total counts split by ontology vs non-ontology fields.

    Returns a dict with keys: ``ontology_correct``, ``ontology_total``,
    ``non_ontology_correct``, ``non_ontology_total``.
    """
    ontology_fields = set(_get_ontology_constrained_fields(schema_path))
    counts = dict.fromkeys(("ontology_correct", "ontology_total", "non_ontology_correct", "non_ontology_total"), 0)

    for name in gold:
        prefix = "ontology" if name in ontology_fields else "non_ontology"
        counts[f"{prefix}_total"] += 1
        if _is_field_correct(predicted, gold, name, match_case=match_case, match_whole_word=match_whole_word):
            counts[f"{prefix}_correct"] += 1

    return counts
