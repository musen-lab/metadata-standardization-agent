"""Evaluation metrics for comparing agent output against gold standard.

Metrics:
    - Field completeness: proportion of gold-standard non-missing fields that are
      also present (non-empty) in the predicted output. No value comparison; only
      checks presence.
    - Field-value accuracy: among fields present in both predicted and gold,
      fraction with identical values (exact match).
"""

from __future__ import annotations

from typing import Any


def _is_missing(value: Any) -> bool:
    """Return ``True`` if *value* is considered missing.

    A field value is missing when it is ``None``.  Empty strings, empty lists,
    and other falsy-but-not-None values are considered present.
    """
    return value is None


def compute_accuracy(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    """Compute field-value accuracy of *predicted* metadata against *gold*.

    The denominator is the set of fields that are non-missing in **both**
    *predicted* and *gold*.  The numerator counts how many of those have
    identical values (exact match).

    Returns 0.0 when there are no comparable fields.
    """
    comparable = {k for k in gold if not _is_missing(gold[k]) and not _is_missing(predicted.get(k))}
    if not comparable:
        return 0.0
    matching = sum(1 for k in comparable if predicted[k] == gold[k])
    return matching / len(comparable)


def compute_completeness(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    """Compute field completeness of *predicted* metadata against *gold*.

    Completeness is the proportion of gold-standard non-missing fields that are
    also non-missing in the predicted output.  Only field presence matters; value
    correctness is ignored.

    Returns 0.0 when the gold standard has no non-missing fields.
    """
    non_missing_gold = {k for k, v in gold.items() if not _is_missing(v)}
    if not non_missing_gold:
        return 0.0
    non_missing_pred = {k for k, v in predicted.items() if not _is_missing(v)}
    return len(non_missing_gold & non_missing_pred) / len(non_missing_gold)
