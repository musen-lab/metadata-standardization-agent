"""Evaluation framework for comparing agent output against gold standard.

Metrics:
    - Precision: fraction of output fields that match the gold standard.
    - Completeness: fraction of gold-standard fields correctly present in predicted output.
"""

from __future__ import annotations

from typing import Any


def compute_precision(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    """Compute field-level precision of predicted metadata against gold standard.

    TODO: Implement detailed field comparison logic.
    """
    if not predicted:
        return 0.0
    matching = sum(1 for k, v in predicted.items() if gold.get(k) == v)
    return matching / len(predicted) if predicted else 0.0


def compute_completeness(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    """Compute completeness of predicted metadata against gold standard.

    Completeness is the fraction of gold-standard fields that are correctly
    present in the predicted output (recall-like metric).
    """
    if not gold:
        return 0.0
    matching = sum(1 for k, v in gold.items() if predicted.get(k) == v)
    return matching / len(gold)
