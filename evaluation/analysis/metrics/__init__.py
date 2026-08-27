"""Evaluation metrics for comparing agent output against gold standard.

Metrics:
    - All-field accuracy: overall record-level agreement across all fields in the
      gold standard.  Both-null counts as a match; any difference in value or
      presence counts as a mismatch.
    - Precision, recall and F1 over asserted values, via
      :func:`compute_field_confusion` and :func:`precision_recall_f1`.  Unlike
      accuracy, these exclude the both-blank case, so agreeing that a field is
      empty earns no credit.  See ``docs/precision-recall-plan.md``.

The implementation is split across :mod:`~analysis.metrics.matching` (what counts as
blank, equal, and correct), :mod:`~analysis.metrics.field_roles` (which fields are
ontology-constrained), :mod:`~analysis.metrics.accuracy` and
:mod:`~analysis.metrics.confusion`.  This module re-exports the whole surface, so
``from analysis.metrics import ...`` reaches every name regardless of which module
defines it.  The underscore-prefixed names are re-exported too: the analysis modules
and the tests already depend on them.
"""

from __future__ import annotations

from analysis.metrics.accuracy import (
    _compute_field_counts,
    compute_all_field_accuracy,
    compute_field_results,
    compute_non_ontology_constrained_field_accuracy,
    compute_ontology_constrained_field_accuracy,
    compute_overall_accuracy,
)
from analysis.metrics.confusion import (
    CONFUSION_CATEGORIES,
    CONFUSION_KEYS,
    compute_field_confusion,
    precision_recall_f1,
)
from analysis.metrics.field_roles import _get_ontology_constrained_fields, _get_permissible_values
from analysis.metrics.matching import _is_field_correct, _is_missing, _values_match

__all__ = [
    "CONFUSION_CATEGORIES",
    "CONFUSION_KEYS",
    "_compute_field_counts",
    "_get_ontology_constrained_fields",
    "_get_permissible_values",
    "_is_field_correct",
    "_is_missing",
    "_values_match",
    "compute_all_field_accuracy",
    "compute_field_confusion",
    "compute_field_results",
    "compute_non_ontology_constrained_field_accuracy",
    "compute_ontology_constrained_field_accuracy",
    "compute_overall_accuracy",
    "precision_recall_f1",
]
