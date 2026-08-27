"""Corpus-level tables built from the saved predictions: accuracy, precision/recall, errors.

Where :mod:`analysis.metrics` scores a single predicted/gold record pair, this package
walks the whole corpus -- every assay in ``ASSAY_ORDER``, every record under
``data/<assay>/`` -- and aggregates those scores into the tables the paper reports.
Nothing here calls an LLM.

The work is split by the question each table answers:

* :mod:`~analysis.data_analysis.accuracy_tables` -- accuracy per file, per assay, and
  pooled over the corpus.
* :mod:`~analysis.data_analysis.precision_recall_tables` -- precision, recall and F1
  over asserted values, pooled and per assay.
* :mod:`~analysis.data_analysis.uncorrected` -- the do-nothing reference point: the
  legacy input scored against gold, with no model in the loop.
* :mod:`~analysis.data_analysis.repetition` -- whether the headline numbers survive
  the corpus's repeated values.
* :mod:`~analysis.data_analysis.errors` -- field-level mismatches, classified and
  aggregated into an error report.
* :mod:`~analysis.data_analysis.availability` -- whether gold's value was already
  written in the input, and whether that predicted the run getting it right.

Every one of them reaches its records through :mod:`analysis.corpus`, which is also
what :mod:`analysis.significance` walks, so the two answer their different questions
over demonstrably the same files.

This module re-exports the whole surface, so ``from analysis.data_analysis import ...``
reaches every name regardless of which module defines it.

pandas is imported inside the functions rather than at module level, so importing this
package stays cheap for callers that only need part of it.
"""

from __future__ import annotations

from analysis.data_analysis.accuracy_tables import (
    apply_metrics,
    create_overall_accuracy_summary,
    create_per_assay_accuracy_summary,
)
from analysis.data_analysis.availability import (
    DERIVED,
    VERBATIM,
    create_availability_summary,
    pool_availability,
)
from analysis.data_analysis.error_taxonomy import (
    CATEGORIES,
    CATEGORY_BY_SUBCATEGORY,
    CLOSE_MATCH_REASONS,
    CONFUSION_CELLS_BY_CATEGORY,
    POOLED_ASSAY,
    SUBCATEGORIES,
    category_shares,
    close_match_reasons,
    collect_field_errors,
    deduplicate_errors,
    reconcile_with_confusion,
    subcategory_shares,
    summarize_error_categories,
    summarize_error_subcategories,
)
from analysis.data_analysis.errors import analyze_prediction_errors, create_error_report
from analysis.data_analysis.precision_recall_tables import (
    create_overall_precision_recall_summary,
    create_per_assay_precision_recall_summary,
)
from analysis.data_analysis.repetition import (
    create_deduplicated_accuracy_summary,
    create_deduplicated_precision_recall_summary,
    create_frequency_split_accuracy_summary,
    create_per_assay_deduplicated_precision_recall_summary,
)
from analysis.data_analysis.uncorrected import create_uncorrected_accuracy_summary

__all__ = [
    "summarize_error_categories",
    "summarize_error_subcategories",
    "reconcile_with_confusion",
    "collect_field_errors",
    "deduplicate_errors",
    "CATEGORIES",
    "SUBCATEGORIES",
    "CATEGORY_BY_SUBCATEGORY",
    "CLOSE_MATCH_REASONS",
    "CONFUSION_CELLS_BY_CATEGORY",
    "close_match_reasons",
    "POOLED_ASSAY",
    "category_shares",
    "subcategory_shares",
    "DERIVED",
    "VERBATIM",
    "analyze_prediction_errors",
    "apply_metrics",
    "create_availability_summary",
    "pool_availability",
    "create_deduplicated_accuracy_summary",
    "create_deduplicated_precision_recall_summary",
    "create_per_assay_deduplicated_precision_recall_summary",
    "create_error_report",
    "create_frequency_split_accuracy_summary",
    "create_overall_accuracy_summary",
    "create_overall_precision_recall_summary",
    "create_per_assay_accuracy_summary",
    "create_per_assay_precision_recall_summary",
    "create_uncorrected_accuracy_summary",
]
