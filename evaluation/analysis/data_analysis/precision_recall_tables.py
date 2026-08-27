"""Precision, recall and F1 tables over asserted values.

Both tables are micro-averaged: raw confusion counts are accumulated over every
gold/predicted pair and the ratios computed once from the totals.  Micro rather than
macro because the assays differ greatly in size -- Lightsheet has 9 records against
ATACseq's 100 -- and averaging per-assay ratios over-weights the small ones.

Unlike the accuracy tables, agreeing that a field is empty earns no credit here; see
:mod:`analysis.metrics.confusion` for what each counter means.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, iter_pairs
from analysis.metrics import (
    CONFUSION_CATEGORIES,
    CONFUSION_KEYS,
    compute_field_confusion,
    precision_recall_f1,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from analysis.corpus import Assay

logger = logging.getLogger(__name__)


def _accumulate_confusion(
    assays: Iterable[Assay],
    model: str,
    run_type: str,
) -> tuple[dict[str, dict[str, int]], int, int]:
    """Pool confusion counts over every gold/predicted pair in *assays*.

    Iterates the gold files rather than the predictions, so a gold record with no
    prediction is a visible skip rather than an invisible omission -- the
    ``TP + FN`` total has to correspond to the gold fields actually evaluated.

    Returns ``(counts, n_pairs, n_skipped)`` where *counts* is keyed by
    :data:`CONFUSION_CATEGORIES`, *n_pairs* is the number of pairs evaluated and
    *n_skipped* the number of gold records whose prediction was absent.
    """
    counts = {category: dict.fromkeys(CONFUSION_KEYS, 0) for category in CONFUSION_CATEGORIES}
    n_pairs = 0
    n_skipped = 0

    for assay in assays:
        if not assay.has_gold:
            continue

        for _gold_file, gold, predicted in iter_pairs(assay.gold_dir, assay.output_dir(model, run_type)):
            if predicted is None:
                n_skipped += 1
                continue
            record_counts = compute_field_confusion(predicted, gold, assay.schema_path)
            for category in CONFUSION_CATEGORIES:
                for key in CONFUSION_KEYS:
                    counts[category][key] += record_counts[category][key]
            n_pairs += 1

    if n_skipped:
        logger.warning(
            "Precision/recall summary skipped %d gold record(s) with no %s/%s prediction (%d pair(s) evaluated)",
            n_skipped,
            model,
            run_type,
            n_pairs,
        )
    return counts, n_pairs, n_skipped


def _scores_row(counts: dict[str, int], n_pairs: int, decimal_places: int) -> dict[str, Any]:
    """Build the shared columns of both tables from one category's counters.

    Counts are reported alongside the ratios because the denominators are needed to
    judge them.
    """
    scores = precision_recall_f1(counts)
    return {
        "n_records": n_pairs,
        **{key: counts[key] for key in CONFUSION_KEYS},
        "precision": round(scores["precision"], decimal_places),
        "recall": round(scores["recall"], decimal_places),
        "f1": round(scores["f1"], decimal_places),
    }


def create_overall_precision_recall_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 3,
) -> pd.DataFrame:
    """Pool precision, recall and F1 across all assays, one row per field category.

    Returns one row per entry in :data:`CONFUSION_CATEGORIES` with the raw
    ``TP``/``FP``/``FN``/``TN``/``insertions``/``deletions``/``substitutions`` counts and the
    derived ``precision``, ``recall`` and ``f1``.
    """
    import pandas as pd

    counts, n_pairs, _ = _accumulate_confusion(iter_assays(data_root), model, run_type)

    return pd.DataFrame(
        [
            {"category": category, **_scores_row(counts[category], n_pairs, decimal_places)}
            for category in CONFUSION_CATEGORIES
        ]
    )


def create_per_assay_precision_recall_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    category: str = "all",
    decimal_places: int = 3,
) -> pd.DataFrame:
    """Precision, recall and F1 per assay, micro-averaged within each assay.

    *category* selects the field grouping and must be one of
    :data:`CONFUSION_CATEGORIES`.  Returns one row per assay in ``ASSAY_ORDER``,
    skipping assays with no evaluated pairs.
    """
    import pandas as pd

    if category not in CONFUSION_CATEGORIES:
        raise ValueError(f"category must be one of {CONFUSION_CATEGORIES}, got {category!r}")

    rows: list[dict[str, Any]] = []
    for assay in iter_assays(data_root):
        counts, n_pairs, _ = _accumulate_confusion([assay], model, run_type)
        if not n_pairs:
            continue
        rows.append({"assay": assay.label, **_scores_row(counts[category], n_pairs, decimal_places)})
    return pd.DataFrame(rows)
