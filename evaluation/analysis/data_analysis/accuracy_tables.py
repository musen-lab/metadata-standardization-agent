"""Accuracy tables: the fraction of gold fields the prediction agrees with.

Three views of the same records -- one row per prediction file
(:func:`apply_metrics`), one row per assay (:func:`create_per_assay_accuracy_summary`)
and one row for the whole corpus (:func:`create_overall_accuracy_summary`).  The
corpus row pools raw counts instead of averaging the per-assay ratios: the assays
differ greatly in size -- Lightsheet has 9 records against ATACseq's 100 -- so
averaging ratios would over-weight the small ones.

Both-blank counts as agreement throughout.  For the asserted-value view that gives no
credit for agreeing a field is empty, see
:mod:`analysis.data_analysis.precision_recall_tables`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, iter_predictions
from analysis.metrics import (
    _compute_field_counts,
    compute_all_field_accuracy,
    compute_non_ontology_constrained_field_accuracy,
    compute_ontology_constrained_field_accuracy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd

#: The per-file accuracy columns :func:`apply_metrics` produces.
ACCURACY_COLUMNS = (
    "ontology_constrained_field_accuracy",
    "non_ontology_constrained_field_accuracy",
    "all_field_accuracy",
)

#: The raw counts an accuracy ratio is computed from, split by field category.
_TALLY_KEYS = ("ontology_correct", "ontology_total", "non_ontology_correct", "non_ontology_total")


def _new_tally() -> dict[str, int]:
    """Return a zeroed correct/total tally, keyed as :func:`_compute_field_counts` keys it."""
    return dict.fromkeys(_TALLY_KEYS, 0)


def _accuracy_row(tally: Mapping[str, int]) -> dict[str, float]:
    """Turn pooled correct/total counts into the three accuracy ratios.

    A category with no fields reports ``0.0`` rather than raising, matching how the
    per-record metrics report an empty category.  Values are not rounded, so the
    caller controls display precision.
    """
    ontology_total = tally["ontology_total"]
    non_ontology_total = tally["non_ontology_total"]
    total_correct = tally["ontology_correct"] + tally["non_ontology_correct"]
    total_fields = ontology_total + non_ontology_total

    return {
        "ontology_constrained_accuracy": tally["ontology_correct"] / ontology_total if ontology_total else 0.0,
        "non_ontology_constrained_accuracy": (
            tally["non_ontology_correct"] / non_ontology_total if non_ontology_total else 0.0
        ),
        "all_field_accuracy": total_correct / total_fields if total_fields else 0.0,
    }


def apply_metrics(input_dir: Path, gold_dir: Path, schema_path: Path) -> pd.DataFrame:
    """Compare predicted outputs in *input_dir* against gold standards in *gold_dir*.

    Returns one row per prediction file that has a gold counterpart, with the three
    :data:`ACCURACY_COLUMNS` alongside the filename.
    """
    import pandas as pd

    results: list[dict[str, Any]] = []
    for input_file, predicted, gold in iter_predictions(input_dir, gold_dir):
        results.append(
            {
                "input_file": input_file.name,
                "ontology_constrained_field_accuracy": compute_ontology_constrained_field_accuracy(
                    predicted, gold, schema_path
                ),
                "non_ontology_constrained_field_accuracy": compute_non_ontology_constrained_field_accuracy(
                    predicted, gold, schema_path
                ),
                "all_field_accuracy": compute_all_field_accuracy(predicted, gold),
            }
        )

    return pd.DataFrame(results)


def create_per_assay_accuracy_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Compute average accuracy per assay across all samples.

    Iterates over assays defined in ``ASSAY_ORDER``, calls :func:`apply_metrics`
    for each, and returns a single DataFrame with one row per assay containing
    the mean accuracy for each metric.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for assay in iter_assays(data_root):
        df = apply_metrics(assay.output_dir(model, run_type), assay.gold_dir, assay.schema_path)
        if df.empty:
            continue
        means = df[list(ACCURACY_COLUMNS)].mean()
        rows.append(
            {
                "assay": assay.label,
                **{column: round(means[column], decimal_places) for column in ACCURACY_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def create_overall_accuracy_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Compute aggregate overall accuracy across all assays from raw counts.

    Accumulates raw correct/total counts from every predicted/gold file pair
    across all assays and computes accuracy ratios once from the totals, rather
    than averaging per-file or per-assay ratios.  Returns a single-row DataFrame
    with columns ``ontology_constrained_accuracy``,
    ``non_ontology_constrained_accuracy``, and ``all_field_accuracy``.
    """
    import pandas as pd

    tally = _new_tally()
    for assay in iter_assays(data_root):
        for _pred_file, predicted, gold in iter_predictions(assay.output_dir(model, run_type), assay.gold_dir):
            counts = _compute_field_counts(predicted, gold, assay.schema_path)
            for key in _TALLY_KEYS:
                tally[key] += counts[key]

    return pd.DataFrame([{key: round(value, decimal_places) for key, value in _accuracy_row(tally).items()}])
