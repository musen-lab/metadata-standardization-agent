"""The numbers the precision/recall figures are drawn from, and the checks they share.

:mod:`plots.pr_space` and :mod:`plots.pr_bars` draw the same measurement two ways, so they
agree here on which rows there are, what the pooled row is called, and what counts as a
usable set of arguments.  Left in each figure, those three would be the first things to
drift apart.
"""

from __future__ import annotations

from analysis.data_analysis import (
    create_overall_precision_recall_summary,
    create_per_assay_precision_recall_summary,
)
from assays import ASSAY_ORDER
from plots.marks import FIELD_TYPE_LABELS

#: What the row is called when the corpus is drawn as one row rather than broken out.
POOLED_LABEL = "All assays"


def _check_pr_arguments(
    baseline_runs: tuple[str, ...], system_runs: tuple[str, ...], field_types: tuple[str, ...]
) -> None:
    """Reject an empty group or an unknown field type before anything is read."""
    if not baseline_runs or not system_runs:
        raise ValueError("baseline_runs and system_runs each need at least one run")
    unknown = [name for name in field_types if name not in FIELD_TYPE_LABELS]
    if unknown:
        raise ValueError(f"field_types must be drawn from {tuple(FIELD_TYPE_LABELS)}, got {unknown}")


def _pr_scores(
    data_root: str,
    model: str,
    runs: tuple[str, ...],
    field_type: str,
    assays: tuple[str, ...],
) -> tuple[list[str], dict[tuple[str, str], tuple[float, float]]]:
    """The (recall, precision) of every run for every row.

    *assays* left empty pools the corpus into a single row, pooled over every
    gold/prediction pair rather than averaged over per-assay ratios, since the assays
    differ in size by more than tenfold.  Returns the row labels and a lookup keyed by
    ``(row, run)``.
    """
    labels = dict(ASSAY_ORDER)
    unknown = [key for key in assays if key not in labels]
    if unknown:
        raise ValueError(f"Unknown assay key(s): {unknown}")

    if not assays:
        summaries = {
            run: create_overall_precision_recall_summary(data_root, model, run).set_index("category") for run in runs
        }
        scores = {
            (POOLED_LABEL, run): (summary.loc[field_type, "recall"], summary.loc[field_type, "precision"])
            for run, summary in summaries.items()
        }
        return [POOLED_LABEL], scores

    frames = {
        run: create_per_assay_precision_recall_summary(data_root, model, run, category=field_type).set_index("assay")
        for run in runs
    }
    wanted = {labels[key] for key in assays}
    rows = [
        label
        for _key, label in ASSAY_ORDER
        if label in wanted and all(label in frame.index for frame in frames.values())
    ]
    if not rows:
        raise ValueError(f"No requested assay has predictions for every one of {runs}")
    scores = {
        (row, run): (frames[run].loc[row, "recall"], frames[run].loc[row, "precision"]) for row in rows for run in runs
    }
    return rows, scores
