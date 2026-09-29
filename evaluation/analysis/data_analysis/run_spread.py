"""How far precision and recall move between repeated runs of one condition.

The precision/recall tables score one run.  Repeat it and the scores move, because the model
does not answer the same way twice.  This module scores every run with those same tables and
reports, per assay and field type, the mean over the runs and the lowest and highest run --
the numbers behind the claim that a result is, or is not, an accident of one run.

The spread is the range, lowest run to highest, rather than a standard deviation: with a
handful of runs a standard deviation is itself poorly estimated, where the range is exactly
what happened.  :mod:`analysis.data_analysis.stability` says how often the answers behind
these scores changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.corpus import iter_assays
from analysis.data_analysis.error_taxonomy import POOLED_ASSAY
from analysis.data_analysis.precision_recall_tables import (
    create_overall_precision_recall_summary,
    create_per_assay_precision_recall_summary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

#: The metrics whose spread is reported.  F1 is left out: it follows from the other two.
SPREAD_METRICS = ("precision", "recall")

SPREAD_COLUMNS = ["assay", "field_type", "metric", "n_records", "mean", "min", "max", "range"]

RUN_SCORE_COLUMNS = ["assay", "field_type", "metric", "n_records", "run", "score"]

#: Decimal places kept while the runs are scored, so rounding happens once, on the spread.
_UNROUNDED = 12


def collect_run_scores(
    data_root: str,
    model: str,
    condition: str,
    *,
    runs: Sequence[int],
    field_types: Sequence[str] = ("ontology", "non_ontology"),
) -> pd.DataFrame:
    """Precision and recall of *condition* in each of *runs*, unrounded: the numbers a spread is taken over.

    One row per (assay, field type, metric, run), the assays in ``ASSAY_ORDER`` and then a
    ``POOLED_ASSAY`` row pooled over every record.  Columns are :data:`RUN_SCORE_COLUMNS`:
    ``n_records`` is how many records stand behind the row in the first run.  Each run is
    scored exactly as :func:`~analysis.data_analysis.create_per_assay_precision_recall_summary`
    scores it -- instance-weighted, over every field instance rather than deduplicated.

    An assay counts only when every run in *runs* scored it, so a run with no predictions
    on disk at all -- not made yet -- leaves the table empty; it is then not scored, which
    keeps the scoring's warnings about every missing record out of the way.

    Raises:
        ValueError: If *runs* holds fewer than two runs, or the same run twice.
    """
    import pandas as pd

    if len(runs) < 2:
        raise ValueError(f"A spread needs at least two runs, got {list(runs)}.")
    if len(set(runs)) != len(runs):
        raise ValueError(f"Each run may be named once, got {list(runs)}.")

    if any(not _made(data_root, model, condition, run) for run in runs):
        return pd.DataFrame(columns=RUN_SCORE_COLUMNS)

    pooled = {
        run: create_overall_precision_recall_summary(
            data_root, model, condition, decimal_places=_UNROUNDED, run=run
        ).set_index("category")
        for run in runs
    }

    rows: list[dict[str, object]] = []
    for field_type in field_types:
        per_run = {
            run: create_per_assay_precision_recall_summary(
                data_root, model, condition, category=field_type, decimal_places=_UNROUNDED, run=run
            )
            for run in runs
        }
        first = per_run[runs[0]]
        scored_everywhere = [
            assay
            for assay in (first["assay"] if not first.empty else [])
            if all(not frame.empty and assay in set(frame["assay"]) for frame in per_run.values())
        ]
        sources = [
            (assay, {run: frame.set_index("assay").loc[assay] for run, frame in per_run.items()})
            for assay in scored_everywhere
        ]
        if scored_everywhere:
            sources.append((POOLED_ASSAY, {run: summary.loc[field_type] for run, summary in pooled.items()}))

        for assay, scores in sources:
            n_records = int(scores[runs[0]]["n_records"])
            for metric in SPREAD_METRICS:
                rows.extend(
                    {
                        "assay": assay,
                        "field_type": field_type,
                        "metric": metric,
                        "n_records": n_records,
                        "run": run,
                        "score": float(scores[run][metric]),
                    }
                    for run in runs
                )
    return pd.DataFrame(rows, columns=RUN_SCORE_COLUMNS)


def create_run_spread_summary(
    data_root: str,
    model: str,
    condition: str,
    *,
    runs: Sequence[int],
    field_types: Sequence[str] = ("ontology", "non_ontology"),
    decimal_places: int = 3,
) -> pd.DataFrame:
    """Precision and recall of *condition* over *runs*: the mean, the lowest and highest run.

    One row per (assay, field type, metric), the assays in ``ASSAY_ORDER`` and then a
    ``POOLED_ASSAY`` row pooled over every record -- pooled, as the single-run tables pool
    it, rather than averaged over the assays.  Columns are :data:`SPREAD_COLUMNS`:
    ``n_records`` is how many records stand behind the row in the first run, and ``range``
    is ``max - min``, how far apart the lowest and highest run landed.

    The runs are :func:`collect_run_scores`'s, so an assay counts only when every run scored
    it, and a mean here is the mean of numbers the single-run tables print.

    Raises:
        ValueError: If *runs* holds fewer than two runs, or the same run twice.
    """
    import pandas as pd

    scores = collect_run_scores(data_root, model, condition, runs=runs, field_types=field_types)
    if scores.empty:
        return pd.DataFrame(columns=SPREAD_COLUMNS)
    spread = (
        scores.groupby(["assay", "field_type", "metric", "n_records"], sort=False)["score"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    spread["range"] = spread["max"] - spread["min"]
    return spread[SPREAD_COLUMNS].round(decimal_places)


def _made(data_root: str, model: str, condition: str, run: int) -> bool:
    """Whether run *run* of *condition* has any predictions on disk, in any assay."""
    return any(any(assay.output_dir(model, condition, run=run).glob("*.json")) for assay in iter_assays(data_root))
