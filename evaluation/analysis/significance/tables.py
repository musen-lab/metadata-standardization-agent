"""The reported tables: one row per assay, and the pooled rows beneath them.

The interval tables collect the paired outcomes once, hand them to
the estimators, and formatting the results as ``point [lo, hi]`` strings.  The columns
are strings rather than numbers because these tables are read, not computed on; the
functions that produce intervals are :mod:`~analysis.significance.bootstrap` and
:mod:`~analysis.significance.hypothesis_tests`.
The pre/post comparison returns unrounded numeric estimates and changes so that
presentation rounding cannot affect the calculated differences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays
from analysis.metrics import precision_recall_f1
from analysis.significance.bootstrap import (
    bootstrap_pooled_accuracy,
    bootstrap_prf,
    cluster_bootstrap_prf,
    cluster_bootstrap_prf_delta,
)
from analysis.significance.hypothesis_tests import (
    adjust_pvalues,
    paired_permutation_prf,
)
from analysis.significance.paired_data import CATEGORIES, CATEGORY_LABELS, PairedData, collect_paired_data
from analysis.significance.single_condition import collect_single_condition_data

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path

    import pandas as pd


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    if mean != mean:  # nan
        return "-"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_p(p: float) -> str:
    if p != p:  # nan
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _pooled_data(
    data_root: str | Path,
    model: str,
    baseline: str,
    system: str,
    *,
    run: int = 1,
    excluded_fields_by_assay: Mapping[str, Collection[str]] | None = None,
) -> PairedData:
    """Paired outcomes for every assay, accumulated into one :class:`PairedData`."""
    pooled = PairedData()
    for assay in iter_assays(data_root):
        pooled.extend(
            collect_paired_data(
                data_root,
                model,
                assay.key,
                baseline=baseline,
                system=system,
                run=run,
                excluded_fields=(excluded_fields_by_assay or {}).get(assay.key, ()),
            )
        )
    return pooled


def build_precision_recall_table(
    data_root: str | Path,
    model: str,
    *,
    baseline: str = "baseline",
    system: str = "arms-agent",
    run: int = 1,
    excluded_fields_by_assay: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame:
    """Precision, recall and F1 with cluster-bootstrap CIs, pooled across assays.

    One row per (field category, metric), with the baseline and ARMS estimates and
    the paired difference, each as ``point [lo, hi]``.  Records are the resampling
    unit throughout, and the three metrics share each replicate's resample, so the
    baseline, ARMS and difference columns of a row are mutually consistent.
    An optional assay-keyed exclusion map removes the same fields from both
    methods in every record. The default retains the original evaluation.
    """
    import pandas as pd

    pooled = _pooled_data(
        data_root,
        model,
        baseline,
        system,
        run=run,
        excluded_fields_by_assay=excluded_fields_by_assay,
    )

    rows = []
    for category in CATEGORIES:
        confusion = pooled.record_confusion[category]
        baseline_ci = cluster_bootstrap_prf(confusion, "baseline")
        system_ci = cluster_bootstrap_prf(confusion, "system")
        delta = cluster_bootstrap_prf_delta(confusion)
        for metric in ("precision", "recall", "f1"):
            rows.append(
                {
                    "category": CATEGORY_LABELS[category],
                    "metric": metric,
                    "n_records": len(confusion),
                    baseline: _fmt_ci(*baseline_ci[metric]),
                    system: _fmt_ci(*system_ci[metric]),
                    "difference": _fmt_ci(*delta[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_pre_post_precision_recall_table(
    data_root: str | Path,
    model: str,
    *,
    excluded_fields: Mapping[str, Collection[str]],
    baseline: str = "baseline",
    system: str = "arms-agent",
    run: int = 1,
) -> pd.DataFrame:
    """Effect of assay-wide exclusions on each method, per assay and pooled.

    Reuse the paired-data collector and the existing TP/FP/FN score definitions.
    Changes are post minus pre for the *same method*, not system minus baseline.
    They describe different sets of scored fields, not a change in predictions.
    Return unrounded point estimates; no uncertainty test of the change is implied.
    A category entirely excluded post-adjudication has no estimate (NaN), rather
    than an artificial zero. Counts name records with fields in each category.
    """
    import pandas as pd

    views: list[tuple[str, PairedData, PairedData]] = []
    pooled_pre, pooled_post = PairedData(), PairedData()
    for assay in iter_assays(data_root):
        pre = collect_paired_data(data_root, model, assay.key, baseline=baseline, system=system, run=run)
        post = collect_paired_data(
            data_root,
            model,
            assay.key,
            baseline=baseline,
            system=system,
            run=run,
            excluded_fields=excluded_fields.get(assay.key, ()),
        )
        views.append((assay.label, pre, post))
        pooled_pre.extend(pre)
        pooled_post.extend(post)
    views.append(("All assays", pooled_pre, pooled_post))

    def scores(data: PairedData, category: str, offset: int) -> dict[str, float]:
        counts = data.record_confusion[category]
        if not counts:
            return dict.fromkeys(("precision", "recall"), float("nan"))
        return precision_recall_f1(
            {key: sum(row[offset + i] for row in counts) for i, key in enumerate(("TP", "FP", "FN"))}
        )

    columns = ["assay", "category", "metric", "n_records_pre", "n_records_post"]
    columns += [f"{condition} {stage}" for condition in (baseline, system) for stage in ("pre", "post", "change")]
    rows: list[dict[str, Any]] = []
    for assay, pre, post in views:
        for category in CATEGORIES:
            if not pre.record_confusion[category]:
                continue
            estimates = {
                condition: (scores(pre, category, offset), scores(post, category, offset))
                for condition, offset in ((baseline, 0), (system, 3))
            }
            for metric in ("precision", "recall"):
                row: dict[str, Any] = {
                    "assay": assay,
                    "category": CATEGORY_LABELS[category],
                    "metric": metric,
                    "n_records_pre": len(pre.record_confusion[category]),
                    "n_records_post": len(post.record_confusion[category]),
                }
                for condition, (before, after) in estimates.items():
                    row[f"{condition} pre"] = before[metric]
                    row[f"{condition} post"] = after[metric]
                    row[f"{condition} change"] = after[metric] - before[metric]
                rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_single_condition_table(data_root: str | Path, model: str, condition: str, *, run: int = 1) -> pd.DataFrame:
    """Accuracy and micro precision/recall/F1 with bootstrap CIs, for one condition alone.

    One row per field category.  Needing no comparison condition, this covers every
    condition in a sweep -- an ablation included -- where :func:`build_precision_recall_table` can
    only describe the two arms it pairs.

    Records are the resampling unit throughout, and the point estimates are the ones
    :mod:`analysis.data_analysis` already reports, so each interval qualifies a number
    that appears in the tables above rather than a differently-weighted one.
    """
    import pandas as pd

    data = collect_single_condition_data(data_root, model, condition, run=run)

    rows = []
    for category in CATEGORIES:
        counts = data.record_counts[category]
        prf = bootstrap_prf(data.record_confusion[category])
        rows.append(
            {
                "condition": condition,
                "category": CATEGORY_LABELS[category],
                "n_records": len(counts),
                "accuracy": _fmt_ci(*bootstrap_pooled_accuracy(counts)),
                "precision": _fmt_ci(*prf["precision"]),
                "recall": _fmt_ci(*prf["recall"]),
                "f1": _fmt_ci(*prf["f1"]),
            }
        )
    return pd.DataFrame(rows)


def build_per_assay_precision_recall_table(
    data_root: str | Path,
    model: str,
    *,
    correction: str = "holm",
    alpha: float = 0.05,
    baseline: str = "baseline",
    system: str = "arms-agent",
    run: int = 1,
    excluded_fields_by_assay: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame:
    """Per-assay precision and recall: ARMS against baseline, with a corrected p-value.

    One row per (assay, field category, metric), holding both conditions' intervals, the
    paired difference with its interval, the record-clustered permutation p-value, and
    that p-value corrected over every row of the table.

    *baseline* and *system* name the two conditions compared, both read from run *run*, so the same
    table answers "does ARMS beat baseline" or "does ARMS beat an ablation"
    without changing anything else.  Every difference is system minus baseline,
    and the two estimate columns are named after the conditions.

    Every row is one hypothesis, and they are tested together, so the correction family
    is the whole table rather than whichever cell looked best.  *correction* is
    ``"holm"`` for a claim about a specific assay or ``"fdr_bh"`` for a claim about the
    corpus; see :func:`~analysis.significance.hypothesis_tests.adjust_pvalues`.

    F1 is deliberately absent from the family: it is a function of the two metrics
    already there, so testing it as well would inflate the correction with a redundant
    hypothesis.  Its interval is in :func:`build_precision_recall_table`.

    The ``significant`` column applies *alpha* to the corrected p-value.  Read it with
    ``n_records`` beside it: the permutation null has only ``2**n`` arrangements, so
    fewer than six differing records cannot reach 0.05 however large the effect.
    ``excluded_fields_by_assay`` uses assay keys (for example, ``atacseq``) and
    removes those fields from every record for both methods before resampling.
    """
    import pandas as pd

    rows = []
    for assay in iter_assays(data_root):
        data = collect_paired_data(
            data_root,
            model,
            assay.key,
            baseline=baseline,
            system=system,
            run=run,
            excluded_fields=(excluded_fields_by_assay or {}).get(assay.key, ()),
        )
        for category in CATEGORIES:
            confusion = data.record_confusion[category]
            if not confusion:
                continue
            baseline_ci = cluster_bootstrap_prf(confusion, "baseline")
            system_ci = cluster_bootstrap_prf(confusion, "system")
            delta = cluster_bootstrap_prf_delta(confusion)
            test = paired_permutation_prf(confusion)
            for metric in ("precision", "recall"):
                rows.append(
                    {
                        "assay": assay.label,
                        "category": CATEGORY_LABELS[category],
                        "metric": metric,
                        "n_records": len(confusion),
                        "n_differing": test["n_effective"],
                        baseline: _fmt_ci(*baseline_ci[metric]),
                        system: _fmt_ci(*system_ci[metric]),
                        "difference": _fmt_ci(*delta[metric]),
                        "perm_p": test[metric]["pvalue"],
                    }
                )

    if not rows:
        return pd.DataFrame(rows)

    adjusted = adjust_pvalues([row["perm_p"] for row in rows], method=correction)
    for row, corrected in zip(rows, adjusted, strict=True):
        row["p_adjusted"] = corrected
        row["significant"] = bool(corrected < alpha)
        row["perm_p"] = _fmt_p(row["perm_p"])
        row["p_adjusted"] = _fmt_p(corrected)
    return pd.DataFrame(rows)
