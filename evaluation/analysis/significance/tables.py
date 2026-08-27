"""The reported tables: one row per assay, and the pooled rows beneath them.

Everything here is presentation -- collecting the paired outcomes once, handing them to
the estimators, and formatting the results as ``point [lo, hi]`` strings.  The columns
are strings rather than numbers because these tables are read, not computed on; the
functions that produce numbers are :mod:`~analysis.significance.bootstrap` and
:mod:`~analysis.significance.hypothesis_tests`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.corpus import iter_assays
from analysis.significance.bootstrap import (
    bootstrap_ci,
    bootstrap_pooled_accuracy,
    bootstrap_prf,
    cluster_bootstrap_pooled,
    cluster_bootstrap_prf,
    cluster_bootstrap_prf_delta,
)
from analysis.significance.hypothesis_tests import (
    adjust_pvalues,
    paired_mcnemar,
    paired_permutation,
    paired_permutation_prf,
    paired_wilcoxon,
)
from analysis.significance.paired_data import CATEGORIES, CATEGORY_LABELS, PairedData, collect_paired_data
from analysis.significance.single_run import collect_single_run_data

if TYPE_CHECKING:
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


def _pooled_data(data_root: str | Path, model: str, baseline_run: str, system_run: str) -> PairedData:
    """Paired outcomes for every assay, accumulated into one :class:`PairedData`."""
    pooled = PairedData()
    for assay in iter_assays(data_root):
        pooled.extend(
            collect_paired_data(data_root, model, assay.key, baseline_run=baseline_run, system_run=system_run)
        )
    return pooled


def build_per_assay_table(
    data_root: str | Path,
    model: str,
    category: str = "all",
    *,
    baseline_run: str = "baseline",
    system_run: str = "agent-tool",
) -> pd.DataFrame:
    """One row per assay for *category*: both runs' mean+CI, Wilcoxon p, McNemar b/c/p.

    The two columns are named after *baseline_run* and *system_run*, which may be
    any two conditions present under ``output/<model>/``.
    """
    import pandas as pd

    rows = []
    for assay in iter_assays(data_root):
        data = collect_paired_data(data_root, model, assay.key, baseline_run=baseline_run, system_run=system_run)
        pairs = data.record_acc[category]
        if not pairs:
            continue
        b_mean, b_lo, b_hi = bootstrap_ci([b for b, _ in pairs])
        a_mean, a_lo, a_hi = bootstrap_ci([a for _, a in pairs])
        _, w_p, _ = paired_wilcoxon(pairs)
        mc = paired_mcnemar(data.field_outcomes[category])
        perm = paired_permutation(data.record_discordant[category])
        rows.append(
            {
                "assay": assay.label,
                "n_records": len(pairs),
                baseline_run: _fmt_ci(b_mean, b_lo, b_hi),
                system_run: _fmt_ci(a_mean, a_lo, a_hi),
                "wilcoxon_p": _fmt_p(w_p),
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_p": _fmt_p(mc["pvalue"]),
                "perm_p": _fmt_p(perm["pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def build_overall_table(
    data_root: str | Path,
    model: str,
    *,
    baseline_run: str = "baseline",
    system_run: str = "agent-tool",
) -> pd.DataFrame:
    """Pooled-across-assays results, one row per field category.

    Accuracy is reported as *pooled* (field-weighted) accuracy with a record-level
    cluster-bootstrap CI, matching the paper's overall bottom row.  Significance is
    the paired Wilcoxon (per-record) and McNemar (per-field) tests, plus a
    record-clustered permutation test that treats the record -- not the field -- as
    the independent unit, so duplicated/clustered fields do not inflate the result.
    """
    import pandas as pd

    pooled = _pooled_data(data_root, model, baseline_run, system_run)

    rows = []
    for category in CATEGORIES:
        counts = pooled.record_counts[category]
        pairs = pooled.record_acc[category]
        b_mean, b_lo, b_hi = cluster_bootstrap_pooled(counts, "baseline")
        a_mean, a_lo, a_hi = cluster_bootstrap_pooled(counts, "system")
        _, w_p, w_n = paired_wilcoxon(pairs)
        mc = paired_mcnemar(pooled.field_outcomes[category])
        perm = paired_permutation(pooled.record_discordant[category])
        rows.append(
            {
                "category": CATEGORY_LABELS[category],
                "n_records": len(pairs),
                "n_fields": len(pooled.field_outcomes[category]),
                baseline_run: _fmt_ci(b_mean, b_lo, b_hi),
                system_run: _fmt_ci(a_mean, a_lo, a_hi),
                "wilcoxon_p": _fmt_p(w_p),
                "wilcoxon_n": w_n,
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_p": _fmt_p(mc["pvalue"]),
                "perm_p": _fmt_p(perm["pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def build_precision_recall_table(
    data_root: str | Path,
    model: str,
    *,
    baseline_run: str = "baseline",
    system_run: str = "agent-tool",
) -> pd.DataFrame:
    """Precision, recall and F1 with cluster-bootstrap CIs, pooled across assays.

    One row per (field category, metric), with the baseline and ARMS estimates and
    the paired difference, each as ``point [lo, hi]``.  Records are the resampling
    unit throughout, and the three metrics share each replicate's resample, so the
    baseline, ARMS and difference columns of a row are mutually consistent.
    """
    import pandas as pd

    pooled = _pooled_data(data_root, model, baseline_run, system_run)

    rows = []
    for category in CATEGORIES:
        confusion = pooled.record_confusion[category]
        baseline = cluster_bootstrap_prf(confusion, "baseline")
        system = cluster_bootstrap_prf(confusion, "system")
        delta = cluster_bootstrap_prf_delta(confusion)
        for metric in ("precision", "recall", "f1"):
            rows.append(
                {
                    "category": CATEGORY_LABELS[category],
                    "metric": metric,
                    "n_records": len(confusion),
                    baseline_run: _fmt_ci(*baseline[metric]),
                    system_run: _fmt_ci(*system[metric]),
                    "difference": _fmt_ci(*delta[metric]),
                }
            )
    return pd.DataFrame(rows)


def build_single_run_table(data_root: str | Path, model: str, run_type: str) -> pd.DataFrame:
    """Accuracy and micro precision/recall/F1 with bootstrap CIs, for one run alone.

    One row per field category.  Needing no comparison run, this covers every run in a
    sweep -- a single repetition included -- where :func:`build_overall_table` can only
    describe the two arms it pairs.

    Records are the resampling unit throughout, and the point estimates are the ones
    :mod:`analysis.data_analysis` already reports, so each interval qualifies a number
    that appears in the tables above rather than a differently-weighted one.
    """
    import pandas as pd

    data = collect_single_run_data(data_root, model, run_type)

    rows = []
    for category in CATEGORIES:
        counts = data.record_counts[category]
        prf = bootstrap_prf(data.record_confusion[category])
        rows.append(
            {
                "run_type": run_type,
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
    baseline_run: str = "baseline",
    system_run: str = "agent-tool",
) -> pd.DataFrame:
    """Per-assay precision and recall: ARMS against baseline, with a corrected p-value.

    One row per (assay, field category, metric), holding both runs' intervals, the
    paired difference with its interval, the record-clustered permutation p-value, and
    that p-value corrected over every row of the table.

    *baseline_run* and *system_run* name the two conditions compared, so the same
    table answers "does ARMS beat baseline" or "does one repetition beat another"
    without changing anything else.  Every difference is system minus baseline,
    and the two estimate columns are named after the runs.

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
    """
    import pandas as pd

    rows = []
    for assay in iter_assays(data_root):
        data = collect_paired_data(data_root, model, assay.key, baseline_run=baseline_run, system_run=system_run)
        for category in CATEGORIES:
            confusion = data.record_confusion[category]
            if not confusion:
                continue
            baseline = cluster_bootstrap_prf(confusion, "baseline")
            system = cluster_bootstrap_prf(confusion, "system")
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
                        baseline_run: _fmt_ci(*baseline[metric]),
                        system_run: _fmt_ci(*system[metric]),
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
