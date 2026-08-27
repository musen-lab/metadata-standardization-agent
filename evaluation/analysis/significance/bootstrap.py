"""Bootstrap confidence intervals, resampling records rather than fields.

Fields within a record are correlated -- one bad template read spoils a whole record --
so every interval here except :func:`bootstrap_ci` (whose input is already one value
per record) draws whole records with replacement.  A field-level bootstrap would treat
correlated fields as independent evidence and report intervals that are too narrow.

Each function is seeded, so the same input gives the same interval on every run.
"""

from __future__ import annotations

import numpy as np

from analysis.metrics import precision_recall_f1

#: A ``(point_estimate, lower, upper)`` triple; ``nan``s when there is nothing to
#: resample.
Interval = tuple[float, float, float]

_NAN_INTERVAL: Interval = (float("nan"), float("nan"), float("nan"))


def _resample_indices(n_items: int, n_resamples: int, seed: int) -> np.ndarray:
    """Row *r* holds the item indices drawn with replacement for replicate *r*."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_resamples, n_items))


def _run_index(which: str) -> int:
    """0 for the baseline run, 1 for the system.

    Rejecting anything else rather than falling through to the system: a caller
    passing a run type by mistake would otherwise silently receive the other run's
    numbers, which is invisible in the output.
    """
    if which not in ("baseline", "system"):
        raise ValueError(f"which must be 'baseline' or 'system', got {which!r}")
    return 0 if which == "baseline" else 1


def _percentile_interval(replicates: np.ndarray, alpha: float) -> tuple[float, float]:
    """The central ``1 - alpha`` percentile interval of a replicate distribution."""
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Return ``(mean, lower, upper)`` for a bootstrap CI of the mean.

    Resamples *values* with replacement ``n_resamples`` times and takes the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled means.  Returns
    ``(nan, nan, nan)`` for an empty input.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return _NAN_INTERVAL
    idx = _resample_indices(arr.size, n_resamples, seed)
    lo, hi = _percentile_interval(arr[idx].mean(axis=1), alpha)
    return (float(arr.mean()), lo, hi)


def bootstrap_pooled_accuracy(
    counts: list[tuple[int, int]],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for one run's *pooled* (field-weighted) accuracy, resampling records.

    *counts* is a list of ``(correct, total)`` per record.  Pooled accuracy is
    ``sum(correct) / sum(total)`` -- the same field-weighted estimate the overall
    accuracy row reports, so this puts an interval on a number that already exists
    rather than a differently-weighted one.

    Any run can be measured this way, paired or not, which is what lets every
    condition get an interval and not only the two the paired tables compare.
    """
    if not counts:
        return _NAN_INTERVAL
    correct = np.array([c[0] for c in counts], dtype=float)
    totals = np.array([c[1] for c in counts], dtype=float)
    point = float(correct.sum() / totals.sum())

    idx = _resample_indices(len(counts), n_resamples, seed)
    lo, hi = _percentile_interval(correct[idx].sum(axis=1) / totals[idx].sum(axis=1), alpha)
    return (point, lo, hi)


def cluster_bootstrap_pooled(
    counts: list[tuple[int, int, int]],
    which: str,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for *pooled* (field-weighted) accuracy of one arm of a pair.

    *counts* is a list of ``(baseline_correct, system_correct, total)`` per record.
    ``which`` is ``"baseline"`` or ``"system"``.  Selects that run's column and hands
    it to :func:`bootstrap_pooled_accuracy`, so a paired arm and the same run
    measured on its own cannot come out differently.
    """
    col = _run_index(which)
    return bootstrap_pooled_accuracy(
        [(record[col], record[2]) for record in counts],
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )


def _prf_from_sums(
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    false_negatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised precision/recall/F1 from summed counts, ``0.0`` on a zero denominator.

    Matches :func:`~analysis.metrics.confusion.precision_recall_f1` elementwise, so a
    bootstrap replicate is scored exactly as the point estimate is.
    """
    asserted = true_positives + false_positives
    expected = true_positives + false_negatives
    zeros = np.zeros_like(true_positives, dtype=float)
    precision = np.divide(true_positives, asserted, out=zeros.copy(), where=asserted > 0)
    recall = np.divide(true_positives, expected, out=zeros.copy(), where=expected > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=zeros.copy(), where=denominator > 0)
    return precision, recall, f1


def _confusion_arrays(confusion: list[tuple[int, int, int, int, int, int]], which: str) -> tuple[np.ndarray, ...]:
    """Return per-record ``(tp, fp, fn)`` arrays for ``"baseline"`` or ``"system"``."""
    offset = 3 * _run_index(which)
    return tuple(np.array([record[offset + i] for record in confusion], dtype=float) for i in range(3))


def bootstrap_prf(
    confusion: list[tuple[int, int, int]],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Interval]:
    """Bootstrap CIs on one run's micro precision, recall and F1, resampling records.

    *confusion* is a list of ``(tp, fp, fn)`` per record.  Counts are summed over the
    resampled records and the ratios taken once from the totals, so the point estimate
    is the same micro-averaged number
    :func:`~analysis.data_analysis.create_overall_precision_recall_summary` reports.

    All three statistics come from the *same* resampled records within each replicate,
    so they stay mutually consistent: the F1 interval is the interval of ``2PR/(P+R)``,
    not something derived independently of the P and R intervals.

    Returns ``{"precision": (point, lo, hi), "recall": ..., "f1": ...}``, or ``nan``
    triples for an empty input.
    """
    if not confusion:
        return dict.fromkeys(("precision", "recall", "f1"), _NAN_INTERVAL)

    tp, fp, fn = (np.array([record[i] for record in confusion], dtype=float) for i in range(3))
    point = precision_recall_f1({"TP": int(tp.sum()), "FP": int(fp.sum()), "FN": int(fn.sum())})

    idx = _resample_indices(len(confusion), n_resamples, seed)
    precision, recall, f1 = _prf_from_sums(tp[idx].sum(axis=1), fp[idx].sum(axis=1), fn[idx].sum(axis=1))

    out: dict[str, Interval] = {}
    for name, replicates in (("precision", precision), ("recall", recall), ("f1", f1)):
        lo, hi = _percentile_interval(replicates, alpha)
        out[name] = (point[name], lo, hi)
    return out


def cluster_bootstrap_prf(
    confusion: list[tuple[int, int, int, int, int, int]],
    which: str,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Interval]:
    """Bootstrap CIs on precision, recall and F1 for one arm of a pair.

    *confusion* is a list of ``(baseline_tp, baseline_fp, baseline_fn, system_tp,
    system_fp, system_fn)`` per record; ``which`` selects ``"baseline"`` or
    ``"system"``.  Selects that run's three columns and hands them to
    :func:`bootstrap_prf`, so a run measured as half of a pair and the same run
    measured on its own cannot come out differently.
    """
    offset = 3 * _run_index(which)
    return bootstrap_prf(
        [(record[offset], record[offset + 1], record[offset + 2]) for record in confusion],
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )


def cluster_bootstrap_prf_delta(
    confusion: list[tuple[int, int, int, int, int, int]],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Interval]:
    """Paired bootstrap CIs on the system-minus-baseline difference in P, R and F1.

    Each replicate resamples records once and scores *both* runs on that same
    resample, so the pairing is preserved and the difference is not inflated by
    the two runs being resampled independently.  An interval excluding zero is
    the confidence statement about the difference.

    Returns ``{"precision": (delta, lo, hi), "recall": ..., "f1": ...}``.
    """
    if not confusion:
        return dict.fromkeys(("precision", "recall", "f1"), _NAN_INTERVAL)

    b_tp, b_fp, b_fn = _confusion_arrays(confusion, "baseline")
    s_tp, s_fp, s_fn = _confusion_arrays(confusion, "system")
    baseline_point = precision_recall_f1({"TP": int(b_tp.sum()), "FP": int(b_fp.sum()), "FN": int(b_fn.sum())})
    system_point = precision_recall_f1({"TP": int(s_tp.sum()), "FP": int(s_fp.sum()), "FN": int(s_fn.sum())})

    idx = _resample_indices(len(confusion), n_resamples, seed)
    baseline = _prf_from_sums(b_tp[idx].sum(axis=1), b_fp[idx].sum(axis=1), b_fn[idx].sum(axis=1))
    system = _prf_from_sums(s_tp[idx].sum(axis=1), s_fp[idx].sum(axis=1), s_fn[idx].sum(axis=1))

    out: dict[str, Interval] = {}
    for i, name in enumerate(("precision", "recall", "f1")):
        lo, hi = _percentile_interval(system[i] - baseline[i], alpha)
        out[name] = (system_point[name] - baseline_point[name], lo, hi)
    return out
