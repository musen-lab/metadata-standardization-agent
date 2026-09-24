"""Paired tests of whether one condition beats another on precision and recall.

:func:`paired_permutation_prf` takes the record as the independent unit: it swaps whole
records between the two conditions, so fields within a record -- and the same value
repeated across records -- are never counted as independent evidence.

:func:`adjust_pvalues` corrects a family of these for multiple testing, which any
per-assay claim needs before it can be reported.

Every test takes already-collected outcomes rather than a data root, so none of them
can disagree about which records were compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_permutation_prf(
    confusion: list[tuple[int, int, int, int, int, int]],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Record-clustered permutation test on the system-minus-baseline difference in P/R/F1.

    *confusion* is a list of ``(baseline_tp, baseline_fp, baseline_fn, system_tp,
    system_fp, system_fn)``
    per record -- the same input :func:`~analysis.significance.bootstrap.cluster_bootstrap_prf`
    takes, so the test and the interval describe one set of records.

    Precision and recall are micro-averaged over summed counts, so they cannot be reduced
    to a per-field win or loss: each replicate recomputes both from the swapped sums.

    Under the null the two conditions are interchangeable, so a record's baseline and
    system triples can be swapped without changing anything.  Each replicate swaps a random
    subset of records, recomputes the micro-averaged difference from the new sums, and
    the two-sided p-value is how often a shuffled difference is at least as large as
    the observed one.  The swap acts on the *whole record*, so fields within a record --
    and the same value repeated across records -- are never treated as independent.

    Returns ``{metric: {"delta", "pvalue"}}`` for precision, recall and F1, plus
    ``n_effective``: the records whose two triples differ at all, which are the only
    ones a swap can change.  With none of them there is nothing to test and every
    p-value is ``1.0``.
    """
    from analysis.metrics import precision_recall_f1
    from analysis.significance.bootstrap import _prf_from_sums

    metrics = ("precision", "recall", "f1")
    columns = [np.array([record[i] for record in confusion], dtype=float) for i in range(6)]
    base_tp, base_fp, base_fn, sys_tp, sys_fp, sys_fn = columns

    baseline_point = precision_recall_f1({"TP": int(base_tp.sum()), "FP": int(base_fp.sum()), "FN": int(base_fn.sum())})
    system_point = precision_recall_f1({"TP": int(sys_tp.sum()), "FP": int(sys_fp.sum()), "FN": int(sys_fn.sum())})
    observed = {metric: system_point[metric] - baseline_point[metric] for metric in metrics}

    differs = (base_tp != sys_tp) | (base_fp != sys_fp) | (base_fn != sys_fn)
    n_effective = int(differs.sum())
    if n_effective == 0:
        return {"n_effective": 0, **{m: {"delta": observed[m], "pvalue": 1.0} for m in metrics}}

    rng = np.random.default_rng(seed)
    swap = rng.random((n_resamples, len(confusion))) < 0.5

    def summed(baseline_side: np.ndarray, system_side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-replicate totals for the two conditions after the swap."""
        return (
            np.where(swap, system_side, baseline_side).sum(axis=1),
            np.where(swap, baseline_side, system_side).sum(axis=1),
        )

    b_tp, s_tp = summed(base_tp, sys_tp)
    b_fp, s_fp = summed(base_fp, sys_fp)
    b_fn, s_fn = summed(base_fn, sys_fn)
    shuffled_baseline = dict(zip(metrics, _prf_from_sums(b_tp, b_fp, b_fn), strict=True))
    shuffled_system = dict(zip(metrics, _prf_from_sums(s_tp, s_fp, s_fn), strict=True))

    out: dict[str, Any] = {"n_effective": n_effective}
    for metric in metrics:
        delta = shuffled_system[metric] - shuffled_baseline[metric]
        # Float tolerance: the ratios are recomputed from sums, so an exact tie can
        # miss by an ulp and would otherwise be counted as more extreme.
        count = int((np.abs(delta) >= abs(observed[metric]) - 1e-12).sum())
        out[metric] = {"delta": observed[metric], "pvalue": float((count + 1) / (n_resamples + 1))}
    return out


def adjust_pvalues(pvalues: list[float], method: str = "holm") -> list[float]:
    """Correct a family of p-values for multiple testing, preserving input order.

    Testing one hypothesis per assay per category per metric means dozens of tests, and
    at a 5% threshold roughly one in twenty comes out significant with nothing behind
    it.  A claim about any individual cell has to survive a correction over the whole
    family it was picked from.

    ``"holm"`` (Holm-Bonferroni) bounds the chance of *any* false positive in the
    family: use it to claim a specific assay beats baseline.  ``"fdr_bh"``
    (Benjamini-Hochberg) instead bounds the expected share of false positives among
    those called significant: more power, and the right choice for a claim about the
    corpus as a whole rather than about one cell.
    """
    if method not in ("holm", "fdr_bh"):
        raise ValueError(f"method must be 'holm' or 'fdr_bh', got {method!r}")
    if not pvalues:
        return []

    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    ordered = [pvalues[i] for i in order]
    adjusted = [0.0] * n

    if method == "holm":
        running = 0.0
        for rank, p in enumerate(ordered):
            running = max(running, (n - rank) * p)  # monotone, so a later p is never smaller
            adjusted[order[rank]] = min(1.0, running)
    else:
        running = 1.0
        for rank in range(n - 1, -1, -1):
            running = min(running, n * ordered[rank] / (rank + 1))
            adjusted[order[rank]] = min(1.0, running)
    return adjusted
