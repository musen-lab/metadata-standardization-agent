"""Paired tests of whether the agent tools beat the prompt, at three units of analysis.

The same comparison run three ways, because the answer depends on what is treated as
one independent observation:

* :func:`paired_wilcoxon` -- the record, scored by its accuracy.
* :func:`paired_mcnemar` -- the field, scored right or wrong.  The most powerful of the
  three and the least defensible on this corpus, since fields within a record are
  correlated and the same correction recurs across records.
* :func:`paired_permutation` -- the record again, but on the same discordant fields
  McNemar uses, so the two are directly comparable and the gap between their p-values
  is the price of the independence assumption.
* :func:`paired_permutation_prf` -- the record once more, on micro-averaged precision
  and recall rather than accuracy.

:func:`adjust_pvalues` corrects a family of these for multiple testing, which any
per-assay claim needs before it can be reported.

Every test takes already-collected outcomes rather than a data root, so none of them
can disagree about which records were compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def paired_wilcoxon(pairs: list[tuple[float, float]]) -> tuple[float, float, int]:
    """Paired Wilcoxon signed-rank test on per-record accuracy.

    *pairs* is a list of ``(baseline_accuracy, arms_accuracy)``.  Returns
    ``(statistic, p_value, n_nonzero_diffs)``.  When every difference is zero (or
    there are no pairs), there is nothing to test and ``p_value`` is ``1.0``.
    """
    if not pairs:
        return (float("nan"), 1.0, 0)
    base = np.array([b for b, _ in pairs], dtype=float)
    arms = np.array([a for _, a in pairs], dtype=float)
    diffs = arms - base
    n_nonzero = int(np.count_nonzero(diffs))
    if n_nonzero == 0:
        return (float("nan"), 1.0, 0)
    try:
        result = wilcoxon(arms, base, zero_method="wilcox")
    except ValueError:
        return (float("nan"), 1.0, n_nonzero)
    return (float(result.statistic), float(result.pvalue), n_nonzero)


def paired_mcnemar(outcomes: list[tuple[bool, bool]]) -> dict[str, float]:
    """Paired McNemar test on per-field correctness.

    *outcomes* is a list of ``(baseline_correct, arms_correct)`` booleans.  Returns
    a dict with ``b`` (only baseline correct), ``c`` (only ARMS correct),
    ``n_discordant``, ``statistic``, and ``pvalue``.  Uses the exact binomial
    variant when discordant pairs are few (<25), else the chi-square approximation
    with continuity correction.
    """
    b = sum(1 for base_ok, arms_ok in outcomes if base_ok and not arms_ok)
    c = sum(1 for base_ok, arms_ok in outcomes if arms_ok and not base_ok)
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": float("nan"), "pvalue": 1.0}
    exact = n_disc < 25
    result = mcnemar([[0, b], [c, 0]], exact=exact, correction=not exact)
    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def paired_permutation(
    discordant: list[tuple[int, int]],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Record-clustered permutation test on McNemar-style discordant counts.

    *discordant* is a list of ``(baseline_only_correct, system_only_correct)``
    integer tuples, one per record.  Each record collapses to a signed net system
    advantage ``d_i = system_only - baseline_only`` and the observed statistic is ``S = sum(d_i)``.

    Under the null that the two runs are interchangeable, swapping a record's
    two labels flips the sign of its ``d_i``, so the null distribution is generated
    by assigning each record an independent random ``+/-`` sign.  Because the sign
    flip acts on the *whole record*, fields within a record -- and the same field
    repeated across records -- are never treated as independent observations: the
    record is the unit of analysis, matching
    :func:`~analysis.significance.bootstrap.cluster_bootstrap_pooled` and the
    per-record :func:`paired_wilcoxon`.  This avoids the inflated significance that a
    flat field-level McNemar test produces on clustered/duplicated fields.

    Returns a dict with ``s_observed`` (net system advantage among discordant fields),
    ``n_effective`` (records with a non-zero ``d_i`` -- the ones that carry signal),
    and a two-sided ``pvalue`` (Monte Carlo, using the ``(count + 1) / (n + 1)``
    convention so it is never exactly zero).  When no record has a non-zero ``d_i``
    there is nothing to test and ``pvalue`` is ``1.0``.
    """
    d = np.array([c - b for b, c in discordant], dtype=float)
    s_observed = float(d.sum())
    nonzero = d[d != 0.0]
    n_effective = int(nonzero.size)
    if n_effective == 0:
        return {"s_observed": s_observed, "n_effective": 0, "pvalue": 1.0}

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n_effective))
    resampled = signs @ nonzero
    # Float tolerance: the statistic is a sum of small integers represented as floats.
    count = int((np.abs(resampled) >= abs(s_observed) - 1e-9).sum())
    pvalue = (count + 1) / (n_resamples + 1)
    return {"s_observed": s_observed, "n_effective": n_effective, "pvalue": float(pvalue)}


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

    :func:`paired_permutation` tests accuracy, by way of discordant field counts.  This
    tests the ratios instead, which no other test here covers: precision and recall are
    micro-averaged over summed counts, so they cannot be reduced to a per-field win or
    loss the way accuracy can.

    Under the null the two runs are interchangeable, so a record's baseline and
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
        """Per-replicate totals for the two runs after the swap."""
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
