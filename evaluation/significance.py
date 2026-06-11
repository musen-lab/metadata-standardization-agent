"""Uncertainty and significance analysis for baseline (prompt-only) vs. ARMS.

This module answers Reviewer 2's request for "confidence intervals, statistical
significance testing, or another formal assessment of uncertainty for the
differences between the baseline and ARMS," reported across assay types and field
categories.  It reads the prediction files already written by the experiment (no
LLM/API calls) and computes:

* **Bootstrap 95% confidence intervals** on per-record accuracy, for the baseline
  and ARMS separately.
* **Paired Wilcoxon signed-rank test** on per-record accuracy (same record under
  both methods).
* **Paired McNemar test** on per-field correctness (same field of the same record
  under both methods), reporting ``b`` (baseline-only correct), ``c`` (ARMS-only
  correct), and the p-value.
* **Record-clustered permutation test** on the same discordant outcomes, but with
  the record as the independent unit (whole-record label swaps).  Unlike the flat
  McNemar test, this does not treat duplicated or within-record-correlated fields
  as independent, so it does not overstate significance.

All four are produced for each of the three field categories used in the paper
(``ontology``, ``non_ontology``, ``all``) and both per assay and pooled overall.

Run from the ``evaluation/`` directory (same convention as ``data_analysis`` and
``plots``)::

    uv run python significance.py --data-root ../data --model gpt5mini
    uv run python significance.py --data-root ../data --model gpt5mini --csv-dir out/

Or, from the project root::

    uv run python evaluation/significance.py --data-root data --model gpt5mini
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from assays import ASSAY_ORDER
from metrics import compute_field_results

if TYPE_CHECKING:
    import pandas as pd

CATEGORIES = ("ontology", "non_ontology", "all")
CATEGORY_LABELS = {
    "ontology": "Ontology-constrained",
    "non_ontology": "Non-ontology-constrained",
    "all": "All fields",
}


@dataclass
class PairedData:
    """Paired baseline/ARMS outcomes for one assay (or pooled across assays).

    * ``record_acc[category]`` is a list of ``(baseline_accuracy, arms_accuracy)``
      tuples, one per record that has at least one field in that category.  Used
      for the per-record (record-weighted) view and the Wilcoxon test, matching
      how the paper's per-assay rows are computed.
    * ``record_counts[category]`` is a list of ``(baseline_correct, arms_correct,
      total)`` integer tuples, one per record.  Used for the pooled
      (field-weighted) accuracy and its cluster bootstrap, matching how the
      paper's overall bottom row is computed.
    * ``field_outcomes[category]`` is a list of ``(baseline_correct, arms_correct)``
      boolean tuples, one per field of that category across all records.  Used
      for the field-level McNemar test.
    * ``record_discordant[category]`` is a list of ``(baseline_only_correct,
      arms_only_correct)`` integer tuples, one per record: the per-record counts
      of McNemar-discordant fields.  Used for the record-clustered permutation
      test, which keeps the record (not the field) as the independent unit.
    """

    record_acc: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    record_counts: dict[str, list[tuple[int, int, int]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    field_outcomes: dict[str, list[tuple[bool, bool]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    record_discordant: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})

    def extend(self, other: PairedData) -> None:
        """Accumulate another assay's data into this one (used for the pooled view)."""
        for c in CATEGORIES:
            self.record_acc[c].extend(other.record_acc[c])
            self.record_counts[c].extend(other.record_counts[c])
            self.field_outcomes[c].extend(other.field_outcomes[c])
            self.record_discordant[c].extend(other.record_discordant[c])


def collect_paired_data(data_root: str | Path, model: str, assay_key: str) -> PairedData:
    """Collect paired baseline/ARMS outcomes for a single assay from saved outputs."""
    root = Path(data_root)
    schema_path = root / "schemas" / f"{assay_key}.json"
    gold_dir = root / assay_key / "gold"
    base_dir = root / assay_key / "output" / model / "baseline"
    arms_dir = root / assay_key / "output" / model / "experiment"

    data = PairedData()
    if not (schema_path.exists() and gold_dir.exists()):
        return data

    for gold_file in sorted(gold_dir.glob("*.json")):
        base_file = base_dir / gold_file.name
        arms_file = arms_dir / gold_file.name
        if not (base_file.exists() and arms_file.exists()):
            continue

        gold = json.loads(gold_file.read_text())
        base_pred = json.loads(base_file.read_text())
        arms_pred = json.loads(arms_file.read_text())

        base_results = compute_field_results(base_pred, gold, schema_path)
        arms_correct = {f: ok for f, _t, ok in compute_field_results(arms_pred, gold, schema_path)}

        # Per-record correct/total counts by category, so a record with no fields
        # in a category is excluded from that category rather than scored 0.
        per_record: dict[str, list[list[int]]] = {c: [[0, 0], [0, 0]] for c in CATEGORIES}
        # Per-record McNemar-discordant counts: [baseline_only_correct, arms_only_correct].
        per_record_disc: dict[str, list[int]] = {c: [0, 0] for c in CATEGORIES}
        for fname, ftype, base_ok in base_results:
            arms_ok = arms_correct.get(fname, False)
            data.field_outcomes[ftype].append((base_ok, arms_ok))
            data.field_outcomes["all"].append((base_ok, arms_ok))
            for cat in (ftype, "all"):
                per_record[cat][0][0] += int(base_ok)
                per_record[cat][0][1] += 1
                per_record[cat][1][0] += int(arms_ok)
                per_record[cat][1][1] += 1
                if base_ok and not arms_ok:
                    per_record_disc[cat][0] += 1
                elif arms_ok and not base_ok:
                    per_record_disc[cat][1] += 1
        for cat in CATEGORIES:
            (b_corr, b_tot), (a_corr, a_tot) = per_record[cat]
            if b_tot > 0:  # category present in this record
                data.record_acc[cat].append((b_corr / b_tot, a_corr / a_tot))
                data.record_counts[cat].append((b_corr, a_corr, b_tot))
                b_only, a_only = per_record_disc[cat]
                data.record_discordant[cat].append((b_only, a_only))
    return data


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` for a bootstrap CI of the mean.

    Resamples *values* with replacement ``n_resamples`` times and takes the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled means.  Returns
    ``(nan, nan, nan)`` for an empty input.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(arr.mean()), float(lo), float(hi))


def cluster_bootstrap_pooled(
    counts: list[tuple[int, int, int]],
    which: str,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for *pooled* (field-weighted) accuracy, resampling whole records.

    *counts* is a list of ``(baseline_correct, arms_correct, total)`` per record.
    ``which`` is ``"baseline"`` or ``"arms"``.  Resampling at the record level
    (a cluster bootstrap) respects that fields within a record are correlated, so
    the CI is not artificially narrow.  Pooled accuracy is
    ``sum(correct) / sum(total)`` -- the same field-weighted estimate the paper's
    overall row reports.
    """
    if not counts:
        return (float("nan"), float("nan"), float("nan"))
    col = 0 if which == "baseline" else 1
    correct = np.array([c[col] for c in counts], dtype=float)
    totals = np.array([c[2] for c in counts], dtype=float)
    point = float(correct.sum() / totals.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_resamples, len(counts)))
    resampled = correct[idx].sum(axis=1) / totals[idx].sum(axis=1)
    lo, hi = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


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


def effective_sample_size(
    data_root: str | Path,
    model: str,
    run_type: str,
    *,
    field_type: str | None = None,
) -> dict[str, float]:
    """Effective number of independent field observations, given the repetition.

    The corpus repeats the same ``(assay, field, gold-value)`` correction across
    many records, so the field outcomes are not independent.  This groups every
    field instance by its ``(assay, field, gold-value)`` key, estimates the
    intra-cluster correlation (ICC) of the binary correctness outcome via a
    one-way random-effects ANOVA, and reports the design effect
    ``DEFF = 1 + (m_bar - 1) * ICC`` and the effective sample size
    ``n_effective = n / DEFF``.  ``field_type`` filters to ``"ontology"`` or
    ``"non_ontology"`` (default: all fields).  Returns a dict with ``n``,
    ``n_clusters``, ``icc``, ``design_effect``, and ``n_effective``.
    """
    from collections import defaultdict

    from metrics import _get_ontology_constrained_fields, _is_missing, _values_match

    root = Path(data_root)
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])  # key -> [n, n_correct]

    for assay_key, _assay_label in ASSAY_ORDER:
        schema_path = root / "schemas" / f"{assay_key}.json"
        gold_dir = root / assay_key / "gold"
        output_dir = root / assay_key / "output" / model / run_type
        if not (schema_path.exists() and gold_dir.exists()):
            continue
        ontology_fields = set(_get_ontology_constrained_fields(schema_path))
        for gold_file in sorted(gold_dir.glob("*.json")):
            pred_file = output_dir / gold_file.name
            if not pred_file.exists():
                continue
            gold = json.loads(gold_file.read_text())
            predicted = json.loads(pred_file.read_text())
            for field_name, gold_val in gold.items():
                is_ont = field_name in ontology_fields
                if field_type == "ontology" and not is_ont:
                    continue
                if field_type == "non_ontology" and is_ont:
                    continue
                gold_missing = _is_missing(gold_val)
                pred_val = predicted.get(field_name)
                pred_missing = _is_missing(pred_val)
                correct = (gold_missing and pred_missing) or (
                    not gold_missing
                    and not pred_missing
                    and _values_match(pred_val, gold_val, match_case=True, match_whole_word=True, field_name=field_name)
                )
                cell = groups[(assay_key, field_name, json.dumps(gold_val, sort_keys=True))]
                cell[0] += 1
                cell[1] += int(correct)

    n = sum(c[0] for c in groups.values())
    k = len(groups)
    if k < 2 or n <= k:
        return {"n": n, "n_clusters": k, "icc": float("nan"), "design_effect": float("nan"), "n_effective": float(n)}

    grand = sum(c[1] for c in groups.values()) / n
    ss_between = sum(c[0] * ((c[1] / c[0]) - grand) ** 2 for c in groups.values())
    ss_within = sum(c[1] * (c[0] - c[1]) / c[0] for c in groups.values())  # binary within-SS = m*p*(1-p)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    m0 = (n - sum(c[0] ** 2 for c in groups.values()) / n) / (k - 1)
    denom = ms_between + (m0 - 1) * ms_within
    icc = max(0.0, (ms_between - ms_within) / denom) if denom > 0 else 0.0
    m_bar = n / k
    design_effect = 1 + (m_bar - 1) * icc
    return {
        "n": n,
        "n_clusters": k,
        "icc": icc,
        "design_effect": design_effect,
        "n_effective": n / design_effect,
    }


def paired_mcnemar(outcomes: list[tuple[bool, bool]]) -> dict[str, float]:
    """Paired McNemar test on per-field correctness.

    *outcomes* is a list of ``(baseline_correct, arms_correct)`` booleans.  Returns
    a dict with ``b`` (baseline-only correct), ``c`` (ARMS-only correct),
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

    *discordant* is a list of ``(baseline_only_correct, arms_only_correct)`` integer
    tuples, one per record.  Each record collapses to a signed net ARMS advantage
    ``d_i = arms_only - baseline_only`` and the observed statistic is ``S = sum(d_i)``.

    Under the null that baseline and ARMS are interchangeable, swapping a record's
    two labels flips the sign of its ``d_i``, so the null distribution is generated
    by assigning each record an independent random ``+/-`` sign.  Because the sign
    flip acts on the *whole record*, fields within a record -- and the same field
    repeated across records -- are never treated as independent observations: the
    record is the unit of analysis, matching :func:`cluster_bootstrap_pooled` and
    the per-record :func:`paired_wilcoxon`.  This avoids the inflated significance
    that a flat field-level McNemar test produces on clustered/duplicated fields.

    Returns a dict with ``s_observed`` (net ARMS advantage among discordant fields),
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


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    if mean != mean:  # nan
        return "-"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_p(p: float) -> str:
    if p != p:  # nan
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_per_assay_table(data_root: str | Path, model: str, category: str = "all") -> pd.DataFrame:
    """One row per assay for *category*: baseline/ARMS mean+CI, Wilcoxon p, McNemar b/c/p."""
    import pandas as pd

    rows = []
    for assay_key, assay_label in ASSAY_ORDER:
        data = collect_paired_data(data_root, model, assay_key)
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
                "assay": assay_label,
                "n_records": len(pairs),
                "baseline": _fmt_ci(b_mean, b_lo, b_hi),
                "arms": _fmt_ci(a_mean, a_lo, a_hi),
                "wilcoxon_p": _fmt_p(w_p),
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_p": _fmt_p(mc["pvalue"]),
                "perm_p": _fmt_p(perm["pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def build_overall_table(data_root: str | Path, model: str) -> pd.DataFrame:
    """Pooled-across-assays results, one row per field category.

    Accuracy is reported as *pooled* (field-weighted) accuracy with a record-level
    cluster-bootstrap CI, matching the paper's overall bottom row.  Significance is
    the paired Wilcoxon (per-record) and McNemar (per-field) tests, plus a
    record-clustered permutation test that treats the record -- not the field -- as
    the independent unit, so duplicated/clustered fields do not inflate the result.
    """
    import pandas as pd

    pooled = PairedData()
    for assay_key, _ in ASSAY_ORDER:
        pooled.extend(collect_paired_data(data_root, model, assay_key))

    rows = []
    for category in CATEGORIES:
        counts = pooled.record_counts[category]
        pairs = pooled.record_acc[category]
        b_mean, b_lo, b_hi = cluster_bootstrap_pooled(counts, "baseline")
        a_mean, a_lo, a_hi = cluster_bootstrap_pooled(counts, "arms")
        _, w_p, w_n = paired_wilcoxon(pairs)
        mc = paired_mcnemar(pooled.field_outcomes[category])
        perm = paired_permutation(pooled.record_discordant[category])
        rows.append(
            {
                "category": CATEGORY_LABELS[category],
                "n_records": len(pairs),
                "n_fields": len(pooled.field_outcomes[category]),
                "baseline": _fmt_ci(b_mean, b_lo, b_hi),
                "arms": _fmt_ci(a_mean, a_lo, a_hi),
                "wilcoxon_p": _fmt_p(w_p),
                "wilcoxon_n": w_n,
                "mcnemar_b": mc["b"],
                "mcnemar_c": mc["c"],
                "mcnemar_p": _fmt_p(mc["pvalue"]),
                "perm_p": _fmt_p(perm["pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence intervals and paired significance tests.")
    parser.add_argument("--data-root", default="../data", help="Path to the data/ directory.")
    parser.add_argument("--model", default="gpt5mini", help="Model output sub-directory (e.g. gpt5mini).")
    parser.add_argument("--csv-dir", default=None, help="Optional directory to write CSV tables.")
    args = parser.parse_args()

    overall = build_overall_table(args.data_root, args.model)
    print("\n=== Overall (pooled across assays) ===")
    print(overall.to_string(index=False))

    print("\n=== Effective sample size (ARMS field outcomes, clustered by (assay, field, value)) ===")
    for cat, label in [(None, "all"), ("ontology", "ontology"), ("non_ontology", "non_ontology")]:
        e = effective_sample_size(args.data_root, args.model, "experiment", field_type=cat)
        print(
            f"  {label:13s} N={e['n']:.0f} clusters={e['n_clusters']:.0f} "
            f"ICC={e['icc']:.3f} DEFF={e['design_effect']:.1f} N_eff={e['n_effective']:.0f}"
        )

    per_assay = {cat: build_per_assay_table(args.data_root, args.model, cat) for cat in CATEGORIES}
    for cat in CATEGORIES:
        print(f"\n=== Per assay: {CATEGORY_LABELS[cat]} ===")
        print(per_assay[cat].to_string(index=False))

    if args.csv_dir:
        out = Path(args.csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        overall.to_csv(out / f"significance_overall_{args.model}.csv", index=False)
        for cat in CATEGORIES:
            per_assay[cat].to_csv(out / f"significance_per_assay_{cat}_{args.model}.csv", index=False)
        print(f"\nWrote CSV tables to {out}/")


if __name__ == "__main__":
    main()
