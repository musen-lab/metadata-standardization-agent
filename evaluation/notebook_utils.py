"""Assemble and print the tables `experiment.ipynb` shows.

The measurements themselves live in :mod:`analysis` -- this module only arranges what
they return and prints it, so the notebook can call rather than define.  Everything here
both prints and returns what it printed, so a cell can show a table and still keep the
frame for a follow-up question.

Nothing here decides anything a reader would want to argue with: no thresholds beyond
the *alpha* passed in, no metric definitions, no scoring.  Those belong in
:mod:`analysis.data_analysis` and :mod:`analysis.significance`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from analysis.corpus import iter_assays
from analysis.data_analysis import (
    collect_field_errors,
    create_per_assay_deduplicated_precision_recall_summary,
    create_per_assay_precision_recall_summary,
    deduplicate_errors,
    reconcile_with_confusion,
    summarize_error_categories,
    summarize_error_subcategories,
)
from analysis.metrics import CONFUSION_CATEGORIES
from analysis.significance import (
    CATEGORIES,
    CATEGORY_LABELS,
    PairedData,
    build_precision_recall_table,
    collect_paired_data,
    deduplicated_paired_tests,
    effective_sample_size,
    paired_permutation_prf,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: The instance-weighted table's columns, in reading order.
WEIGHTED_COLUMNS = ["assay", "field_type", "n_records", "TP", "FP", "FN", "precision", "recall", "f1"]

#: The deduplicated table's columns.  The four counts are the numerators and denominators
#: the two ratios are built from, which is what makes a surprising ratio checkable.
DEDUPLICATED_COLUMNS = [
    "assay",
    "field_type",
    "n_gold_values",  # distinct (field, gold value) pairs gold asks for -- recall's denominator
    "gold_values_reproduced",  # how many of those the run produced -- recall's numerator
    "n_asserted_values",  # distinct (field, predicted value) pairs the run filled in -- precision's denominator
    "asserted_values_correct",  # how many of those assertions were right -- precision's numerator
    "precision",
    "recall",
    "f1",
]

METRICS = ("precision", "recall")


def count_predictions(data_root: str | Path, model: str, run_types: Sequence[str]) -> dict[str, int]:
    """How many predictions each of *run_types* has on disk, across every assay."""
    return {
        run_type: sum(len(list(assay.output_dir(model, run_type).glob("*.json"))) for assay in iter_assays(data_root))
        for run_type in run_types
    }


def order_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Assay-major, field types in ontology / non_ontology / all order."""
    assays = list(dict.fromkeys(table["assay"]))
    table = table.assign(
        assay=pd.Categorical(table["assay"], assays, ordered=True),
        field_type=pd.Categorical(table["field_type"], CONFUSION_CATEGORIES, ordered=True),
    )
    return table.sort_values(["assay", "field_type"]).reset_index(drop=True)


def show_precision_recall_tables(
    data_root: str | Path,
    model: str,
    run_types: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Print the instance-weighted table for every run with predictions, and return them.

    One row per (assay, field type): one field of one record is one unit, which is the
    workload number -- the fraction of values a curator would have to fix by hand.
    """
    scored = count_predictions(data_root, model, run_types)
    tables: dict[str, pd.DataFrame] = {}

    print("=== instance-weighted: one field of one record is one unit ===")
    for run_type, count in scored.items():
        if not count:
            continue
        frames = [
            create_per_assay_precision_recall_summary(data_root, model, run_type, category=field_type).assign(
                field_type=field_type
            )
            for field_type in CONFUSION_CATEGORIES
        ]
        frames = [frame for frame in frames if len(frame)]
        tables[run_type] = order_rows(pd.concat(frames, ignore_index=True))
        print(f"\n--- {run_type} ({count} prediction(s)) ---")
        print(tables[run_type][WEIGHTED_COLUMNS].to_string(index=False))

    if not tables:
        print(f"  no predictions on disk for any of {list(run_types)}.")
    return tables


def show_deduplicated_tables(
    data_root: str | Path,
    model: str,
    run_types: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Print the deduplicated table for every run with predictions, and return them.

    Same rows, but one distinct value is one unit, which is the capability number: how
    many different things the run gets right rather than how much work it saves.
    """
    scored = count_predictions(data_root, model, run_types)
    tables: dict[str, pd.DataFrame] = {}

    print("=== deduplicated: one distinct value is one unit ===")
    for run_type, count in scored.items():
        if not count:
            continue
        rows = create_per_assay_deduplicated_precision_recall_summary(data_root, model, run_type)
        tables[run_type] = order_rows(rows.rename(columns={"category": "field_type"}))
        print(f"\n--- {run_type} ---")
        print(tables[run_type][DEDUPLICATED_COLUMNS].to_string(index=False))

    if not tables:
        print(f"  no predictions on disk for any of {list(run_types)}.")
    return tables


def show_hypothesis_tests(
    data_root: str | Path,
    model: str,
    *,
    baseline_run: str,
    system_run: str,
    alpha: float = 0.05,
) -> pd.DataFrame | None:
    """Print the record-level verdict on precision and recall, and return it.

    An interval from a record-cluster bootstrap says how big the difference is; a
    record-clustered permutation test says how surprising it would be under the null.
    The effective sample size printed afterwards is the caveat on both: it counts how
    much independent evidence the corpus holds once values repeated across records are
    clustered together.  Returns ``None`` when either run has no predictions.
    """
    scored = count_predictions(data_root, model, (baseline_run, system_run))
    if not (scored[baseline_run] and scored[system_run]):
        print(
            f"Nothing to test: found {scored[baseline_run]} prediction(s) for {baseline_run!r} "
            f"and {scored[system_run]} for {system_run!r}; both are needed for a paired test."
        )
        return None

    print(
        f"H0: {system_run} and {baseline_run} are interchangeable.  "
        f"Differences are {system_run} minus {baseline_run}.\n"
    )

    intervals = build_precision_recall_table(data_root, model, baseline_run=baseline_run, system_run=system_run)
    intervals = intervals[intervals["metric"].isin(METRICS)]
    print("=== 95% confidence intervals, record-cluster bootstrap ===")
    print(intervals.to_string(index=False))

    pooled = PairedData()
    for assay in iter_assays(data_root):
        pooled.extend(
            collect_paired_data(data_root, model, assay.key, baseline_run=baseline_run, system_run=system_run)
        )

    verdicts: list[dict[str, Any]] = []
    for category in CATEGORIES:
        test = paired_permutation_prf(pooled.record_confusion[category])
        for metric in METRICS:
            difference = intervals.query("category == @CATEGORY_LABELS[@category] and metric == @metric")
            verdicts.append(
                {
                    "field_type": CATEGORY_LABELS[category],
                    "metric": metric,
                    "difference [95% CI]": difference["difference"].iloc[0],
                    "p_value": round(test[metric]["pvalue"], 4),
                    "n_records": len(pooled.record_confusion[category]),
                    "n_differing": test["n_effective"],
                    f"reject H0 at {alpha}": "yes" if test[metric]["pvalue"] < alpha else "no",
                }
            )

    table = pd.DataFrame(verdicts)
    print("\n=== record-clustered permutation test, 10,000 resamples ===")
    print(table.to_string(index=False))

    print("\n=== in words ===")
    for verdict in verdicts:
        significant = verdict[f"reject H0 at {alpha}"] == "yes"
        print(
            f"  {verdict['field_type']:<22} {verdict['metric']:<10} "
            f"{'significant' if significant else 'not significant'} "
            f"(p={verdict['p_value']}, difference {verdict['difference [95% CI]']})"
        )

    show_effective_sample_size(data_root, model, system_run)
    return table


def show_effective_sample_size(data_root: str | Path, model: str, run_type: str) -> None:
    """Print how much independent evidence the corpus holds for *run_type*."""
    print(f"\n=== effective sample size ({run_type} field outcomes, clustered by (assay, field, value)) ===")
    print("  N_eff far below N means the corpus holds less independent evidence than the record count suggests.")
    for field_type in (None, "ontology", "non_ontology"):
        ess = effective_sample_size(data_root, model, run_type, field_type=field_type)
        print(
            f"  {field_type or 'all':<13} N={ess['n']:.0f} clusters={ess['n_clusters']:.0f} "
            f"ICC={ess['icc']:.3f} DEFF={ess['design_effect']:.1f} N_eff={ess['n_effective']:.0f}"
        )


def show_deduplicated_tests(
    data_root: str | Path,
    model: str,
    *,
    baseline_run: str,
    system_run: str,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
) -> pd.DataFrame | None:
    """Print the same question asked of distinct values, and return the verdict table.

    Recall pairs on the distinct gold value, precision on the field -- the two runs
    assert different values, so precision has no shared item list to pair on.  Returns
    ``None`` when either run has no predictions.
    """
    scored = count_predictions(data_root, model, (baseline_run, system_run))
    if not (scored[baseline_run] and scored[system_run]):
        print(f"Nothing to test: {baseline_run!r} and {system_run!r} must both have predictions.")
        return None

    rows = deduplicated_paired_tests(
        data_root, model, baseline_run=baseline_run, system_run=system_run, n_resamples=n_resamples
    )
    table = pd.DataFrame(rows)
    table["difference [95% CI]"] = table.apply(
        lambda row: f"{row['delta']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]", axis=1
    )
    table["p_value"] = table["pvalue"].round(4)
    table[f"reject H0 at {alpha}"] = ["yes" if pvalue < alpha else "no" for pvalue in table["pvalue"]]
    table = table.rename(columns={"baseline": baseline_run, "system": system_run}).round(
        {baseline_run: 3, system_run: 3}
    )

    print(f"=== deduplicated: paired over items, not records ({n_resamples:,} resamples) ===")
    columns = [
        "field_type",
        "metric",
        "paired on",
        "n_items",
        baseline_run,
        system_run,
        "difference [95% CI]",
        "p_value",
        f"reject H0 at {alpha}",
    ]
    print(table[columns].to_string(index=False))
    print("\nA row significant here is not explained by repetition.  A row significant in the")
    print("instance-weighted test but not here is a real saving of work, carried by values the")
    print("corpus repeats -- not evidence that the run knows more distinct answers.")
    return table


def show_error_analysis(
    data_root: str | Path,
    model: str,
    run_type: str,
    *,
    field_type: str | None = None,
    apply_dedup: bool = False,
    top_fields: int = 10,
) -> pd.DataFrame:
    """Print why *run_type* loses precision and why it loses recall, and return the errors.

    Two tables, counting by assay: the categories first, then the sub-categories they break
    into.  Both levels are printed because they answer different questions -- the category
    is what the figures draw and what a headline can carry, the sub-category is what
    :func:`show_error_examples` reads back -- and the sub-categories run in their
    categories' order, so printing them together is what makes the roll-up checkable down a
    column rather than something the reader has to reconstruct.

    One row per error, so each table counts every error once; the ``costs`` column says
    which side of the score a row lands on, and the reconciliation line at the top recovers
    ``FP`` and ``FN`` from it, since a taxonomy that quietly dropped errors would make every
    share below it wrong.

    *apply_dedup* counts each distinct error once instead of once per record it occurs in,
    which is the same question :func:`show_deduplicated_tables` asks of the headline
    numbers: how many different things the run gets wrong, rather than how much work they
    cost.  The reconciliation is still run against every instance -- it is a property of the
    collection and not of the reading -- so the line at the top holds either way, and the
    tables under it say which of the two they are counting.

    The returned frame is one row per error, carrying both levels, the gold, predicted and
    legacy values, the run's own stated resolution and reasoning where it kept them, and a
    ``pointer`` of ``<assay>/<record>#<field>`` to open.  Pass it to
    :func:`show_error_examples` to read a category or a sub-category.
    """
    errors = collect_field_errors(data_root, model, run_type)
    if errors.empty:
        print(f"No errors to categorise: {run_type!r} has no predictions under {model!r}.")
        return errors

    # Reconciled before any deduplication: the check is that the collection lost nothing,
    # which is a fact about every instance and not about the reading chosen below.
    counts = reconcile_with_confusion(errors, data_root, model, run_type)
    print(f"=== {run_type}: every counted error, categorised ===")
    for cell, totals in counts.items():
        agree = "accounted for" if totals["counted"] == totals["categorised"] else "MISMATCH"
        print(f"  {cell}: {totals['counted']} counted, {totals['categorised']} categorised -- {agree}")
    if field_type is not None:
        print(f"  restricted to {field_type} fields")

    if apply_dedup:
        instances = len(errors)
        errors = deduplicate_errors(errors)
        print(f"  deduplicated: {instances} instances of {len(errors)} distinct errors, counted once each")

    unit = "distinct error" if apply_dedup else "error"
    categories = summarize_error_categories(errors, field_type=field_type)
    print(f"\n--- by category: what the run did (one {unit} is one unit) ---")
    print(categories.to_string(index=False) if len(categories) else "  none")

    subcategories = summarize_error_subcategories(errors, field_type=field_type)
    print("\n    the same errors, by sub-category:")
    print(subcategories.to_string(index=False) if len(subcategories) else "  none")

    selected = errors if field_type is None else errors[errors["field_type"] == field_type]
    print(f"\n--- the {top_fields} fields carrying the most errors ---")
    worst = (
        selected.groupby(["assay", "field"], observed=True)
        .agg(
            errors=("subcategory", "size"),
            categories=("category", lambda values: ", ".join(sorted(set(values)))),
            subcategories=("subcategory", lambda values: ", ".join(sorted(set(values))[:3])),
        )
        .sort_values("errors", ascending=False)
        .head(top_fields)
    )
    print(worst.to_string())

    print("\nTo read any of these: show_error_examples(errors, subcategory=...) or")
    print("show_error_examples(errors, category=...), or open the pointer column --")
    print("<assay>/<record>#<field> -- against data/<assay>/gold/<record>.json.")
    return errors


def show_error_examples(
    errors: pd.DataFrame,
    *,
    category: str | None = None,
    subcategory: str | None = None,
    assay: str | None = None,
    n: int = 5,
) -> pd.DataFrame:
    """Print *n* concrete errors from one category or sub-category, with the evidence.

    Shows what gold wanted, what the run wrote, what the legacy record held for that
    field, and the run's own reasoning where it kept one -- the four things needed to
    decide whether the label is fair.  Returns the rows printed.

    Pass exactly one of *category* or *subcategory*: a category to see the range of things
    that fall under it, a sub-category to read one kind of error at a time.
    """
    if (category is None) == (subcategory is None):
        raise ValueError("pass exactly one of category= or subcategory=")

    level, wanted = ("category", category) if subcategory is None else ("subcategory", subcategory)
    selected = errors[errors[level] == wanted]
    if assay is not None:
        selected = selected[selected["assay"] == assay]
    if selected.empty:
        print(f"No errors with {level} {wanted!r}" + (f" for {assay}." if assay else "."))
        return selected

    print(f"=== {wanted}: {len(selected)} error(s), showing {min(n, len(selected))} ===")
    for _index, row in selected.head(n).iterrows():
        # The sub-category is printed even when it is what was asked for: with a category
        # selected, it is the one thing distinguishing the rows from each other.
        print(f"\n  {row['pointer']}   [{row['subcategory']}, {row['field_type']}, {row['case']}]")
        print(f"    gold      {row['gold_value']!r}")
        print(f"    predicted {row['predicted_value']!r}")
        print(f"    legacy    {row['legacy_value']!r}")
        if row["resolution"]:
            print(f"    the run called this {row['resolution']!r}: {row['reasoning']}")
    return selected.head(n)
