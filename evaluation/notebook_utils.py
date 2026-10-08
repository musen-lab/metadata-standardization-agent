"""Assemble and print the tables `experiment.ipynb` shows.

The measurements themselves live in :mod:`analysis` -- this module only arranges what
they return and prints it, so the notebook can call rather than define. Everything here
both prints and returns what it printed, so a cell can show a table and still keep the
frame for a follow-up question.

Nothing here decides anything a reader would want to argue with: no thresholds beyond
the *alpha* passed in, no metric definitions, no scoring.  Those belong in
:mod:`analysis.data_analysis` and :mod:`analysis.significance`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd

from analysis.corpus import iter_assays
from analysis.data_analysis import (
    collect_field_errors,
    create_per_assay_deduplicated_precision_recall_summary,
    create_per_assay_precision_recall_summary,
    create_run_spread_summary,
    deduplicate_errors,
    reconcile_with_confusion,
    summarize_error_categories,
    summarize_error_subcategories,
    summarize_error_subcategories_by_field_type,
)
from analysis.metrics import CONFUSION_CATEGORIES
from analysis.reference_constraints import audit_reference_constraints
from analysis.significance import (
    CATEGORIES,
    CATEGORY_LABELS,
    PairedData,
    build_pre_post_precision_recall_table,
    build_precision_recall_table,
    collect_paired_data,
    deduplicated_paired_tests,
    effective_sample_size,
    paired_permutation_prf,
)
from plots.marks import condition_label

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from analysis.reference_constraints import ReferenceConstraintAudit

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


def show_reference_exclusions(
    data_root: str | Path,
    *,
    report_dir: str | Path | None = None,
) -> ReferenceConstraintAudit:
    """Print assay-wide exclusions and optionally save complete review evidence.

    The HTML report groups fields by assay, with expandable constraint/value/record
    evidence. CSVs retain the field summary, all witnesses, and audit coverage.
    No reference records are modified and no exclusion is labelled an ARMS error.
    """
    from html import escape
    from pathlib import Path

    audit = audit_reference_constraints(data_root)
    print("=== reference constraint audit: fields labelled excluded within each assay ===")
    print(audit.coverage.to_string(index=False))
    if audit.fields.empty:
        print("No populated reference values violate the checked constraints.")
    else:
        print("\nEach listed field is excluded from every record in its assay, for both methods.")
        print(audit.fields.to_string(index=False))

    if report_dir is not None:
        destination = Path(report_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in (("fields", audit.fields), ("violations", audit.violations), ("coverage", audit.coverage)):
            table.to_csv(destination / f"reference_exclusions_{name}.csv", index=False)

        parts = [
            '<!doctype html><html lang="en"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Reference constraint exclusions</title>",
            "<style>body{font:16px system-ui;margin:2rem;line-height:1.5}"
            "table{border-collapse:collapse;font-size:14px;width:100%}"
            "td,th{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}"
            "td{overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}"
            "details{margin:1rem 0}nav a{margin-right:1rem}</style><body>",
            "<h1>Reference constraint exclusions</h1>",
            "<p>A populated reference value violating an explicit template constraint excludes its entire field "
            "within that assay, including valid and blank instances, for both methods and all runs. "
            "These are exclusions, not ARMS errors. Checks use saved templates and all reference records only.</p>",
            "<p>Checks: JSON data type, multiplicity, regex patterns, and closed permissible-value lists. "
            "Matching is case sensitive with no trimming or semantic reassessment. Blank or missing "
            "values do not trigger exclusion, even in required fields. Open ontology lists, descriptions, "
            "and fields absent from a template cannot establish a violation. Link and temporal types "
            "are checked as strings; format restrictions require an explicit pattern.</p>",
            "<p>Invalid records counts only violating records. Excluded comparisons counts all reference "
            "instances of that field; the scoring tables use records predicted by both methods.</p>",
            "<nav>"
            + " ".join(f'<a href="#{escape(assay)}">{escape(assay)}</a>' for assay in audit.coverage["assay"])
            + "</nav>",
            audit.coverage.rename(columns={"n_gold_records": "n_reference_records"}).to_html(index=False, escape=True),
        ]
        for assay in audit.coverage["assay"]:
            parts.append(f'<h2 id="{escape(assay)}">{escape(assay)}</h2>')
            fields = audit.fields[audit.fields["assay"] == assay]
            if fields.empty:
                parts.append("<p>No excluded fields.</p>")
                continue
            parts.append(
                fields.drop(columns=["assay"])
                .rename(columns={"n_gold_records": "n_reference_records"})
                .to_html(index=False, escape=True)
            )
            for row in fields.itertuples(index=False):
                evidence = audit.violations.query("assay == @assay and field == @row.field")
                parts.append(
                    f"<details><summary>{escape(row.field)}: {row.n_invalid_records} invalid of "
                    f"{row.n_gold_records} reference records — "
                    f"{row.n_excluded_comparisons} excluded comparisons</summary>"
                )
                for constraint, group in evidence.groupby("constraint", sort=False):
                    parts.append(
                        f"<h3>{escape(constraint)}</h3><p>Expected: "
                        f"<code>{escape(group['expected'].iloc[0])}</code></p>"
                    )
                    parts.append(
                        group[["gold_value", "n_records", "records"]]
                        .rename(columns={"gold_value": "reference_value"})
                        .to_html(index=False, escape=True)
                    )
                parts.append("</details>")
        parts.append("</body></html>")
        (destination / "reference_exclusions.html").write_text("\n".join(parts), encoding="utf-8")
        print(f"\nReview report: {destination / 'reference_exclusions.html'}")
    return audit


def show_post_adjudication_evaluation(
    data_root: str | Path,
    model: str,
    *,
    baseline: str,
    system: str,
    run: int = 1,
    n_resamples: int = 10_000,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Report a separate evaluation after objective assay-level reference exclusions.

    Reuses the original record-cluster intervals, applying the same gold-derived
    exclusions to both conditions. Prints plain per-assay and pooled precision
    and recall tables, then reuses ``show_deduplicated_tests`` with the same
    exclusions. The returned audit and tables allow further inspection.
    """
    from pathlib import Path

    from analysis.significance import build_per_assay_precision_recall_table

    audit = show_reference_exclusions(data_root, report_dir=report_dir)
    exclusions = audit.excluded_fields_by_assay
    per_assay = build_per_assay_precision_recall_table(
        data_root,
        model,
        baseline=baseline,
        system=system,
        run=run,
        excluded_fields_by_assay=exclusions,
    )
    pooled = build_precision_recall_table(
        data_root,
        model,
        baseline=baseline,
        system=system,
        run=run,
        excluded_fields_by_assay=exclusions,
    )
    columns = ["assay", "category", "metric", "n_records", baseline, system, "difference"]
    pooled = pooled[pooled["metric"].isin(METRICS)].assign(assay="All assays")
    frames = [table[columns] for table in (per_assay, pooled) if not table.empty]
    field_level = pd.concat(frames, ignore_index=True)
    print(f"\n=== post-adjudication per-assay precision and recall (run {run}): point [95% CI] ===")
    print(per_assay.reindex(columns=columns).to_string(index=False))
    print("\n=== post-adjudication pooled precision and recall: point [95% CI] ===")
    print(pooled.drop(columns="assay").to_string(index=False))
    print("\n=== post-adjudication deduplicated precision and recall ===")
    deduplicated = show_deduplicated_tests(
        data_root,
        model,
        baseline=baseline,
        system=system,
        run=run,
        n_resamples=n_resamples,
        excluded_fields_by_assay=exclusions,
    )
    if report_dir is not None:
        destination = Path(report_dir)
        # Metrics are labelled by model, condition pair and run; the audit itself
        # is independent of all three and can be reused across comparisons.
        stem = f"post_adjudication_{model}_{baseline}_vs_{system}_run-{run}"
        field_level.to_csv(destination / f"{stem}_field_level.csv", index=False)
        per_assay.to_csv(destination / f"{stem}_per_assay_tests.csv", index=False)
        pooled.drop(columns="assay").to_csv(destination / f"{stem}_pooled.csv", index=False)
        if deduplicated is not None:
            deduplicated.to_csv(destination / f"{stem}_deduplicated.csv", index=False)
    return {
        "audit": audit,
        "field_level": field_level,
        "per_assay_tests": per_assay,
        "pooled": pooled,
        "deduplicated": deduplicated,
    }


def show_pre_post_adjudication_comparison(
    data_root: str | Path,
    model: str,
    *,
    excluded_fields: Mapping[str, Collection[str]],
    post_deduplicated: pd.DataFrame | None,
    baseline: str,
    system: str,
    run: int = 1,
    n_resamples: int = 10_000,
) -> dict[str, pd.DataFrame]:
    """Print two effect tables: per-assay field-level scores, then deduplicated scores.

    Use the same run and shared records on both sides. ``change`` means post minus
    pre for each method. These are descriptive changes in the evaluation scope,
    not significance tests of a performance change. Keep point estimates unrounded
    until printing; parsing formatted confidence intervals would lose precision.
    """
    per_assay = build_pre_post_precision_recall_table(
        data_root,
        model,
        excluded_fields=excluded_fields,
        baseline=baseline,
        system=system,
        run=run,
    )
    pre_deduplicated = pd.DataFrame(
        deduplicated_paired_tests(
            data_root,
            model,
            baseline=baseline,
            system=system,
            run=run,
            n_resamples=n_resamples,
        )
    ).rename(columns={"baseline": baseline, "system": system})
    keys = ["field_type", "metric", "paired on"]
    columns = [*keys, "n_items", baseline, system]

    def view(table: pd.DataFrame | None, stage: str) -> pd.DataFrame:
        table = table if table is not None else pd.DataFrame()
        return table.reindex(columns=columns).rename(
            columns={
                "n_items": f"n_items_{stage}",
                baseline: f"{baseline} {stage}",
                system: f"{system} {stage}",
            }
        )

    pre_view = view(pre_deduplicated, "pre")
    pre_view["_order"] = range(len(pre_view))
    deduplicated = pre_view.merge(
        view(post_deduplicated, "post"),
        on=keys,
        how="outer",
        sort=False,
        validate="one_to_one",
    )
    deduplicated = deduplicated.sort_values("_order", kind="stable").drop(columns="_order")
    ordered = [*keys, "n_items_pre", "n_items_post"]
    for condition in (baseline, system):
        deduplicated[f"{condition} change"] = deduplicated[f"{condition} post"] - deduplicated[f"{condition} pre"]
        ordered.extend(f"{condition} {stage}" for stage in ("pre", "post", "change"))
    deduplicated = deduplicated[ordered]
    formatters = {f"{condition} change": lambda value: f"{value:+.3f}" for condition in (baseline, system)}
    print("=== pre- and post-adjudication: per assay and pooled; change = post minus pre ===")
    print(per_assay.to_string(index=False, float_format=lambda value: f"{value:.3f}", formatters=formatters))
    print("\n=== pre- and post-adjudication: deduplicated; change = post minus pre ===")
    print(deduplicated.to_string(index=False, float_format=lambda value: f"{value:.3f}", formatters=formatters))
    return {"per_assay": per_assay, "deduplicated": deduplicated}


def count_predictions(data_root: str | Path, model: str, conditions: Sequence[str], *, run: int = 1) -> dict[str, int]:
    """How many predictions each of *conditions* has on disk, across every assay."""
    return {
        condition: sum(
            len(list(assay.output_dir(model, condition, run=run).glob("*.json"))) for assay in iter_assays(data_root)
        )
        for condition in conditions
    }


def show_run_spread(
    data_root: str | Path,
    model: str,
    conditions: Sequence[str],
    *,
    runs: Sequence[int],
    field_types: Sequence[str] = ("ontology", "non_ontology"),
) -> pd.DataFrame:
    """Print how far precision and recall move between *runs*, and return the table.

    One row per (assay, field type), the pooled corpus last; one column per condition and
    metric, each cell ``mean [lowest run, highest run]``.  A narrow bracket means the score
    barely depends on which run produced it.  With no assay scored in every run -- the runs not
    made yet -- it says so and returns an empty table.  The numbers behind each cell are
    :func:`~analysis.data_analysis.create_run_spread_summary`'s.
    """
    frames = []
    for condition in conditions:
        spread = create_run_spread_summary(data_root, model, condition, runs=runs, field_types=field_types)
        spread["cell"] = spread.apply(lambda row: f"{row['mean']:.3f} [{row['min']:.3f}, {row['max']:.3f}]", axis=1)
        spread["column"] = condition_label(condition) + " " + spread["metric"]
        frames.append(spread)
    long = pd.concat(frames, ignore_index=True)
    if long.empty:
        print(
            f"Nothing to show: no assay has predictions in every one of runs {list(runs)} for "
            f"{', '.join(conditions)}.  Make the runs first with run_sweep: n_repeat for how many, first_run to add "
            "them after runs already on disk."
        )
        return pd.DataFrame(columns=["assay", "field_type"])
    table = (
        long.pivot_table(index=["assay", "field_type"], columns="column", values="cell", aggfunc="first", sort=False)
        .reset_index()
        .rename_axis(columns=None)
    )
    columns = [
        f"{condition_label(condition)} {metric}" for condition in conditions for metric in ("precision", "recall")
    ]
    table = table[["assay", "field_type", *[column for column in columns if column in table.columns]]]

    print(f"=== precision and recall over runs {list(runs)}: mean [lowest run, highest run] ===")
    print(table.to_string(index=False))
    return table


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
    conditions: Sequence[str],
    *,
    run: int = 1,
) -> dict[str, pd.DataFrame]:
    """Print the instance-weighted table for every condition with predictions, and return them.

    One row per (assay, field type): one field of one record is one unit, which is the
    workload number -- the fraction of values a curator would have to fix by hand.
    """
    scored = count_predictions(data_root, model, conditions, run=run)
    tables: dict[str, pd.DataFrame] = {}

    print("=== instance-weighted: one field of one record is one unit ===")
    for condition, count in scored.items():
        if not count:
            continue
        frames = [
            create_per_assay_precision_recall_summary(data_root, model, condition, category=field_type, run=run).assign(
                field_type=field_type
            )
            for field_type in CONFUSION_CATEGORIES
        ]
        frames = [frame for frame in frames if len(frame)]
        tables[condition] = order_rows(pd.concat(frames, ignore_index=True))
        print(f"\n--- {condition} ({count} prediction(s)) ---")
        print(tables[condition][WEIGHTED_COLUMNS].to_string(index=False))

    if not tables:
        print(f"  no predictions on disk for any of {list(conditions)}.")
    return tables


def show_deduplicated_tables(
    data_root: str | Path,
    model: str,
    conditions: Sequence[str],
    *,
    run: int = 1,
) -> dict[str, pd.DataFrame]:
    """Print the deduplicated table for every condition with predictions, and return them.

    Same rows, but one distinct value is one unit, which is the capability number: how
    many different things the run gets right rather than how much work it saves.
    """
    scored = count_predictions(data_root, model, conditions, run=run)
    tables: dict[str, pd.DataFrame] = {}

    print("=== deduplicated: one distinct value is one unit ===")
    for condition, count in scored.items():
        if not count:
            continue
        rows = create_per_assay_deduplicated_precision_recall_summary(data_root, model, condition, run=run)
        tables[condition] = order_rows(rows.rename(columns={"category": "field_type"}))
        print(f"\n--- {condition} ---")
        print(tables[condition][DEDUPLICATED_COLUMNS].to_string(index=False))

    if not tables:
        print(f"  no predictions on disk for any of {list(conditions)}.")
    return tables


def show_hypothesis_tests(
    data_root: str | Path,
    model: str,
    *,
    baseline: str,
    system: str,
    alpha: float = 0.05,
    run: int = 1,
) -> pd.DataFrame | None:
    """Print the record-level verdict on precision and recall, and return it.

    An interval from a record-cluster bootstrap says how big the difference is; a
    record-clustered permutation test says how surprising it would be under the null.
    The effective sample size printed afterwards is the caveat on both: it counts how
    much independent evidence the corpus holds once values repeated across records are
    clustered together.  Returns ``None`` when either condition has no predictions.
    """
    scored = count_predictions(data_root, model, (baseline, system), run=run)
    if not (scored[baseline] and scored[system]):
        print(
            f"Nothing to test: found {scored[baseline]} prediction(s) for {baseline!r} "
            f"and {scored[system]} for {system!r}; both are needed for a paired test."
        )
        return None

    print(f"H0: {system} and {baseline} are interchangeable.  Differences are {system} minus {baseline}.\n")

    intervals = build_precision_recall_table(data_root, model, baseline=baseline, system=system, run=run)
    intervals = intervals[intervals["metric"].isin(METRICS)]
    print("=== 95% confidence intervals, record-cluster bootstrap ===")
    print(intervals.to_string(index=False))

    pooled = PairedData()
    for assay in iter_assays(data_root):
        pooled.extend(collect_paired_data(data_root, model, assay.key, baseline=baseline, system=system, run=run))

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

    show_effective_sample_size(data_root, model, system, run=run)
    return table


def show_effective_sample_size(data_root: str | Path, model: str, condition: str, *, run: int = 1) -> None:
    """Print how much independent evidence the corpus holds for *condition*."""
    print(f"\n=== effective sample size ({condition} field outcomes, clustered by (assay, field, value)) ===")
    print("  N_eff far below N means the corpus holds less independent evidence than the record count suggests.")
    for field_type in (None, "ontology", "non_ontology"):
        ess = effective_sample_size(data_root, model, condition, field_type=field_type, run=run)
        print(
            f"  {field_type or 'all':<13} N={ess['n']:.0f} clusters={ess['n_clusters']:.0f} "
            f"ICC={ess['icc']:.3f} DEFF={ess['design_effect']:.1f} N_eff={ess['n_effective']:.0f}"
        )


def show_deduplicated_tests(
    data_root: str | Path,
    model: str,
    *,
    baseline: str,
    system: str,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    run: int = 1,
    excluded_fields_by_assay: Mapping[str, Collection[str]] | None = None,
) -> pd.DataFrame | None:
    """Print the same question asked of distinct values, and return the verdict table.

    Recall pairs on the distinct gold value, precision on the field -- the two conditions
    assert different values, so precision has no shared item list to pair on.  Returns
    ``None`` when either condition has no predictions.
    """
    scored = count_predictions(data_root, model, (baseline, system), run=run)
    if not (scored[baseline] and scored[system]):
        print(f"Nothing to test: {baseline!r} and {system!r} must both have predictions.")
        return None

    rows = deduplicated_paired_tests(
        data_root,
        model,
        baseline=baseline,
        system=system,
        n_resamples=n_resamples,
        run=run,
        excluded_fields_by_assay=excluded_fields_by_assay,
    )
    if not rows:
        print("Nothing to test: no retained paired values or fields after exclusions.")
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table["difference [95% CI]"] = table.apply(
        lambda row: f"{row['delta']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]", axis=1
    )
    table["p_value"] = table["pvalue"].round(4)
    table[f"reject H0 at {alpha}"] = ["yes" if pvalue < alpha else "no" for pvalue in table["pvalue"]]
    table = table.rename(columns={"baseline": baseline, "system": system})

    print(f"=== deduplicated: paired over items, not records ({n_resamples:,} resamples) ===")
    columns = [
        "field_type",
        "metric",
        "paired on",
        "n_items",
        baseline,
        system,
        "difference [95% CI]",
        "p_value",
        f"reject H0 at {alpha}",
    ]
    print(table[columns].round({baseline: 3, system: 3}).to_string(index=False))
    print("\nA row significant here is not explained by repetition.  A row significant in the")
    print("instance-weighted test but not here is a real saving of work, carried by values the")
    print("corpus repeats -- not evidence that the run knows more distinct answers.")
    return table


def show_error_subcategories_by_field_type(errors: pd.DataFrame) -> pd.DataFrame:
    """Print subcategory counts by field type and return the table, including totals.

    Count the supplied rows once each. Pass deduplicated errors to count distinct
    disagreements, or the full collection to count record–field instances. All seven
    subcategories and both field types are retained, including zeros.
    """
    table = summarize_error_subcategories_by_field_type(errors)
    table.loc[len(table)] = ["Total", *table[["ontology", "non_ontology", "total"]].sum().tolist()]
    print("\n    sub-categories by field type:")
    print(table.to_string(index=False))
    return table


def show_error_analysis(
    data_root: str | Path,
    model: str,
    condition: str,
    *,
    field_type: str | None = None,
    apply_dedup: bool = False,
    top_fields: int = 10,
    run: int = 1,
    confirmed_value_mappings: dict[str, dict[tuple[str, str], str]] | None = None,
) -> pd.DataFrame:
    """Print *condition*'s reference disagreements by category and subcategory.

    Substitutions split into ``near_match``, ``wrong_mapping``, and ``different_value``;
    omissions into ``missed_value`` and ``external_gap``; insertions into
    ``unexpected_copy`` and ``unexpected_fill``.  Near matches allow equivalent text,
    numbers, DOI resolver URLs, dataset-path notation, read-length separators, and value
    mappings confirmed by a reviewer to preserve the same information.  Text containment,
    legacy preservation, and permitted-vocabulary membership alone do not qualify.
    Different values include remaining substitutions, even when
    copied from the same legacy field.  Wrong mappings are candidates identified by a
    matching value in another legacy field, excluding reviewed valid source mappings.
    ATACseq's assay_type, cell_barcode_offset, and cell_barcode_size correctly map to
    dataset_type, barcode_offset, and barcode_size; differing values from those sources
    are different_value. Other source-to-target semantics still need review.
    Omissions are ``missed_value`` only when the reference is established by a value
    at the target field or a supported legacy source alias. Related methods, kits,
    barcode fields, and PCR counts alone cannot establish UMI values. Missing new-schema
    fields suggest ``external_gap`` unless a supported alias supplies the value.
    External gaps remain candidates pending external-provenance verification.
    ``legacy_hint_sources`` records context without deciding the label;
    ``legacy_support_sources``, ``legacy_target_present``, and ``omission_basis``
    expose the actual evidence used.
    Legacy-value lookup includes nested objects and list items.

    Two tables count by assay: the categories first, then their subcategories. A third
    printed table counts subcategories by field type, including a total row, using
    :func:`show_error_subcategories_by_field_type`. It follows the same deduplication and
    field-type filter. Both category levels are printed because they answer different questions -- the category
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
    errors = collect_field_errors(
        data_root, model, condition, run=run, confirmed_value_mappings=confirmed_value_mappings
    )
    if errors.empty:
        print(f"No errors to categorise: {condition!r} has no predictions under {model!r}.")
        return errors

    # Reconciled before any deduplication: the check is that the collection lost nothing,
    # which is a fact about every instance and not about the reading chosen below.
    counts = reconcile_with_confusion(errors, data_root, model, condition, run=run)
    print(f"=== {condition}: every counted error, categorised ===")
    for cell, totals in counts.items():
        agree = "accounted for" if totals["counted"] == totals["categorised"] else "MISMATCH"
        print(f"  {cell}: {totals['counted']} counted, {totals['categorised']} categorised -- {agree}")
    print("  near_match: a different representation confirmed to convey the same information")
    print("  wrong_mapping: candidate mapping error, excluding reviewed valid source mappings")
    print("  missed_value: reference established by a matching target field or supported legacy alias")
    print("  external_gap: legacy information does not establish the reference; external provenance requires review")
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
    show_error_subcategories_by_field_type(selected)

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
        sources = row.get("legacy_sources", {})
        print(f"    legacy sources  {json.dumps(sources, ensure_ascii=False)}")
        if row.get("omission_basis"):
            print(f"    omission basis  {row['omission_basis']}")
            print(f"    legacy hints    {json.dumps(row.get('legacy_hint_sources', {}), ensure_ascii=False)}")
            print(f"    legacy support  {json.dumps(row.get('legacy_support_sources', {}), ensure_ascii=False)}")
            print(f"    target present  {row.get('legacy_target_present')}")
        if row["near_match_reasons"]:
            print(f"    near-match evidence  {row['near_match_reasons']}")
        if row["resolution"]:
            print(f"    the run called this {row['resolution']!r}: {row['reasoning']}")
    return selected.head(n)
