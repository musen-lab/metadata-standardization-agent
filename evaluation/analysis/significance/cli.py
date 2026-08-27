"""Command-line entry point: print (and optionally save) every significance table."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.significance.clustering import effective_sample_size
from analysis.significance.paired_data import CATEGORIES, CATEGORY_LABELS
from analysis.significance.tables import build_overall_table, build_per_assay_table, build_precision_recall_table


def main() -> None:
    """Run every table for one model and print it; write CSVs when ``--csv-dir`` is given."""
    parser = argparse.ArgumentParser(description="Confidence intervals and paired significance tests.")
    parser.add_argument("--data-root", default="../data", help="Path to the data/ directory.")
    parser.add_argument("--model", default="gpt5mini", help="Model output sub-directory (e.g. gpt5mini).")
    parser.add_argument("--csv-dir", default=None, help="Optional directory to write CSV tables.")
    parser.add_argument("--baseline-run", default="baseline", help="Output sub-directory of the run compared against.")
    parser.add_argument("--system-run", default="agent-tool", help="Output sub-directory of the run tested.")
    args = parser.parse_args()

    runs = {"baseline_run": args.baseline_run, "system_run": args.system_run}
    overall = build_overall_table(args.data_root, args.model, **runs)
    print("\n=== Overall (pooled across assays) ===")
    print(overall.to_string(index=False))

    prf = build_precision_recall_table(args.data_root, args.model, **runs)
    print("\n=== Precision / recall / F1 (record cluster bootstrap, pooled) ===")
    print(prf.to_string(index=False))

    print(f"\n=== Effective sample size ({args.system_run} field outcomes, clustered by (assay, field, value)) ===")
    for cat, label in [(None, "all"), ("ontology", "ontology"), ("non_ontology", "non_ontology")]:
        e = effective_sample_size(args.data_root, args.model, args.system_run, field_type=cat)
        print(
            f"  {label:13s} N={e['n']:.0f} clusters={e['n_clusters']:.0f} "
            f"ICC={e['icc']:.3f} DEFF={e['design_effect']:.1f} N_eff={e['n_effective']:.0f}"
        )

    per_assay = {cat: build_per_assay_table(args.data_root, args.model, cat, **runs) for cat in CATEGORIES}
    for cat in CATEGORIES:
        print(f"\n=== Per assay: {CATEGORY_LABELS[cat]} ===")
        print(per_assay[cat].to_string(index=False))

    if args.csv_dir:
        out = Path(args.csv_dir)
        out.mkdir(parents=True, exist_ok=True)
        overall.to_csv(out / f"significance_overall_{args.model}.csv", index=False)
        prf.to_csv(out / f"significance_precision_recall_{args.model}.csv", index=False)
        for cat in CATEGORIES:
            per_assay[cat].to_csv(out / f"significance_per_assay_{cat}_{args.model}.csv", index=False)
        print(f"\nWrote CSV tables to {out}/")
