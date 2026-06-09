"""Quantify the *cause* of each prediction error, for the error analysis.

This answers Reviewer 2's request to quantify recurring error types -- "how often
failures were due to missing ontology search results, unavailable contextual
information, field-mapping confusion, or lack of access to external protocol
documents."  It builds on :func:`data_analysis.analyze_prediction_errors`, which
already returns one row per wrong field, and assigns each error a *cause*:

* ``format_or_variant`` -- a surface mismatch (case/delimiter, boolean wording,
  numeric formatting, DOI URL vs bare DOI).  Reuses the existing surface-level
  ``error_type`` labels.
* ``field_mapping_confusion`` -- the predicted value is actually the legacy value
  of a *different* field (e.g. ``preparation_protocol_doi`` filled from
  ``section_prep_protocols_io_doi``).
* ``external_information_required`` -- the gold value does not appear anywhere in
  the legacy input record, so it could only have come from an external source such
  as a protocol document (e.g. the UMI configuration fields).  Restricted to
  non-ontology fields, because ontology fields legitimately differ from the legacy
  text after canonicalization.
* ``missing_ontology_result`` -- (requires traces) the agent searched BioPortal
  with the legacy value but the search returned no candidates, so it kept the
  legacy value (e.g. ``AxioScan.Z1``).  Detected by mining the LangSmith trace
  CSVs for ``term_search`` tool calls that returned ``totalCount: 0``.
* ``other`` -- everything else (e.g. an ontology field where candidates were
  returned but the wrong one was chosen).

The first three causes are computed from the input/gold/prediction files alone
(no trace files, no API calls).  ``missing_ontology_result`` is an optional refine
pass over the (large) experiment trace files; pass ``--traces`` to enable it.

Output is reported separately for the baseline and the tool-use agent (``--run-type
both``, the default), each with a coarse ``cause`` breakdown and a finer
``error_type`` breakdown (which separates Boolean and DOI failures).

Run from the ``evaluation/`` directory::

    uv run python error_causes.py --data-root ../data --model gpt5mini
    uv run python error_causes.py --data-root ../data --model gpt5mini \
        --run-type both --traces ../traces/gpt5mini --csv-dir out/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_analysis import analyze_prediction_errors
from metrics import _get_required_fields

if TYPE_CHECKING:
    import pandas as pd

# Surface-level error_type labels (from data_analysis._classify_error) that are
# best described as formatting / variant-spelling differences.
_FORMAT_ERROR_TYPES = {
    "delimiter_or_case",
    "boolean_representation",
    "numeric_format_in_string",
    "doi_format",
}

_SEARCH_STRING_RE = re.compile(r"'search_string':\s*'((?:[^'\\]|\\.)*)'")
_TOTAL_COUNT_RE = re.compile(r'"totalCount":\s*(\d+)')


def _norm(value: Any) -> str:
    """Normalize a value to a comparable lowercase string (empty for missing)."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _legacy_value_index(input_record: dict[str, Any]) -> dict[str, set[str]]:
    """Map each normalized legacy value -> set of field names that hold it."""
    index: dict[str, set[str]] = {}
    for fname, val in input_record.items():
        index.setdefault(_norm(val), set()).add(fname)
    return index


def classify_cause(row: dict[str, Any], input_record: dict[str, Any], empty_terms: set[str] | None) -> str:
    """Classify one error row into a cause (see module docstring)."""
    field = row["field"]
    gold_norm = _norm(row["gold_value"])
    pred_norm = _norm(row["predicted_value"])
    is_ontology = row["field_type"] == "ontology-constrained"
    legacy_index = _legacy_value_index(input_record)
    legacy_self = _norm(input_record.get(field))

    # 1. Missing ontology result (trace-derived, checked first as the root cause):
    #    the agent searched the legacy value, BioPortal returned nothing, so it
    #    kept the legacy value.  This often *also* looks like a surface mismatch
    #    (e.g. "AxioScan.Z1" vs "Axio Scan.Z1"), but the empty search is the cause.
    if (
        empty_terms is not None
        and is_ontology
        and legacy_self
        and legacy_self in empty_terms
        and pred_norm == legacy_self
    ):
        return "missing_ontology_result"

    # 2. Surface formatting / variant spelling.
    if row["error_type"] in _FORMAT_ERROR_TYPES:
        return "format_or_variant"

    # 3. Field-mapping confusion: the prediction is some *other* field's legacy value.
    if pred_norm:
        holders = legacy_index.get(pred_norm, set())
        if holders and field not in holders:
            return "field_mapping_confusion"

    # 4. External information required: gold value is nowhere in the legacy record.
    #    Limited to non-ontology fields (ontology fields differ after canonicalization).
    if gold_norm and not is_ontology:
        present = gold_norm in legacy_index or any(gold_norm in lv for lv in legacy_index if lv)
        if not present:
            return "external_information_required"

    return "other"


def extract_empty_search_terms(trace_dir: str | Path) -> set[str]:
    """Return the set of normalized search strings that ever returned no candidates.

    Streams the ``*experiment*`` LangSmith trace CSVs in *trace_dir* and, for each
    ``term_search`` tool call, records the search string and whether the BioPortal
    response had ``totalCount == 0``.  A search string is reported as "empty" only
    if it *never* returned candidates in any call (so a term that succeeds elsewhere
    is not mislabeled).
    """
    trace_dir = Path(trace_dir)
    csv.field_size_limit(sys.maxsize)
    empty: set[str] = set()
    nonempty: set[str] = set()

    files = sorted(trace_dir.glob("*experiment*.csv"))
    for path in files:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for trace_row in reader:
                if trace_row.get("run_type") != "tool":
                    continue
                if "term_search" not in (trace_row.get("name") or ""):
                    continue
                m_term = _SEARCH_STRING_RE.search(trace_row.get("inputs") or "")
                if not m_term:
                    continue
                term = m_term.group(1).replace("\\'", "'").strip().lower()
                m_count = _TOTAL_COUNT_RE.search(trace_row.get("outputs") or "")
                total = int(m_count.group(1)) if m_count else 0
                if total == 0:
                    empty.add(term)
                else:
                    nonempty.add(term)

    return empty - nonempty


def build_error_detail(
    data_root: str | Path,
    model: str,
    run_type: str,
    *,
    trace_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return one row per error with added ``cause`` and ``run_type`` columns.

    The trace-derived ``missing_ontology_result`` cause is only meaningful for the
    tool-using agent, so trace mining is skipped when ``run_type == "baseline"``.
    """
    errors = analyze_prediction_errors(str(data_root), model, run_type)
    empty_terms = extract_empty_search_terms(trace_dir) if (trace_dir and run_type == "experiment") else None

    root = Path(data_root)
    input_cache: dict[tuple[str, str], dict[str, Any]] = {}
    required_cache: dict[str, set[str]] = {}
    causes: list[str] = []
    required_flags: list[bool] = []
    for _, row in errors.iterrows():
        assay = row["assay"]
        key = (assay, row["file"])
        if key not in input_cache:
            input_file = root / assay / "input" / row["file"]
            input_cache[key] = json.loads(input_file.read_text()) if input_file.exists() else {}
        if assay not in required_cache:
            required_cache[assay] = set(_get_required_fields(root / "schemas" / f"{assay}.json"))
        causes.append(classify_cause(row.to_dict(), input_cache[key], empty_terms))
        required_flags.append(row["field"] in required_cache[assay])

    return errors.assign(cause=causes, run_type=run_type, required=required_flags)


def _summarize(detail: pd.DataFrame, key: str) -> pd.DataFrame:
    """Count errors grouped by ``field_type`` and *key*, with a percentage column."""
    import pandas as pd

    if detail.empty:
        return pd.DataFrame(columns=["field_type", key, "count", "pct_of_all_errors"])
    summary = detail.groupby(["field_type", key]).size().rename("count").reset_index()
    summary["pct_of_all_errors"] = (100 * summary["count"] / len(detail)).round(1)
    return summary.sort_values("count", ascending=False, ignore_index=True)


def summarize_causes(detail: pd.DataFrame) -> pd.DataFrame:
    """Counts by field type x cause (missing_ontology_result, field_mapping_confusion, ...)."""
    return _summarize(detail, "cause")


def summarize_error_types(detail: pd.DataFrame) -> pd.DataFrame:
    """Counts by field type x surface error type (boolean_representation, doi_format, ...).

    This finer view separates the specific mechanisms the paper calls out -- e.g.
    Boolean conversion failures (``boolean_representation``) and DOI reformatting
    failures (``doi_format``) -- which the coarser ``cause`` view lumps under
    ``format_or_variant``.
    """
    return _summarize(detail, "error_type")


def build_error_cause_table(
    data_root: str | Path,
    model: str,
    run_type: str = "experiment",
    *,
    trace_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(detail, cause_summary)`` for one condition (kept for convenience)."""
    detail = build_error_detail(data_root, model, run_type, trace_dir=trace_dir)
    return detail, summarize_causes(detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify prediction-error causes.")
    parser.add_argument("--data-root", default="../data", help="Path to the data/ directory.")
    parser.add_argument("--model", default="gpt5mini", help="Model output sub-directory (e.g. gpt5mini).")
    parser.add_argument(
        "--run-type",
        default="both",
        choices=["baseline", "experiment", "both"],
        help="Which condition(s) to analyze; 'both' reports baseline and ARMS separately.",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Optional path to the model's trace directory (enables missing_ontology_result for ARMS).",
    )
    parser.add_argument("--csv-dir", default=None, help="Optional directory to write CSV tables.")
    args = parser.parse_args()

    run_types = ["baseline", "experiment"] if args.run_type == "both" else [args.run_type]
    for run_type in run_types:
        if run_type == "experiment" and args.traces:
            print(f"\nMining traces in {args.traces} for empty BioPortal searches (this may take a few minutes)...")
        detail = build_error_detail(args.data_root, args.model, run_type, trace_dir=args.traces)
        causes = summarize_causes(detail)
        error_types = summarize_error_types(detail)

        label = "Baseline (prompt-only)" if run_type == "baseline" else "ARMS (tool-use)"
        print(f"\n=== {label}: {len(detail)} total errors ({args.model}) ===")
        print("\n-- by cause --")
        print(causes.to_string(index=False))
        print("\n-- by error type (surface mechanism) --")
        print(error_types.to_string(index=False))
        if not detail.empty:
            missed = detail[detail["error_type"] == "missed_non_null"]
            missed_required = int(missed["required"].sum())
            print(
                f"\n-- of {len(missed)} missed-value errors (gold has a value, prediction blank), "
                f"{missed_required} are on schema-required fields --"
            )
        if run_type == "experiment" and args.traces is None:
            print("\nNote: 'missing_ontology_result' not computed (pass --traces to enable it);")
            print("those errors currently fall under 'format_or_variant' or 'other'.")

        if args.csv_dir:
            out = Path(args.csv_dir)
            out.mkdir(parents=True, exist_ok=True)
            detail.to_csv(out / f"error_causes_detail_{args.model}_{run_type}.csv", index=False)
            causes.to_csv(out / f"error_causes_by_cause_{args.model}_{run_type}.csv", index=False)
            error_types.to_csv(out / f"error_causes_by_type_{args.model}_{run_type}.csv", index=False)
            print(f"\nWrote CSV tables to {out}/")


if __name__ == "__main__":
    main()
