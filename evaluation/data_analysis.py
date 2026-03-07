"""Accuracy summaries and error analysis for metadata standardization evaluation.

Provides functions to measure transformation quality at different granularities:
per-assay accuracy summaries, field-level error classification, and
aggregated error reports for diagnosing systematic transformation failures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from metrics import _get_ontology_constrained_fields, _is_missing, _values_match

_BOOLEAN_STRINGS = {"yes", "no", "true", "false"}

_DOI_PATTERN = re.compile(r"https?://(dx\.)?doi\.org/")


def create_per_assay_accuracy_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Compute average accuracy per assay across all samples.

    Iterates over assays defined in ``ASSAY_ORDER``, calls :func:`apply_metrics`
    for each, and returns a single DataFrame with one row per assay containing
    the mean accuracy for each metric.
    """
    import pandas as pd

    from assays import ASSAY_ORDER

    rows: list[dict[str, Any]] = []
    for assay_key, assay_label in ASSAY_ORDER:
        df = apply_metrics(
            Path(data_root, assay_key, "output", model, run_type),
            Path(data_root, assay_key, "gold"),
            Path(data_root, "schemas", f"{assay_key}.json"),
        )
        if df.empty:
            continue
        means = df[
            [
                "ontology_constrained_field_accuracy",
                "non_ontology_constrained_field_accuracy",
                "all_field_accuracy",
            ]
        ].mean()
        rows.append(
            {
                "assay": assay_label,
                "ontology_constrained_field_accuracy": round(means["ontology_constrained_field_accuracy"], decimal_places),
                "non_ontology_constrained_field_accuracy": round(means["non_ontology_constrained_field_accuracy"], decimal_places),
                "all_field_accuracy": round(means["all_field_accuracy"], decimal_places),
            }
        )
    return pd.DataFrame(rows)


def create_overall_accuracy_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Compute average overall accuracy per assay using :func:`compute_overall_accuracy`.

    Unlike :func:`create_per_assay_accuracy_summary` which delegates to
    :func:`apply_metrics`, this function calls :func:`compute_overall_accuracy`
    directly for each predicted/gold file pair and averages the results per assay.
    """
    import json

    import pandas as pd

    from assays import ASSAY_ORDER
    from metrics import compute_overall_accuracy

    rows: list[dict[str, Any]] = []
    for assay_key, assay_label in ASSAY_ORDER:
        output_dir = Path(data_root, assay_key, "output", model, run_type)
        gold_dir = Path(data_root, assay_key, "gold")
        schema_path = Path(data_root, "schemas", f"{assay_key}.json")

        predicted_files = sorted(output_dir.glob("*.json"))
        if not predicted_files:
            continue

        accuracies: list[dict[str, float]] = []
        for pred_file in predicted_files:
            gold_file = gold_dir / pred_file.name
            if not gold_file.exists():
                continue
            with open(pred_file) as f:
                predicted = json.load(f)
            with open(gold_file) as f:
                gold = json.load(f)
            accuracies.append(compute_overall_accuracy(predicted, gold, schema_path))

        if not accuracies:
            continue

        acc_df = pd.DataFrame(accuracies)
        means = acc_df.mean()
        rows.append(
            {
                "assay": assay_label,
                "ontology_constrained_accuracy": round(means["ontology_constrained_accuracy"], decimal_places),
                "non_ontology_constrained_accuracy": round(means["non_ontology_constrained_accuracy"], decimal_places),
                "all_field_accuracy": round(means["all_field_accuracy"], decimal_places),
            }
        )
    return pd.DataFrame(rows)


def apply_metrics(input_dir: Path, gold_dir: Path, schema_path: Path) -> pd.DataFrame:
    """Compare predicted outputs in *input_dir* against gold standards in *gold_dir*."""
    import pandas as pd

    from metrics import (
        compute_all_field_accuracy,
        compute_non_ontology_constrained_field_accuracy,
        compute_ontology_constrained_field_accuracy,
    )

    input_files = sorted(input_dir.glob("*.json"))
    results: list[dict[str, Any]] = []

    for input_file in input_files:
        gold_file = gold_dir / input_file.name
        if not gold_file.exists():
            continue

        with open(input_file) as f:
            predicted = json.load(f)
        with open(gold_file) as f:
            gold = json.load(f)

        results.append(
            {
                "input_file": input_file.name,
                "ontology_constrained_field_accuracy": compute_ontology_constrained_field_accuracy(
                    predicted, gold, schema_path
                ),
                "non_ontology_constrained_field_accuracy": compute_non_ontology_constrained_field_accuracy(
                    predicted, gold, schema_path
                ),
                "all_field_accuracy": compute_all_field_accuracy(predicted, gold),
            }
        )

    return pd.DataFrame(results)


def analyze_prediction_errors(
    assays: list[str],
    data_root: str,
    models: list[str],
    condition: str = "baseline",
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of field-level prediction errors.

    Each row represents one field where the predicted value does not match the
    gold standard.  The ``error_type`` column categorises the mismatch to aid
    diagnosis of systematic issues.

    Parameters
    ----------
    assays:
        Assay names to evaluate (e.g. ``["rnaseq"]``).
    data_root:
        Path to the ``data/`` directory containing ``schemas/``, gold files,
        and model outputs.
    models:
        Model names to evaluate (e.g. ``["gpt41mini"]``).
    condition:
        Output sub-directory under each model (``"baseline"`` or
        ``"experiment"``).
    match_case:
        Whether string comparison is case-sensitive.
    match_whole_word:
        Whether to require exact match (``True``) or substring containment.
    """
    import pandas as pd

    root = Path(data_root)
    rows: list[dict[str, Any]] = []

    for assay in assays:
        schema_path = root / "schemas" / f"{assay}.json"
        ontology_fields = set(_get_ontology_constrained_fields(schema_path))

        gold_dir = root / assay / "gold"

        for model in models:
            output_dir = root / assay / "output" / model / condition

            for gold_file in sorted(gold_dir.glob("*.json")):
                pred_file = output_dir / gold_file.name
                if not pred_file.exists():
                    continue

                with open(gold_file) as f:
                    gold = json.load(f)
                with open(pred_file) as f:
                    predicted = json.load(f)

                for field, gold_val in gold.items():
                    pred_val = predicted.get(field)
                    error_type = _classify_error(
                        field,
                        gold_val,
                        pred_val,
                        match_case=match_case,
                        match_whole_word=match_whole_word,
                    )
                    if error_type is not None:
                        rows.append(
                            {
                                "assay": assay,
                                "model": model,
                                "file": gold_file.name,
                                "field": field,
                                "error_type": error_type,
                                "field_type": "ontology-constrained"
                                if field in ontology_fields
                                else "non-ontology-constrained",
                                "gold_value": gold_val,
                                "predicted_value": pred_val,
                            }
                        )

    return pd.DataFrame(
        rows,
        columns=[
            "assay",
            "model",
            "file",
            "field",
            "error_type",
            "field_type",
            "gold_value",
            "predicted_value",
        ],
    )


def build_error_report(
    assays: list[str],
    data_root: str,
    models: list[str],
    condition: str = "baseline",
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> pd.DataFrame:
    """Aggregate field-level prediction errors into a summary report.

    Calls :func:`analyze_prediction_errors` and groups the results by error
    pattern (error_type, field_type, expected value, predicted value), counting
    how often each pattern occurs and which assays are affected.

    Returns a DataFrame sorted by frequency (descending) with columns:
    ``error_type``, ``field_type``, ``expected_value``, ``predicted_value``,
    ``frequency``, ``assays``.
    """
    import pandas as pd

    errors_df = analyze_prediction_errors(
        assays,
        data_root,
        models,
        condition,
        match_case=match_case,
        match_whole_word=match_whole_word,
    )

    if errors_df.empty:
        return pd.DataFrame(
            columns=["error_type", "field_type", "expected_value", "predicted_value", "frequency", "assays"]
        )

    errors_df = errors_df.copy()
    errors_df["expected_value"] = errors_df["field"] + ": " + errors_df["gold_value"].astype(str)
    errors_df["predicted_value"] = errors_df["field"] + ": " + errors_df["predicted_value"].astype(str)

    report = (
        errors_df.groupby(["error_type", "field_type", "expected_value", "predicted_value"])
        .agg(
            frequency=("field", "size"),
            assays=("assay", lambda s: ", ".join(sorted(s.unique()))),
        )
        .reset_index()
        .sort_values("frequency", ascending=False, ignore_index=True)
    )

    return report


def _is_boolean_like(value: str) -> bool:
    """Return ``True`` if *value* looks like a boolean string."""
    return value.strip().lower() in _BOOLEAN_STRINGS


def _classify_error(
    field: str,
    gold_val: Any,
    predicted_val: Any,
    *,
    match_case: bool,
    match_whole_word: bool,
) -> str | None:
    """Classify the error between *gold_val* and *predicted_val*.

    Returns an error-type string, or ``None`` when the values match (i.e. no
    error).
    """
    gold_missing = _is_missing(gold_val)
    pred_missing = _is_missing(predicted_val)

    # Both missing → correct
    if gold_missing and pred_missing:
        return None

    # Null mismatch categories
    if gold_missing and not pred_missing:
        return "hallucinated_non_null"
    if not gold_missing and pred_missing:
        return "missed_non_null"

    # Both present – check if they already match
    if _values_match(
        predicted_val,
        gold_val,
        match_case=match_case,
        match_whole_word=match_whole_word,
        field_name=field,
    ):
        return None

    # --- Both present, values differ – classify the mismatch ---

    # Type mismatch (str vs int, etc.)
    if type(gold_val) is not type(predicted_val):
        return "type_mismatch"

    # Non-string types: value differs
    if not isinstance(gold_val, str):
        return "value_mismatch"

    # --- Both are strings from here ---

    gold_str = gold_val.strip()
    pred_str = predicted_val.strip()
    gold_lower = gold_str.lower()
    pred_lower = pred_str.lower()

    # Boolean representation ("No"/"Yes" vs "false"/"true")
    if _is_boolean_like(gold_str) and _is_boolean_like(pred_str):
        return "boolean_representation"

    # DOI format (URL vs bare DOI)
    if field.endswith("_doi"):
        gold_is_url = bool(_DOI_PATTERN.match(gold_str))
        pred_is_url = bool(_DOI_PATTERN.match(pred_str))
        if gold_is_url != pred_is_url:
            return "doi_format"

    # Delimiter / case normalization
    gold_norm = re.sub(r"[\s_\-]+", "", gold_lower)
    pred_norm = re.sub(r"[\s_\-]+", "", pred_lower)
    if gold_norm == pred_norm:
        return "delimiter_or_case"

    # Numeric format in string (different separators between numbers)
    gold_digits = re.sub(r"[^0-9]", " ", gold_str).split()
    pred_digits = re.sub(r"[^0-9]", " ", pred_str).split()
    if gold_digits and gold_digits == pred_digits:
        return "numeric_format_in_string"

    return "wrong_value"
