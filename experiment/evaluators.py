"""Strict deterministic SDK evaluators backed by the existing metric definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langfuse import Evaluation

from analysis.metrics.accuracy import _compute_field_counts, compute_overall_accuracy
from analysis.metrics.confusion import CONFUSION_KEYS, compute_field_confusion, precision_recall_f1

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langfuse.experiment import ExperimentItemResult

ITEM_SCORE_NAMES = frozenset(
    {
        "precision",
        "recall",
        "f1",
        "all_field_accuracy",
        "ontology_constrained_accuracy",
        "non_ontology_constrained_accuracy",
    }
)
RUN_SCORE_NAMES = frozenset(
    {"micro_precision", "micro_recall", "micro_f1", "attempted_records", "completed_records", "failed_records"}
    | {f"field_weighted_{name}" for name in ITEM_SCORE_NAMES if name.endswith("accuracy")}
)


def _validate_pair(output: Any, gold: Any) -> None:
    if not isinstance(output, dict) or not output or not isinstance(gold, dict) or not gold:
        raise ValueError("Evaluators require nonempty predicted and gold objects")


def make_evaluators(schema_path: Path, *, expected_records: int) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return item and run evaluators with strict case-sensitive whole-value matching.

    Failed/missing item results are excluded from ratios, but counted explicitly;
    the CLI rejects incomplete runs. Corpus metrics sum counts, not record means.
    """
    if expected_records < 1:
        raise ValueError("expected_records must be positive")

    def item_evaluator(*, output: Any, expected_output: Any, **kwargs: Any) -> list[Evaluation]:
        """Evaluate a migrated record; only this evaluator accesses its gold."""
        _validate_pair(output, expected_output)
        scores = precision_recall_f1(compute_field_confusion(output, expected_output, schema_path)["all"])
        scores.update(compute_overall_accuracy(output, expected_output, schema_path))
        return [Evaluation(name=name, value=value, data_type="NUMERIC") for name, value in scores.items()]

    def run_evaluator(*, item_results: list[ExperimentItemResult], **kwargs: Any) -> list[Evaluation]:
        """Aggregate raw confusion and correct/total counts over fully scored items."""
        confusion = dict.fromkeys(CONFUSION_KEYS, 0)
        fields = dict.fromkeys(("ontology_correct", "ontology_total", "non_ontology_correct", "non_ontology_total"), 0)
        completed = 0
        for result in item_results:
            if {evaluation.name for evaluation in result.evaluations} != ITEM_SCORE_NAMES:
                continue
            gold = result.item["expected_output"]
            _validate_pair(result.output, gold)
            record_confusion = compute_field_confusion(result.output, gold, schema_path)["all"]
            record_fields = _compute_field_counts(result.output, gold, schema_path)
            for key in confusion:
                confusion[key] += record_confusion[key]
            for key in fields:
                fields[key] += record_fields[key]
            completed += 1
        scores = {f"micro_{name}": value for name, value in precision_recall_f1(confusion).items()}
        for category, name in (
            ("ontology", "ontology_constrained_accuracy"),
            ("non_ontology", "non_ontology_constrained_accuracy"),
            ("all", "all_field_accuracy"),
        ):
            prefixes = ("ontology", "non_ontology") if category == "all" else (category,)
            correct = sum(fields[f"{prefix}_correct"] for prefix in prefixes)
            total = sum(fields[f"{prefix}_total"] for prefix in prefixes)
            scores[f"field_weighted_{name}"] = correct / total if total else 0.0
        scores.update(
            attempted_records=expected_records,
            completed_records=completed,
            failed_records=expected_records - completed,
        )
        comment = "Ratios cover completed, fully scored records only; inspect coverage before comparing runs."
        return [
            Evaluation(name=name, value=value, comment=comment, data_type="NUMERIC") for name, value in scores.items()
        ]

    return item_evaluator, run_evaluator
