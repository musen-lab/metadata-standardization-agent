"""Existing strict metric parity and count-weighted, not record-mean, aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langfuse.experiment import ExperimentItemResult

from analysis.metrics.accuracy import compute_overall_accuracy
from analysis.metrics.confusion import compute_field_confusion, precision_recall_f1
from experiment.evaluators import ITEM_SCORE_NAMES, make_evaluators

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_existing_definitions_and_weighted_aggregation(corpus: tuple[Path, Callable]) -> None:
    """Cover match, both blank, substitution, deletion, insertion and category splits."""
    root, _ = corpus
    schema = root / "schemas/rnaseq.json"
    item_eval, run_eval = make_evaluators(schema, expected_records=3)
    gold = {"a": "X", "b": None, "c": "right", "d": "missing", "e": ""}
    predicted = {"a": "X", "b": "  ", "c": "wrong", "e": "inserted"}
    results = []
    for output, expected in ((predicted, gold), ({"a": "X"}, {"a": "X"})):
        evaluations = item_eval(output=output, expected_output=expected)
        scores = {evaluation.name: evaluation.value for evaluation in evaluations}
        assert set(scores) == ITEM_SCORE_NAMES
        assert scores == {
            **compute_overall_accuracy(output, expected, schema),
            **precision_recall_f1(compute_field_confusion(output, expected, schema)["all"]),
        }
        results.append(
            ExperimentItemResult(
                item={"expected_output": expected},
                output=output,
                evaluations=evaluations,
                trace_id=None,
                dataset_run_id=None,
            )
        )
    scores = {evaluation.name: evaluation.value for evaluation in run_eval(item_results=results)}
    assert scores["micro_precision"] == scores["micro_recall"] == scores["micro_f1"] == 0.5
    assert scores["field_weighted_all_field_accuracy"] == 0.5  # 3/6, NOT mean(2/5, 1)
    assert scores["field_weighted_ontology_constrained_accuracy"] == pytest.approx(2 / 3)
    assert scores["field_weighted_non_ontology_constrained_accuracy"] == pytest.approx(1 / 3)
    assert scores["attempted_records"] == 3
    assert scores["completed_records"] == 2
    assert scores["failed_records"] == 1
    results[0].evaluations = []  # SDK can swallow evaluator failures; expose reduced coverage.
    scores = {evaluation.name: evaluation.value for evaluation in run_eval(item_results=results)}
    assert scores["completed_records"] == 1
    assert scores["failed_records"] == 2


def test_strict_case_whole_value_and_existing_doi_normalization(corpus: tuple[Path, Callable]) -> None:
    """Keep the repository's DOI equivalence but introduce no semantic matching."""
    root, _ = corpus
    item_eval, _ = make_evaluators(root / "schemas/rnaseq.json", expected_records=1)
    evaluations = item_eval(
        output={"a": "value", "c": "long VALUE", "protocol_doi": "https://doi.org/10.1/x"},
        expected_output={"a": "VALUE", "c": "VALUE", "protocol_doi": "https://dx.doi.org/10.1/x"},
    )
    scores = {evaluation.name: evaluation.value for evaluation in evaluations}
    assert scores["precision"] == scores["recall"] == scores["all_field_accuracy"] == pytest.approx(1 / 3)


def test_blank_and_empty_category_denominators(corpus: tuple[Path, Callable]) -> None:
    """Both blank earns accuracy, not PRF credit; empty categories score zero."""
    root, _ = corpus
    item_eval, run_eval = make_evaluators(root / "schemas/rnaseq.json", expected_records=1)
    scores = {
        evaluation.name: evaluation.value for evaluation in item_eval(output={"a": None}, expected_output={"a": ""})
    }
    assert scores["all_field_accuracy"] == 1
    assert scores["precision"] == scores["recall"] == scores["f1"] == scores["non_ontology_constrained_accuracy"] == 0
    scores = {evaluation.name: evaluation.value for evaluation in run_eval(item_results=[])}
    assert scores["micro_f1"] == scores["field_weighted_all_field_accuracy"] == 0
    assert scores["failed_records"] == 1
    with pytest.raises(ValueError, match="nonempty"):
        item_eval(output=None, expected_output={"a": "X"})
