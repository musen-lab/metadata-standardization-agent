"""Tests for evaluation metrics (accuracy, per-field results, and precision/recall)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.metrics import (
    CONFUSION_KEYS,
    _is_missing,
    compute_all_field_accuracy,
    compute_field_confusion,
    compute_field_results,
    compute_non_ontology_constrained_field_accuracy,
    compute_ontology_constrained_field_accuracy,
    compute_overall_accuracy,
    precision_recall_f1,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def schema_path(tmp_path: Path) -> Path:
    """A minimal CEDAR-style schema with two ontology fields and two plain fields."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "cell_type", "permissible_values": [{"type": "branch"}]},
            {"name": "title", "permissible_values": []},
            {"name": "count", "permissible_values": []},
        ]
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema))
    return path


class TestComputeAllFieldAccuracy:
    """Tests for the all-field accuracy metric."""

    def test_perfect_match(self) -> None:
        gold = {"a": 1, "b": "hello", "c": [1, 2]}
        predicted = {"a": 1, "b": "hello", "c": [1, 2]}
        assert compute_all_field_accuracy(predicted, gold) == 1.0

    def test_both_empty(self) -> None:
        assert compute_all_field_accuracy({}, {}) == 0.0

    def test_both_null_counts_as_match(self) -> None:
        gold = {"a": 1, "b": None, "c": None}
        predicted = {"a": 1, "b": None, "c": None}
        assert compute_all_field_accuracy(predicted, gold) == 1.0

    def test_value_mismatch(self) -> None:
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1, "b": 99}
        assert compute_all_field_accuracy(predicted, gold) == 0.5

    def test_pred_null_gold_non_null(self) -> None:
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1, "b": None}
        assert compute_all_field_accuracy(predicted, gold) == 0.5

    def test_pred_non_null_gold_null(self) -> None:
        gold = {"a": 1, "b": None}
        predicted = {"a": 1, "b": 2}
        assert compute_all_field_accuracy(predicted, gold) == 0.5

    def test_pred_missing_key_counts_as_mismatch(self) -> None:
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1}
        assert compute_all_field_accuracy(predicted, gold) == 0.5

    def test_extra_predicted_fields_ignored(self) -> None:
        gold = {"a": 1}
        predicted = {"a": 1, "extra": "data"}
        assert compute_all_field_accuracy(predicted, gold) == 1.0


class TestBlankEquivalence:
    """``None`` and ``""`` both mean "no value" and must agree with each other."""

    def test_empty_string_gold_matches_null_prediction(self) -> None:
        gold = {"a": ""}
        predicted = {"a": None}
        assert compute_all_field_accuracy(predicted, gold) == 1.0

    def test_null_gold_matches_empty_string_prediction(self) -> None:
        gold = {"a": None}
        predicted = {"a": ""}
        assert compute_all_field_accuracy(predicted, gold) == 1.0

    def test_empty_string_matches_empty_string(self) -> None:
        gold = {"a": ""}
        predicted = {"a": ""}
        assert compute_all_field_accuracy(predicted, gold) == 1.0

    def test_empty_string_gold_missing_key_matches(self) -> None:
        gold = {"a": ""}
        predicted: dict[str, object] = {}
        assert compute_all_field_accuracy(predicted, gold) == 1.0

    def test_empty_string_gold_vs_real_value_is_mismatch(self) -> None:
        gold = {"a": ""}
        predicted = {"a": "No"}
        assert compute_all_field_accuracy(predicted, gold) == 0.0

    def test_whitespace_only_is_blank(self) -> None:
        assert _is_missing("   ")
        assert _is_missing("\t\n")

    def test_zero_and_false_are_present(self) -> None:
        assert not _is_missing(0)
        assert not _is_missing(False)
        assert not _is_missing([])
        gold = {"a": 0}
        predicted = {"a": None}
        assert compute_all_field_accuracy(predicted, gold) == 0.0


class TestMatchOptions:
    """Tests for match_case and match_whole_word flags."""

    def test_case_insensitive_match(self) -> None:
        gold = {"a": "hello"}
        predicted = {"a": "Hello"}
        assert compute_all_field_accuracy(predicted, gold, match_case=False) == 1.0

    def test_case_sensitive_mismatch(self) -> None:
        gold = {"a": "hello"}
        predicted = {"a": "Hello"}
        assert compute_all_field_accuracy(predicted, gold, match_case=True) == 0.0

    def test_substring_match(self) -> None:
        gold = {"a": "world"}
        predicted = {"a": "hello world"}
        assert compute_all_field_accuracy(predicted, gold, match_whole_word=False) == 1.0

    def test_non_string_ignores_flags(self) -> None:
        gold = {"a": 42}
        predicted = {"a": 42}
        assert compute_all_field_accuracy(predicted, gold, match_case=False, match_whole_word=False) == 1.0

    def test_doi_normalization(self) -> None:
        gold = {"protocol_doi": "https://doi.org/10.1/x"}
        predicted = {"protocol_doi": "https://dx.doi.org/10.1/x"}
        assert compute_all_field_accuracy(predicted, gold) == 1.0


class TestOntologySplit:
    """Tests for ontology- vs non-ontology-constrained accuracy."""

    def test_ontology_only_fields_evaluated(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "cell_type": "T cell", "title": "x", "count": 5}
        predicted = {"tissue": "lung", "cell_type": "WRONG", "title": "y", "count": 9}
        # ontology fields: tissue (match), cell_type (mismatch) -> 1/2
        assert compute_ontology_constrained_field_accuracy(predicted, gold, schema_path) == 0.5

    def test_non_ontology_only_fields_evaluated(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "cell_type": "T cell", "title": "x", "count": 5}
        predicted = {"tissue": "WRONG", "cell_type": "WRONG", "title": "x", "count": 9}
        # non-ontology fields: title (match), count (mismatch) -> 1/2
        assert compute_non_ontology_constrained_field_accuracy(predicted, gold, schema_path) == 0.5

    def test_overall_accuracy_dict(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "cell_type": "T cell", "title": "x", "count": 5}
        predicted = {"tissue": "lung", "cell_type": "T cell", "title": "x", "count": 9}
        result = compute_overall_accuracy(predicted, gold, schema_path)
        assert result["ontology_constrained_accuracy"] == 1.0
        assert result["non_ontology_constrained_accuracy"] == 0.5
        assert result["all_field_accuracy"] == 0.75


class TestComputeFieldResults:
    """Tests for the per-field results helper used by significance tests."""

    def test_returns_one_tuple_per_gold_field(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "cell_type": "T cell", "title": "x", "count": 5}
        predicted = {"tissue": "lung", "cell_type": "WRONG", "title": "x", "count": 5}
        results = compute_field_results(predicted, gold, schema_path)
        assert len(results) == 4

    def test_field_type_tagging(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "title": "x"}
        predicted = {"tissue": "lung", "title": "x"}
        types = {name: ftype for name, ftype, _ in compute_field_results(predicted, gold, schema_path)}
        assert types["tissue"] == "ontology"
        assert types["title"] == "non_ontology"

    def test_correctness_flags(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "cell_type": "T cell"}
        predicted = {"tissue": "lung", "cell_type": "WRONG"}
        correct = {name: ok for name, _t, ok in compute_field_results(predicted, gold, schema_path)}
        assert correct["tissue"] is True
        assert correct["cell_type"] is False

    def test_both_missing_is_correct(self, schema_path: Path) -> None:
        gold = {"title": None}
        predicted = {"title": None}
        results = compute_field_results(predicted, gold, schema_path)
        assert results[0][2] is True


class TestComputeFieldConfusion:
    """Tests for the confusion-matrix classification of gold fields."""

    def test_true_positive(self, schema_path: Path) -> None:
        counts = compute_field_confusion({"title": "a"}, {"title": "a"}, schema_path)
        assert counts["all"]["TP"] == 1
        assert counts["all"]["FP"] == counts["all"]["FN"] == counts["all"]["TN"] == 0

    def test_true_negative(self, schema_path: Path) -> None:
        counts = compute_field_confusion({"title": None}, {"title": None}, schema_path)
        assert counts["all"]["TN"] == 1
        assert counts["all"]["TP"] == counts["all"]["FP"] == counts["all"]["FN"] == 0

    def test_insertion_is_false_positive_only(self, schema_path: Path) -> None:
        counts = compute_field_confusion({"title": "invented"}, {"title": None}, schema_path)
        assert counts["all"]["FP"] == 1
        assert counts["all"]["insertions"] == 1
        assert counts["all"]["FN"] == counts["all"]["substitutions"] == counts["all"]["deletions"] == 0

    def test_deletion_is_false_negative_only(self, schema_path: Path) -> None:
        counts = compute_field_confusion({"title": None}, {"title": "wanted"}, schema_path)
        assert counts["all"]["FN"] == 1
        assert counts["all"]["deletions"] == 1
        assert counts["all"]["FP"] == counts["all"]["substitutions"] == counts["all"]["insertions"] == 0

    def test_substitution_increments_fp_fn_and_sub_only(self, schema_path: Path) -> None:
        counts = compute_field_confusion({"title": "wrong"}, {"title": "right"}, schema_path)
        assert counts["all"]["FP"] == 1
        assert counts["all"]["FN"] == 1
        assert counts["all"]["substitutions"] == 1
        assert counts["all"]["TP"] == counts["all"]["TN"] == 0
        assert counts["all"]["insertions"] == counts["all"]["deletions"] == 0

    def test_empty_string_gold_against_null_prediction_is_true_negative(self, schema_path: Path) -> None:
        """Empty string and null are the same blank, so this is TN rather than an error."""
        counts = compute_field_confusion({"title": None}, {"title": "   "}, schema_path)
        assert counts["all"]["TN"] == 1
        assert counts["all"]["FP"] == counts["all"]["FN"] == 0

    def test_field_absent_from_prediction_with_gold_value_is_deletion(self, schema_path: Path) -> None:
        counts = compute_field_confusion({}, {"title": "wanted"}, schema_path)
        assert counts["all"]["FN"] == 1
        assert counts["all"]["deletions"] == 1

    def test_field_absent_from_prediction_with_blank_gold_is_true_negative(self, schema_path: Path) -> None:
        counts = compute_field_confusion({}, {"title": None}, schema_path)
        assert counts["all"]["TN"] == 1

    def test_field_absent_from_gold_is_ignored(self, schema_path: Path) -> None:
        """Only gold fields are classified, mirroring the accuracy metrics' denominator."""
        counts = compute_field_confusion({"title": "a", "not_in_gold": "x"}, {"title": "a"}, schema_path)
        assert sum(counts["all"].values()) == 1
        assert counts["all"]["TP"] == 1

    def test_categories_split_by_ontology_constraint(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "title": "a"}
        predicted = {"tissue": "liver", "title": "a"}
        counts = compute_field_confusion(predicted, gold, schema_path)
        assert counts["ontology"]["substitutions"] == 1
        assert counts["non_ontology"]["TP"] == 1
        assert counts["all"]["substitutions"] == 1
        assert counts["all"]["TP"] == 1

    def test_all_category_is_the_sum_of_the_other_two(self, schema_path: Path) -> None:
        gold = {"tissue": "lung", "cell_type": None, "title": "a", "count": 3}
        predicted = {"tissue": "liver", "cell_type": "invented", "title": "a", "count": None}
        counts = compute_field_confusion(predicted, gold, schema_path)
        for key in CONFUSION_KEYS:
            assert counts["all"][key] == counts["ontology"][key] + counts["non_ontology"][key]

    def test_empty_gold_yields_zero_counts(self, schema_path: Path) -> None:
        counts = compute_field_confusion({}, {}, schema_path)
        assert all(value == 0 for value in counts["all"].values())

    def test_invariants_hold(self, schema_path: Path) -> None:
        """The seven invariants from docs/precision-recall-plan.md, on one record."""
        gold = {"tissue": "lung", "cell_type": None, "title": "wanted", "count": 3}
        predicted = {"tissue": "liver", "cell_type": "invented", "title": None, "count": 3}
        counts = compute_field_confusion(predicted, gold, schema_path)["all"]
        blank_gold = sum(1 for value in gold.values() if _is_missing(value))
        non_blank_gold = len(gold) - blank_gold

        assert counts["FP"] == counts["insertions"] + counts["substitutions"]
        assert counts["FN"] == counts["deletions"] + counts["substitutions"]
        assert counts["TP"] + counts["FN"] == non_blank_gold
        assert counts["TN"] + counts["insertions"] == blank_gold
        assert counts["TP"] + counts["FP"] + counts["FN"] + counts["TN"] - counts["substitutions"] == len(gold)
        assert counts["insertions"] + counts["deletions"] + counts["substitutions"] == 3


class TestPrecisionRecallF1:
    """Tests for the precision/recall/F1 arithmetic."""

    def test_hand_computed_values(self) -> None:
        scores = precision_recall_f1({"TP": 6, "FP": 2, "FN": 3})
        assert scores["precision"] == pytest.approx(0.75)
        assert scores["recall"] == pytest.approx(2 / 3)
        assert scores["f1"] == pytest.approx(2 * 0.75 * (2 / 3) / (0.75 + 2 / 3))

    def test_perfect_scores(self) -> None:
        scores = precision_recall_f1({"TP": 5, "FP": 0, "FN": 0})
        assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_nothing_asserted_scores_zero_not_raises(self) -> None:
        """An all-null prediction has no true or false positives at all."""
        scores = precision_recall_f1({"TP": 0, "FP": 0, "FN": 7})
        assert scores == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_all_counts_zero_scores_zero(self) -> None:
        assert precision_recall_f1({"TP": 0, "FP": 0, "FN": 0}) == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    def test_f1_is_zero_when_either_side_is_zero(self) -> None:
        assert precision_recall_f1({"TP": 0, "FP": 4, "FN": 0})["f1"] == 0.0
        assert precision_recall_f1({"TP": 0, "FP": 0, "FN": 4})["f1"] == 0.0

    def test_missing_keys_default_to_zero(self) -> None:
        assert precision_recall_f1({"TP": 3})["precision"] == 1.0
