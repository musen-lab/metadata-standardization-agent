"""Tests for evaluation metrics (accuracy and per-field results)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from evaluation.metrics import (
    _get_required_fields,
    compute_all_field_accuracy,
    compute_field_results,
    compute_non_ontology_constrained_field_accuracy,
    compute_ontology_constrained_field_accuracy,
    compute_overall_accuracy,
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


class TestGetRequiredFields:
    """Tests for reading the schema's required flag."""

    def test_returns_only_required_true(self, tmp_path: Path) -> None:
        schema = {
            "children": [
                {"name": "a", "required": True},
                {"name": "b", "required": False},
                {"name": "c"},  # no required key -> not required
            ]
        }
        path = tmp_path / "s.json"
        path.write_text(json.dumps(schema))
        assert _get_required_fields(path) == ["a"]
