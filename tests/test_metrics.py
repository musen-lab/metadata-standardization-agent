"""Tests for evaluation metrics (completeness and accuracy)."""

from __future__ import annotations

from evaluation.metrics import compute_accuracy, compute_completeness


class TestComputeCompleteness:
    """Tests for the field completeness metric."""

    def test_both_empty(self) -> None:
        """Both dicts empty returns 0.0."""
        assert compute_completeness({}, {}) == 0.0

    def test_gold_empty(self) -> None:
        """Gold with no fields returns 0.0 (no denominator)."""
        assert compute_completeness({"a": 1}, {}) == 0.0

    def test_perfect_overlap(self) -> None:
        """All gold fields present in predicted yields 1.0."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1, "b": 2}
        assert compute_completeness(predicted, gold) == 1.0

    def test_partial_overlap(self) -> None:
        """Some gold fields missing in predicted gives correct fraction."""
        gold = {"a": 1, "b": 2, "c": 3, "d": 4}
        predicted = {"a": 1, "c": 99}  # 2 of 4 gold keys present
        assert compute_completeness(predicted, gold) == 0.5

    def test_predicted_has_extra_fields(self) -> None:
        """Extra fields in predicted do not affect completeness."""
        gold = {"a": 1}
        predicted = {"a": 1, "extra": "stuff"}
        assert compute_completeness(predicted, gold) == 1.0

    def test_different_value_still_counts_as_complete(self) -> None:
        """Presence is enough; value mismatch still counts as complete."""
        gold = {"a": 1}
        predicted = {"a": 999}
        assert compute_completeness(predicted, gold) == 1.0

    def test_none_in_gold_excluded_from_denominator(self) -> None:
        """Gold fields with None values are excluded from the denominator."""
        gold = {"a": 1, "b": None, "c": None}
        predicted = {"a": 1}
        # Only 'a' is non-missing in gold → 1/1
        assert compute_completeness(predicted, gold) == 1.0

    def test_none_in_predicted_counts_as_missing(self) -> None:
        """Predicted field set to None is not considered present."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1, "b": None}
        assert compute_completeness(predicted, gold) == 0.5

    def test_empty_string_is_present(self) -> None:
        """Empty string is a valid value, not missing."""
        gold = {"a": "hello"}
        predicted = {"a": ""}
        assert compute_completeness(predicted, gold) == 1.0

    def test_empty_list_is_present(self) -> None:
        """Empty list is a valid value, not missing."""
        gold = {"a": [1, 2]}
        predicted = {"a": []}
        assert compute_completeness(predicted, gold) == 1.0


class TestComputeAccuracy:
    """Tests for the field-value accuracy metric."""

    def test_both_empty(self) -> None:
        """Both dicts empty returns 0.0."""
        assert compute_accuracy({}, {}) == 0.0

    def test_no_comparable_fields_all_none_in_gold(self) -> None:
        """All gold values are None; no comparable fields yields 0.0."""
        gold = {"a": None, "b": None}
        predicted = {"a": 1, "b": 2}
        assert compute_accuracy(predicted, gold) == 0.0

    def test_no_comparable_fields_all_none_in_predicted(self) -> None:
        """All predicted values are None; no comparable fields yields 0.0."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": None, "b": None}
        assert compute_accuracy(predicted, gold) == 0.0

    def test_all_comparable_match(self) -> None:
        """All comparable fields match gives 1.0."""
        gold = {"a": 1, "b": "hello"}
        predicted = {"a": 1, "b": "hello"}
        assert compute_accuracy(predicted, gold) == 1.0

    def test_some_matches_some_mismatches(self) -> None:
        """Mixed matches and mismatches give correct fraction."""
        gold = {"a": 1, "b": 2, "c": 3, "d": 4}
        predicted = {"a": 1, "b": 2, "c": 99, "d": 0}
        # 4 comparable, 2 match → 0.5
        assert compute_accuracy(predicted, gold) == 0.5

    def test_none_excluded_from_comparable(self) -> None:
        """Fields with None on either side are excluded from comparison."""
        gold = {"a": 1, "b": None, "c": 3}
        predicted = {"a": 1, "b": 2, "c": None}
        # Only 'a' is comparable → 1/1
        assert compute_accuracy(predicted, gold) == 1.0

    def test_fields_only_in_predicted_not_counted(self) -> None:
        """Fields present only in predicted (not in gold) are ignored."""
        gold = {"a": 1}
        predicted = {"a": 1, "extra": "data"}
        assert compute_accuracy(predicted, gold) == 1.0

    def test_fields_only_in_gold_not_counted(self) -> None:
        """Fields in gold but absent from predicted are not comparable."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1}
        # 'b' is missing from predicted (get returns None) → excluded
        # Only 'a' is comparable → 1/1
        assert compute_accuracy(predicted, gold) == 1.0

    def test_empty_string_mismatch(self) -> None:
        """Empty string does not equal a non-empty string."""
        gold = {"a": "hello"}
        predicted = {"a": ""}
        assert compute_accuracy(predicted, gold) == 0.0
