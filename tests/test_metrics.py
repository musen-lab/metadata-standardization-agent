"""Tests for evaluation metrics (completeness and accuracy)."""

from __future__ import annotations

from evaluation.metrics import compute_accuracy, compute_completeness, compute_correctness


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


class TestComputeCorrectness:
    """Tests for the field-value correctness metric."""

    def test_both_empty(self) -> None:
        """Both dicts empty returns 0.0."""
        assert compute_correctness({}, {}) == 0.0

    def test_no_non_missing_gold_fields_all_none(self) -> None:
        """All gold values are None; no non-missing gold fields yields 0.0."""
        gold = {"a": None, "b": None}
        predicted = {"a": 1, "b": 2}
        assert compute_correctness(predicted, gold) == 0.0

    def test_all_predicted_none(self) -> None:
        """All predicted values are None; no matches yields 0.0."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": None, "b": None}
        assert compute_correctness(predicted, gold) == 0.0

    def test_all_non_missing_gold_fields_match(self) -> None:
        """All non-missing gold fields match gives 1.0."""
        gold = {"a": 1, "b": "hello"}
        predicted = {"a": 1, "b": "hello"}
        assert compute_correctness(predicted, gold) == 1.0

    def test_some_matches_some_mismatches(self) -> None:
        """Mixed matches and mismatches give correct fraction."""
        gold = {"a": 1, "b": 2, "c": 3, "d": 4}
        predicted = {"a": 1, "b": 2, "c": 99, "d": 0}
        # 4 non-missing gold fields, 2 match → 0.5
        assert compute_correctness(predicted, gold) == 0.5

    def test_none_in_gold_excluded_from_denominator(self) -> None:
        """Fields with None in gold are excluded from the denominator."""
        gold = {"a": 1, "b": None, "c": 3}
        predicted = {"a": 1, "b": 2, "c": None}
        # Non-missing gold: {a, c} → predicted matches a but misses c → 1/2
        assert compute_correctness(predicted, gold) == 0.5

    def test_fields_only_in_predicted_not_counted(self) -> None:
        """Fields present only in predicted (not in gold) are ignored."""
        gold = {"a": 1}
        predicted = {"a": 1, "extra": "data"}
        assert compute_correctness(predicted, gold) == 1.0

    def test_fields_only_in_gold_penalise_prediction(self) -> None:
        """Fields in gold but absent from predicted count against the score."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1}
        # Non-missing gold: {a, b} → predicted matches a, misses b → 1/2
        assert compute_correctness(predicted, gold) == 0.5

    def test_empty_string_mismatch(self) -> None:
        """Empty string does not equal a non-empty string."""
        gold = {"a": "hello"}
        predicted = {"a": ""}
        assert compute_correctness(predicted, gold) == 0.0


class TestComputeCorrectnessMatchOptions:
    """Tests for match_case and match_whole_word parameters of compute_correctness."""

    def test_case_insensitive_match(self) -> None:
        """Differing case matches when match_case=False."""
        gold = {"a": "hello"}
        predicted = {"a": "Hello"}
        assert compute_correctness(predicted, gold, match_case=False) == 1.0

    def test_case_sensitive_mismatch(self) -> None:
        """Differing case does NOT match when match_case=True."""
        gold = {"a": "hello"}
        predicted = {"a": "Hello"}
        assert compute_correctness(predicted, gold, match_case=True) == 0.0

    def test_substring_match(self) -> None:
        """Gold value as substring of predicted matches when match_whole_word=False."""
        gold = {"a": "world"}
        predicted = {"a": "hello world"}
        assert compute_correctness(predicted, gold, match_whole_word=False) == 1.0

    def test_substring_direction(self) -> None:
        """Gold value longer than predicted does NOT match with match_whole_word=False."""
        gold = {"a": "hello world"}
        predicted = {"a": "world"}
        assert compute_correctness(predicted, gold, match_whole_word=False) == 0.0

    def test_combined_case_insensitive_substring(self) -> None:
        """Case-insensitive substring match works when both flags are relaxed."""
        gold = {"a": "World"}
        predicted = {"a": "hello world"}
        assert compute_correctness(predicted, gold, match_case=False, match_whole_word=False) == 1.0

    def test_non_string_ignores_flags(self) -> None:
        """Non-string values use exact match regardless of flags."""
        gold = {"a": 42}
        predicted = {"a": 42}
        assert compute_correctness(predicted, gold, match_case=False, match_whole_word=False) == 1.0

        predicted_mismatch = {"a": 43}
        assert compute_correctness(predicted_mismatch, gold, match_case=False, match_whole_word=False) == 0.0

    def test_defaults_unchanged(self) -> None:
        """Default parameters behave as exact match (backward-compatible)."""
        gold = {"a": "Hello", "b": "world"}
        predicted = {"a": "Hello", "b": "World"}
        # 'a' matches exactly, 'b' differs in case → 1/2
        assert compute_correctness(predicted, gold) == 0.5


class TestComputeAccuracy:
    """Tests for the accuracy metric."""

    def test_perfect_match(self) -> None:
        """All fields identical yields 1.0."""
        gold = {"a": 1, "b": "hello", "c": [1, 2]}
        predicted = {"a": 1, "b": "hello", "c": [1, 2]}
        assert compute_accuracy(predicted, gold) == 1.0

    def test_both_empty(self) -> None:
        """Both dicts empty returns 0.0."""
        assert compute_accuracy({}, {}) == 0.0

    def test_both_null_counts_as_match(self) -> None:
        """Fields that are None in both predicted and gold count as matches."""
        gold = {"a": 1, "b": None, "c": None}
        predicted = {"a": 1, "b": None, "c": None}
        assert compute_accuracy(predicted, gold) == 1.0

    def test_value_mismatch(self) -> None:
        """Value mismatches reduce the score."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1, "b": 99}
        assert compute_accuracy(predicted, gold) == 0.5

    def test_pred_null_gold_non_null(self) -> None:
        """Predicted null vs gold non-null counts as mismatch."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1, "b": None}
        assert compute_accuracy(predicted, gold) == 0.5

    def test_pred_non_null_gold_null(self) -> None:
        """Predicted non-null vs gold null counts as mismatch."""
        gold = {"a": 1, "b": None}
        predicted = {"a": 1, "b": 2}
        assert compute_accuracy(predicted, gold) == 0.5

    def test_mixed_scenario(self) -> None:
        """Mixed scenario: match, both-null match, value mismatch, presence mismatch."""
        gold = {"a": "x", "b": None, "c": "y", "d": "z"}
        predicted = {"a": "x", "b": None, "c": "wrong", "d": None}
        # a: match, b: both null (match), c: mismatch, d: presence mismatch → 2/4
        assert compute_accuracy(predicted, gold) == 0.5

    def test_pred_missing_key(self) -> None:
        """Key in gold but absent from predicted (get returns None) counts as presence mismatch."""
        gold = {"a": 1, "b": 2}
        predicted = {"a": 1}
        # 'b': predicted.get("b") is None vs gold 2 → mismatch → 1/2
        assert compute_accuracy(predicted, gold) == 0.5
