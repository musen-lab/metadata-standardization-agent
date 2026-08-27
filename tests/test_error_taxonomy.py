"""Tests for the error taxonomy.

Two load-bearing properties.  Reconciliation: every false positive and false negative the
precision/recall tables count must be recoverable from the labelled rows, since a taxonomy
that silently dropped errors would make every share it reports wrong.  And partition: each
error carries exactly one sub-category, whose category is the confusion case, so the two
levels cannot drift apart.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import (
    CATEGORIES,
    CATEGORY_BY_SUBCATEGORY,
    CONFUSION_CELLS_BY_CATEGORY,
    SUBCATEGORIES,
    category_shares,
    close_match_reasons,
    collect_field_errors,
    deduplicate_errors,
    reconcile_with_confusion,
    summarize_error_categories,
    summarize_error_subcategories,
)
from analysis.data_analysis.error_taxonomy import (
    CLOSE_MATCH,
    COMPLETELY_WRONG,
    DELETIONS,
    ENTIRELY_DONT_KNOW,
    INSERTIONS,
    KEPT_THIS_FIELD_S_VALUE,
    MISLOCATE_LEGACY_VALUE,
    ONE_CONTAINS_THE_OTHER,
    OVERESTIMATE_LEGACY_VALUE,
    POOLED_ASSAY,
    SAME_VALUE_OTHER_SHAPE,
    SUBSTITUTIONS,
    TOO_OPTIMISTIC_ANSWER,
    UNDERESTIMATE_LEGACY_VALUE,
    USED_THE_VOCABULARY,
    _check_levels_agree,
)

if TYPE_CHECKING:
    from pathlib import Path

#: ``ms_scan_mode`` enumerates its permissible values and the others do not, so a fixture
#: can reach the vocabulary test or avoid it by choosing a field.
SCHEMA = {
    "children": [
        {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
        {"name": "title", "permissible_values": []},
        {"name": "count", "permissible_values": []},
        {"name": "ms_scan_mode", "permissible_values": [{"type": "branch", "options": ["MS1", "MS2"]}]},
    ]
}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _case(root: Path, name: str, *, gold: dict, predicted: dict, legacy: dict) -> None:
    """One atacseq record, with its legacy input and one run's prediction."""
    _write(root / "schemas" / "atacseq.json", SCHEMA)
    _write(root / "atacseq" / "gold" / f"{name}.json", gold)
    _write(root / "atacseq" / "input" / f"{name}.json", legacy)
    _write(root / "atacseq" / "output" / "m" / "sys" / f"{name}.json", predicted)


def _one(root: Path) -> dict:
    """The single labelled row for a one-error fixture."""
    errors = collect_field_errors(str(root), "m", "sys")
    assert len(errors) == 1, errors.to_string()
    return errors.iloc[0].to_dict()


class TestSubstitutions:
    def test_close_match_when_only_the_shape_differs(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": "Lung Biopsy"}, predicted={"title": "lung  biopsy "}, legacy={})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (SUBSTITUTIONS, CLOSE_MATCH)
        assert SAME_VALUE_OTHER_SHAPE in row["close_match_reasons"]

    def test_a_number_in_another_shape_is_close(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"count": 10}, predicted={"count": "10.0"}, legacy={})
        assert SAME_VALUE_OTHER_SHAPE in _one(tmp_path)["close_match_reasons"]

    def test_close_match_when_the_run_said_gold_s_value_plus_extra(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": "Orbitrap Fusion"}, predicted={"title": "Orbitrap Fusion Lumos"}, legacy={})
        row = _one(tmp_path)
        assert row["subcategory"] == CLOSE_MATCH
        assert ONE_CONTAINS_THE_OTHER in row["close_match_reasons"]

    def test_close_match_when_the_run_kept_this_field_s_record_value(self, tmp_path: Path) -> None:
        # The record said "lungs", gold corrected it, the run kept the record's word for it.
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "lungs"}, legacy={"tissue": "lungs"})
        row = _one(tmp_path)
        assert row["subcategory"] == CLOSE_MATCH
        assert KEPT_THIS_FIELD_S_VALUE in row["close_match_reasons"]

    def test_close_match_when_the_run_used_the_vocabulary_and_gold_did_not(self, tmp_path: Path) -> None:
        # The curator kept the record's "MS", which the template does not permit; the run
        # answered "MS1", which it does.  A disagreement about normalising, not about fact.
        _case(
            tmp_path,
            "r",
            gold={"ms_scan_mode": "MS"},
            predicted={"ms_scan_mode": "MS1"},
            legacy={"ms_scan_mode": "MS"},
        )
        row = _one(tmp_path)
        assert row["subcategory"] == CLOSE_MATCH
        assert USED_THE_VOCABULARY in row["close_match_reasons"]

    def test_the_vocabulary_reason_needs_the_run_to_be_permissible_too(self, tmp_path: Path) -> None:
        # Gold being off-vocabulary is not on its own a reason to call the run close: the
        # run has to have picked a term the template allows.
        _case(
            tmp_path,
            "r",
            gold={"ms_scan_mode": "MS"},
            predicted={"ms_scan_mode": "nonsense"},
            legacy={"ms_scan_mode": "MS"},
        )
        assert USED_THE_VOCABULARY not in _one(tmp_path)["close_match_reasons"]

    def test_mislocated_when_the_value_came_from_another_field(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"tissue": "lung"},
            predicted={"tissue": "SN123"},
            legacy={"tissue": "lung tissue", "title": "SN123"},
        )
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (SUBSTITUTIONS, MISLOCATE_LEGACY_VALUE)

    def test_close_match_wins_over_mislocated(self, tmp_path: Path) -> None:
        # The asserted value equals gold's once shape is relaxed *and* sits in another
        # field.  Calling it a mislocation would report where a right answer came from.
        _case(tmp_path, "r", gold={"tissue": "Lung"}, predicted={"tissue": "lung"}, legacy={"title": "lung"})
        assert _one(tmp_path)["subcategory"] == CLOSE_MATCH

    def test_completely_wrong_when_neither_near_gold_nor_in_the_record(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={"title": "unrelated"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (SUBSTITUTIONS, COMPLETELY_WRONG)

    def test_a_value_that_loosens_away_to_nothing_is_carried_by_no_field(self, tmp_path: Path) -> None:
        # "." and "/" both loosen to the empty string, so the record is not the source of
        # the assertion -- but the two are equal once loosened, so the row is close.
        _case(tmp_path, "r", gold={"title": "/"}, predicted={"title": "."}, legacy={"title": "/"})
        assert _one(tmp_path)["subcategory"] == CLOSE_MATCH


class TestDeletionsAndInsertions:
    def test_underestimated_when_the_record_held_gold_s_value(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "lung"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (DELETIONS, UNDERESTIMATE_LEGACY_VALUE)

    def test_dont_know_when_the_record_held_nothing(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "kidney"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (DELETIONS, ENTIRELY_DONT_KNOW)

    def test_overestimated_when_the_run_wrote_a_value_the_record_holds(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": ""}, predicted={"title": "lung"}, legacy={"tissue": "lung"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (INSERTIONS, OVERESTIMATE_LEGACY_VALUE)

    def test_too_optimistic_when_the_record_holds_nothing(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": ""}, predicted={"title": "from nowhere"}, legacy={"tissue": "lung"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (INSERTIONS, TOO_OPTIMISTIC_ANSWER)

    def test_the_two_pairs_mirror_each_other(self, tmp_path: Path) -> None:
        # One question -- did the record hold the value? -- asked of gold's for a deletion
        # and of the run's for an insertion.  The mirror is the point of the names.
        _case(tmp_path, "a", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "lung"})
        _case(tmp_path, "b", gold={"title": ""}, predicted={"title": "lung"}, legacy={"organ": "lung"})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert set(errors["subcategory"]) == {UNDERESTIMATE_LEGACY_VALUE, OVERESTIMATE_LEGACY_VALUE}


class TestPartition:
    def _mixed(self, tmp_path: Path):
        _case(
            tmp_path,
            "mixed",
            gold={"tissue": "lung", "title": "", "count": 5, "ms_scan_mode": "MS1"},
            predicted={"tissue": "kidney", "title": "spurious", "count": None, "ms_scan_mode": "MS1"},
            legacy={},
        )
        return collect_field_errors(str(tmp_path), "m", "sys")

    def test_one_row_per_error(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        assert len(errors) == errors["pointer"].nunique() == 3  # substitution + insertion + deletion

    def test_every_counted_error_is_recovered(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        counts = reconcile_with_confusion(errors, str(tmp_path), "m", "sys")
        assert counts["FP"]["counted"] == counts["FP"]["categorised"] == 2  # substitution + insertion
        assert counts["FN"]["counted"] == counts["FN"]["categorised"] == 2  # substitution + deletion

    def test_the_category_is_the_confusion_case(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        expected = {"substitution": SUBSTITUTIONS, "deletion": DELETIONS, "insertion": INSERTIONS}
        assert [expected[case] for case in errors["case"]] == list(errors["category"])

    def test_costs_follows_the_case(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path).set_index("case")
        assert errors.loc["insertion", "costs"] == "precision"
        assert errors.loc["deletion", "costs"] == "recall"
        assert errors.loc["substitution", "costs"] == "precision and recall"

    def test_the_confusion_cells_match_what_each_category_costs(self) -> None:
        # The figure braces the categories with these labels, so a category naming the wrong
        # cells would contradict the reconciliation counted right beside it.
        assert CONFUSION_CELLS_BY_CATEGORY == {
            SUBSTITUTIONS: "FP + FN",
            DELETIONS: "FN",
            INSERTIONS: "FP",
        }

    def test_the_cells_agree_with_the_costs_column(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        in_words = {"precision and recall": "FP + FN", "recall": "FN", "precision": "FP"}
        for _index, row in errors.iterrows():
            assert CONFUSION_CELLS_BY_CATEGORY[row["category"]] == in_words[row["costs"]]

    def test_every_subcategory_has_a_category(self) -> None:
        assert set(SUBCATEGORIES) == set(CATEGORY_BY_SUBCATEGORY)
        assert set(CATEGORY_BY_SUBCATEGORY.values()) == set(CATEGORIES)

    def test_the_subcategory_order_nests_inside_the_category_order(self) -> None:
        # The two tables print one above the other, so the finer order has to run in the
        # coarser one's order for the roll-up to be checkable by eye.
        walked = [CATEGORY_BY_SUBCATEGORY[name] for name in SUBCATEGORIES]
        assert walked == sorted(walked, key=CATEGORIES.index)

    def test_levels_that_contradict_each_other_are_refused(self, tmp_path: Path) -> None:
        # The category comes from the case and the sub-category from the classifier, by two
        # routes that could drift apart without anything raising.
        errors = self._mixed(tmp_path)
        errors.loc[errors.index[0], "category"] = DELETIONS
        errors.loc[errors.index[0], "subcategory"] = CLOSE_MATCH
        with pytest.raises(ValueError, match="wrong category"):
            _check_levels_agree(errors)

    def test_matches_and_blanks_produce_no_rows(self, tmp_path: Path) -> None:
        _case(
            tmp_path, "r", gold={"tissue": "lung", "title": ""}, predicted={"tissue": "lung", "title": None}, legacy={}
        )
        assert collect_field_errors(str(tmp_path), "m", "sys").empty


class TestCloseMatchReasons:
    def test_every_reason_is_reported_not_only_the_first(self) -> None:
        # The four overlap heavily, and a break-down keeping only the first would describe
        # the order they are tested in as much as the run.
        reasons = close_match_reasons("Orbitrap", "Orbitrap", "tissue", {"tissue": "Orbitrap"}, None)
        assert SAME_VALUE_OTHER_SHAPE in reasons
        assert ONE_CONTAINS_THE_OTHER in reasons
        assert KEPT_THIS_FIELD_S_VALUE in reasons

    def test_no_reasons_means_not_a_close_match(self) -> None:
        assert close_match_reasons("lung", "kidney", "tissue", {}, None) == []

    def test_rows_that_are_not_close_carry_no_reasons(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={})
        assert _one(tmp_path)["close_match_reasons"] == ""


class TestBookkeeping:
    def test_pointer_locates_the_field(self, tmp_path: Path) -> None:
        _case(tmp_path, "rec01", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert set(errors["pointer"]) == {"atacseq/rec01#tissue"}

    def test_the_run_s_own_reasoning_is_attached_when_kept(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        _write(
            tmp_path / "atacseq" / "output" / "m" / "sys" / "decisions" / "r.json",
            [{"key": "tissue", "resolution": "harmonized", "reasoning": "picked the nearest term"}],
        )
        row = _one(tmp_path)
        assert row["resolution"] == "harmonized"
        assert row["reasoning"] == "picked the nearest term"

    def test_no_decision_log_leaves_those_columns_blank(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        row = _one(tmp_path)
        assert row["resolution"] is None
        assert row["reasoning"] is None

    def test_summary_shares_sum_to_one(self, tmp_path: Path) -> None:
        for index in range(4):
            _case(
                tmp_path,
                f"r{index}",
                gold={"tissue": "lung"},
                predicted={"tissue": "kidney" if index else None},
                legacy={},
            )
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        for table in (summarize_error_categories(errors), summarize_error_subcategories(errors)):
            assert abs(table["share"].sum() - 1.0) < 1e-9

    def test_a_run_without_predictions_gives_an_empty_frame(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        assert collect_field_errors(str(tmp_path), "m", "absent").empty

    def test_a_prediction_without_its_legacy_input_is_refused(self, tmp_path: Path) -> None:
        # Not a partial corpus -- the run read that input to write the prediction.  Left to
        # default to an empty record, every provenance test quietly answers "no".
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={"tissue": "lung"})
        (tmp_path / "atacseq" / "input" / "r.json").unlink()
        with pytest.raises(FileNotFoundError, match="no legacy record"):
            collect_field_errors(str(tmp_path), "m", "sys")


class TestDeduplication:
    def _repetitive(self, tmp_path: Path):
        # The same mistake in three records, and a different one in a fourth.
        for index in range(3):
            _case(
                tmp_path,
                f"same{index}",
                gold={"tissue": "lung"},
                predicted={"tissue": "kidney"},
                legacy={"title": "unrelated"},
            )
        _case(tmp_path, "other", gold={"tissue": "lung"}, predicted={"tissue": "spleen"}, legacy={})
        return collect_field_errors(str(tmp_path), "m", "sys")

    def test_a_repeated_mistake_collapses_to_one_row(self, tmp_path: Path) -> None:
        errors = self._repetitive(tmp_path)
        assert len(errors) == 4
        assert len(deduplicate_errors(errors)) == 2

    def test_the_instances_are_counted_not_discarded(self, tmp_path: Path) -> None:
        errors = self._repetitive(tmp_path)
        distinct = deduplicate_errors(errors)
        assert distinct["n_instances"].sum() == len(errors)
        assert sorted(distinct["n_instances"]) == [1, 3]

    def test_the_two_levels_still_agree_afterwards(self, tmp_path: Path) -> None:
        _check_levels_agree(deduplicate_errors(self._repetitive(tmp_path)))

    def test_the_same_values_labelled_differently_stay_apart(self, tmp_path: Path) -> None:
        # Same gold and same asserted value, but one record's legacy holds gold's value and
        # the other's does not, so the taxonomy calls them different things -- which makes
        # them different mistakes, and the sub-category has to be part of the key.
        _case(tmp_path, "a", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "lung"})
        _case(tmp_path, "b", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert set(errors["subcategory"]) == {UNDERESTIMATE_LEGACY_VALUE, ENTIRELY_DONT_KNOW}
        assert len(deduplicate_errors(errors)) == 2

    def test_an_unhashable_value_does_not_raise(self, tmp_path: Path) -> None:
        # Gold and prediction hold JSON of any type, and a list would make a plain groupby
        # raise rather than group.
        _case(tmp_path, "r", gold={"tissue": ["lung", "left"]}, predicted={"tissue": ["lung"]}, legacy={})
        assert len(deduplicate_errors(collect_field_errors(str(tmp_path), "m", "sys"))) == 1

    def test_an_empty_frame_comes_back_empty(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={})
        distinct = deduplicate_errors(collect_field_errors(str(tmp_path), "m", "sys"))
        assert distinct.empty
        assert "n_instances" in distinct.columns


class TestCategoryShares:
    def _corpus(self, tmp_path: Path):
        for index in range(3):
            _case(
                tmp_path,
                f"r{index}",
                gold={"tissue": "lung", "title": "", "count": 5},
                predicted={"tissue": "kidney", "title": "x", "count": None},
                legacy={},
            )
        return collect_field_errors(str(tmp_path), "m", "sys")

    def test_shares_sum_to_one_within_each_assay(self, tmp_path: Path) -> None:
        shares = category_shares(self._corpus(tmp_path))
        for _assay, frame in shares.groupby("assay", observed=True):
            assert abs(frame["share"].sum() - 1.0) < 1e-9

    def test_the_pooled_row_counts_every_error(self, tmp_path: Path) -> None:
        errors = self._corpus(tmp_path)
        shares = category_shares(errors)
        pooled = shares[shares["assay"] == POOLED_ASSAY]
        assert pooled["n"].sum() == len(errors)

    def test_a_category_with_no_errors_is_kept_as_a_zero(self, tmp_path: Path) -> None:
        # A figure draws one segment per category; one dropped for being empty in an assay
        # and present in another would silently reorder the stack between rows.
        shares = category_shares(self._corpus(tmp_path))
        pooled = shares[shares["assay"] == POOLED_ASSAY]
        assert list(pooled["category"]) == list(CATEGORIES)
