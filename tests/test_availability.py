"""Tests for the availability split: was gold's value already written in the input?

The load-bearing properties are that the four groups partition recall's denominator, and
that a group is pooled over values rather than over assays -- the assays differ in size by
more than tenfold, so averaging their rates would answer a different question.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import (
    DERIVED,
    VERBATIM,
    create_availability_summary,
    pool_availability,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = {
    "children": [
        {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
        {"name": "title", "permissible_values": []},
    ]
}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _case(root: Path, assay: str, name: str, *, gold: dict, predicted: dict, legacy: dict) -> None:
    """One record of *assay*, with its legacy input and one run's prediction."""
    _write(root / "schemas" / f"{assay}.json", SCHEMA)
    _write(root / assay / "gold" / f"{name}.json", gold)
    _write(root / assay / "input" / f"{name}.json", legacy)
    _write(root / assay / "output" / "m" / "sys" / f"{name}.json", predicted)


def _row(root: Path):
    """The single summary row for a one-field fixture."""
    summary = create_availability_summary(str(root), "m", "sys")
    assert len(summary) == 1, summary.to_string()
    return summary.iloc[0]


class TestAvailability:
    def test_a_value_the_record_carries_under_another_name_is_verbatim(self, tmp_path: Path) -> None:
        _case(tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={"organ": "lung"})
        assert _row(tmp_path)["availability"] == VERBATIM

    def test_a_value_the_record_does_not_carry_is_derived(self, tmp_path: Path) -> None:
        _case(
            tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={"organ": "kidney"}
        )
        assert _row(tmp_path)["availability"] == DERIVED

    def test_a_normalised_term_is_derived_not_verbatim(self, tmp_path: Path) -> None:
        # The record said "lungs" and gold asks for "lung": the value is *in* the record in
        # the loose sense a reader means, and not in the strict sense this split measures.
        # Which is why the split is difficulty and not recoverability -- the run gets these.
        _case(
            tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={"tissue": "lungs"}
        )
        row = _row(tmp_path)
        assert row["availability"] == DERIVED
        assert row["correct_rate"] == 1.0

    def test_the_ontology_half_of_the_schema_is_separated(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "atacseq",
            "r",
            gold={"tissue": "lung", "title": "a biopsy"},
            predicted={"tissue": "lung", "title": "a biopsy"},
            legacy={},
        )
        summary = create_availability_summary(str(tmp_path), "m", "sys")
        assert set(summary["field_type"]) == {"ontology", "non_ontology"}


class TestBookkeeping:
    def test_a_blank_gold_field_asks_for_nothing_and_is_not_counted(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "atacseq",
            "r",
            gold={"tissue": "lung", "title": ""},
            predicted={"tissue": "lung", "title": "spurious"},
            legacy={},
        )
        summary = create_availability_summary(str(tmp_path), "m", "sys")
        assert summary["n_gold_values"].sum() == 1

    def test_the_groups_partition_recalls_denominator(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "atacseq",
            "r",
            gold={"tissue": "lung", "title": "a biopsy"},
            predicted={"tissue": "kidney", "title": None},
            legacy={"tissue": "lung"},
        )
        summary = create_availability_summary(str(tmp_path), "m", "sys")
        # Two non-blank gold fields: one wrong, one blank, so neither is correct.
        assert summary["n_gold_values"].sum() == 2
        assert summary["n_correct"].sum() == 0

    def test_the_rate_counts_only_matches(self, tmp_path: Path) -> None:
        for index in range(4):
            _case(
                tmp_path,
                "atacseq",
                f"r{index}",
                gold={"tissue": "lung"},
                predicted={"tissue": "lung" if index else "kidney"},
                legacy={},
            )
        row = _row(tmp_path)
        assert row["n_gold_values"] == 4
        assert row["n_correct"] == 3
        assert row["correct_rate"] == 0.75

    def test_pooling_weights_by_values_not_by_assay(self, tmp_path: Path) -> None:
        # A big assay that is always right and a small one that is always wrong.  Averaging
        # the two rates would give 0.5; pooling over values gives the corpus's rate.
        for index in range(9):
            _case(
                tmp_path,
                "atacseq",
                f"r{index}",
                gold={"tissue": "lung"},
                predicted={"tissue": "lung"},
                legacy={},
            )
        _case(tmp_path, "rnaseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})

        summary = create_availability_summary(str(tmp_path), "m", "sys")
        assert sorted(summary["correct_rate"]) == [0.0, 1.0]
        pooled = pool_availability(summary, "ontology", DERIVED)
        assert pooled["n_gold_values"] == 10
        assert pooled["correct_rate"] == 0.9

    def test_an_empty_group_pools_to_zero_rather_than_nan(self, tmp_path: Path) -> None:
        _case(tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={})
        summary = create_availability_summary(str(tmp_path), "m", "sys")
        pooled = pool_availability(summary, "ontology", VERBATIM)
        assert pooled == {"n_gold_values": 0, "n_correct": 0, "correct_rate": 0.0}

    def test_a_run_without_predictions_gives_an_empty_frame(self, tmp_path: Path) -> None:
        _case(tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={})
        assert create_availability_summary(str(tmp_path), "m", "absent").empty

    def test_a_prediction_without_its_legacy_input_is_refused(self, tmp_path: Path) -> None:
        # The same guard the taxonomy uses: without the input, every value looks derived.
        _case(
            tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={"tissue": "lung"}
        )
        (tmp_path / "atacseq" / "input" / "r.json").unlink()
        with pytest.raises(FileNotFoundError, match="no legacy record"):
            create_availability_summary(str(tmp_path), "m", "sys")
