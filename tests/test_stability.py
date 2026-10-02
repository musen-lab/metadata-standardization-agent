"""Tests for :mod:`analysis.data_analysis.stability`: consistency on reference fields."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import (
    STABILITY_BANDS,
    collect_field_stability,
    rank_inconsistent_fields,
    summarize_field_stability,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


_GOLD = {
    "r1": {"tissue": "lung", "title": "study", "lab_id": "", "note": "", "instrument": ""},
    "r2": {"tissue": "lung", "title": "a", "lab_id": "", "note": "n", "instrument": "Axio"},
    "r3": {"tissue": "lung", "title": "b", "lab_id": "", "note": "", "instrument": ""},
}

#: What each of three runs answered.  r3 is missing from run 3.
_RUNS = {
    1: {
        "r1": {"tissue": "lung", "title": "study", "lab_id": "X", "note": "", "instrument": ""},
        "r2": {"tissue": "kidney", "title": "a", "lab_id": "", "note": "", "instrument": "Axio"},
        "r3": {"tissue": "lung", "title": "b", "lab_id": "", "note": "", "instrument": ""},
    },
    2: {
        "r1": {"tissue": "lung", "title": "WRONG", "lab_id": "", "note": None, "instrument": ""},
        "r2": {"tissue": "kidney", "title": "a", "lab_id": "", "note": None, "instrument": "axio"},
        "r3": {"tissue": "lung", "title": "b", "lab_id": "", "note": "", "instrument": ""},
    },
    3: {
        "r1": {"tissue": "lung", "title": "study", "lab_id": "", "note": "", "instrument": ""},
        "r2": {"tissue": "kidney", "title": "a", "lab_id": "", "note": "", "instrument": "Axio"},
    },
}


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """One assay, three records, three runs of one condition."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
            {"name": "lab_id", "permissible_values": []},
            {"name": "note", "permissible_values": []},
            {"name": "instrument", "permissible_values": []},
        ]
    }
    _write(tmp_path / "schemas" / "atacseq.json", schema)
    for name, gold in _GOLD.items():
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", gold)
    for run, records in _RUNS.items():
        for name, record in records.items():
            _write(tmp_path / "atacseq" / "output" / "m" / "sys" / f"run-{run}" / f"{name}.json", record)
    return tmp_path


@pytest.fixture
def stability(data_root: Path) -> pd.DataFrame:
    return collect_field_stability(data_root, "m", "sys", runs=(1, 2, 3))


def _row(stability: pd.DataFrame, record: str, field: str) -> pd.Series:
    rows = stability[(stability["record"] == f"{record}.json") & (stability["field"] == field)]
    assert len(rows) == 1, (record, field)
    return rows.iloc[0]


class TestCollectFieldStability:
    def test_the_same_answer_in_every_run_is_consistent(self, stability: pd.DataFrame) -> None:
        assert _row(stability, "r1", "tissue")["band"] == "consistent"

    def test_one_different_answer_makes_it_inconsistent(self, stability: pd.DataFrame) -> None:
        row = _row(stability, "r1", "title")  # "study", "WRONG", "study"
        assert (row["n_answers"], row["band"]) == (2, "inconsistent")

    def test_the_same_wrong_answer_every_time_is_consistent(self, stability: pd.DataFrame) -> None:
        """A stable wrong value remains consistent when the reference field is populated."""
        assert _row(stability, "r2", "tissue")["band"] == "consistent"

    def test_a_change_of_case_is_a_different_answer(self, stability: pd.DataFrame) -> None:
        assert _row(stability, "r2", "instrument")["band"] == "inconsistent"  # "Axio", "axio"

    def test_blank_and_null_are_the_same_answer(self, stability: pd.DataFrame) -> None:
        # Gold wants a value; the runs answered "", None, "" -- reliably nothing.
        assert _row(stability, "r2", "note")["band"] == "consistent"

    def test_a_value_filled_in_only_some_runs_is_inconsistent(self, stability: pd.DataFrame) -> None:
        assert _row(stability, "r1", "lab_id")["band"] == "inconsistent"  # "X", "", ""

    def test_reference_blank_fields_left_blank_are_counted_as_consistent(self, stability: pd.DataFrame) -> None:
        for record, field in (("r1", "note"), ("r1", "instrument"), ("r2", "lab_id")):
            row = _row(stability, record, field)
            assert (row["n_answers"], row["band"]) == (1, "consistent")

    def test_reference_blank_field_filled_identically_in_all_runs_is_consistent(self, data_root: Path) -> None:
        for run in (1, 2, 3):
            path = data_root / "atacseq" / "output" / "m" / "sys" / f"run-{run}" / "r2.json"
            record = json.loads(path.read_text())
            record["lab_id"] = "X"
            _write(path, record)
        row = _row(collect_field_stability(data_root, "m", "sys", runs=(1, 2, 3)), "r2", "lab_id")
        assert (row["n_answers"], row["band"]) == (1, "consistent")

    def test_a_record_missing_from_one_run_is_skipped(self, stability: pd.DataFrame) -> None:
        assert "r3.json" not in set(stability["record"])

    def test_the_field_type_comes_from_the_schema(self, stability: pd.DataFrame) -> None:
        types = dict(zip(stability["field"], stability["field_type"], strict=False))
        assert types["tissue"] == "ontology"
        assert types["title"] == "non_ontology"

    def test_the_order_of_the_runs_does_not_matter(self, data_root: Path, stability: pd.DataFrame) -> None:
        assert collect_field_stability(data_root, "m", "sys", runs=(3, 1, 2)).equals(stability)

    def test_two_runs_are_enough(self, data_root: Path) -> None:
        # Runs 1 and 3 agree on r1's title, so it is no longer inconsistent.
        pair = collect_field_stability(data_root, "m", "sys", runs=(1, 3))
        assert _row(pair, "r1", "title")["band"] == "consistent"
        assert set(pair["n_runs"]) == {2}

    @pytest.mark.parametrize("runs", [(1,), (), (1, 1)])
    def test_fewer_than_two_distinct_runs_are_refused(self, data_root: Path, runs: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="run"):
            collect_field_stability(data_root, "m", "sys", runs=runs)

    def test_a_condition_with_no_runs_on_disk_is_an_empty_table(self, data_root: Path) -> None:
        empty = collect_field_stability(data_root, "m", "absent", runs=(1, 2))
        assert empty.empty
        assert list(empty.columns) == ["assay", "record", "field", "field_type", "n_runs", "n_answers", "band"]


class TestSummarizeFieldStability:
    def test_shares_per_assay_and_pooled(self, stability: pd.DataFrame) -> None:
        # Both complete records contribute every reference field: 2 x 5 = 10 instances.
        # Inconsistent: r1 title, r1 lab_id, r2 instrument.
        table = summarize_field_stability(stability).set_index("assay")
        for row in ("ATACseq", "All assays"):
            assert table.loc[row, "n_instances"] == 10
            assert table.loc[row, "consistent"] == pytest.approx(7 / 10)
            assert table.loc[row, "inconsistent"] == pytest.approx(3 / 10)

    def test_the_shares_of_a_row_add_up_to_one(self, stability: pd.DataFrame) -> None:
        table = summarize_field_stability(stability)
        assert table[list(STABILITY_BANDS)].sum(axis=1).tolist() == pytest.approx([1.0] * len(table))

    def test_a_field_type_restricts_the_table(self, stability: pd.DataFrame) -> None:
        table = summarize_field_stability(stability, field_type="ontology").set_index("assay")
        assert table.loc["All assays", "n_instances"] == 2
        assert table.loc["All assays", "inconsistent"] == 0.0

    def test_the_pooled_row_comes_last(self, stability: pd.DataFrame) -> None:
        assert summarize_field_stability(stability)["assay"].iloc[-1] == "All assays"


class TestRankInconsistentFields:
    def test_only_inconsistent_fields_are_listed(self, stability: pd.DataFrame) -> None:
        ranked = rank_inconsistent_fields(stability)
        assert set(ranked["field"]) == {"title", "lab_id", "instrument"}
        assert (ranked["n_inconsistent"] > 0).all()

    def test_the_share_is_inconsistent_over_all_reference_instances(self, stability: pd.DataFrame) -> None:
        # Title is counted in both records: one inconsistent and one consistent.
        title = rank_inconsistent_fields(stability).set_index("field").loc["title"]
        assert (title["n_records"], title["n_inconsistent"], title["inconsistent_share"]) == (2, 1, 0.5)

    def test_fields_filled_identically_despite_blank_reference_are_not_inconsistent(self, data_root: Path) -> None:
        for run in (1, 2, 3):
            path = data_root / "atacseq" / "output" / "m" / "sys" / f"run-{run}" / "r2.json"
            record = json.loads(path.read_text())
            record["lab_id"] = "X"
            _write(path, record)
        stability = collect_field_stability(data_root, "m", "sys", runs=(1, 2, 3))
        lab_id = rank_inconsistent_fields(stability).set_index("field").loc["lab_id"]
        assert (lab_id["n_records"], lab_id["n_inconsistent"]) == (2, 1)

    def test_top_keeps_that_many_rows(self, stability: pd.DataFrame) -> None:
        assert len(rank_inconsistent_fields(stability, top=1)) == 1
        assert len(rank_inconsistent_fields(stability, top=None)) == 3
