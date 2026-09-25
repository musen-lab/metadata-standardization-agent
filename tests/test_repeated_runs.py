"""Tests for what reads several runs at once: the spread of the scores."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import create_run_spread_summary
from notebook_utils import show_run_spread

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


_GOLD = {"r1": {"tissue": "lung", "title": "study"}, "r2": {"tissue": "heart", "title": "x"}}

#: condition -> run -> record -> prediction.  ARMS gets r1's title wrong in run 2 only;
#: baseline gets r1's tissue wrong every time and drops r2's tissue in run 2.
_PREDICTIONS = {
    "baseline": {
        run: {
            "r1": {"tissue": "WRONG", "title": "study"},
            "r2": {"tissue": "" if run == 2 else "heart", "title": "x"},
        }
        for run in (1, 2, 3)
    },
    "arms-agent": {
        run: {
            "r1": {"tissue": "lung", "title": "bad" if run == 2 else "study"},
            "r2": {"tissue": "heart", "title": "x"},
        }
        for run in (1, 2, 3)
    },
}


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """One assay, two records, three runs of baseline and of ARMS."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
        ]
    }
    _write(tmp_path / "schemas" / "atacseq.json", schema)
    for name, gold in _GOLD.items():
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", gold)
    for condition, runs in _PREDICTIONS.items():
        for run, records in runs.items():
            for name, record in records.items():
                _write(tmp_path / "atacseq" / "output" / "m" / condition / f"run-{run}" / f"{name}.json", record)
    return tmp_path


class TestRunSpread:
    """ARMS gets r1's title wrong in run 2 only, so its non-ontology scores move; nothing else does."""

    def test_the_mean_and_the_lowest_and_highest_run(self, data_root: Path) -> None:
        spread = create_run_spread_summary(str(data_root), "m", "arms-agent", runs=(1, 2, 3))
        row = spread.query("assay == 'ATACseq' and field_type == 'non_ontology' and metric == 'precision'").iloc[0]
        # Precision per run: 2/2, 1/2, 2/2.
        assert (row["mean"], row["min"], row["max"], row["range"]) == (0.833, 0.5, 1.0, 0.5)

    def test_a_score_no_run_moves_has_no_range(self, data_root: Path) -> None:
        spread = create_run_spread_summary(str(data_root), "m", "arms-agent", runs=(1, 2, 3))
        ontology = spread[spread["field_type"] == "ontology"]
        assert set(ontology["range"]) == {0.0}
        assert set(ontology["mean"]) == {1.0}

    def test_the_pooled_row_follows_the_assays(self, data_root: Path) -> None:
        spread = create_run_spread_summary(str(data_root), "m", "arms-agent", runs=(1, 2, 3))
        assert spread["assay"].tolist() == ["ATACseq", "ATACseq", "All assays", "All assays"] * 2
        # One assay, so the pooled rows repeat it.
        by_assay = spread.set_index(["assay", "field_type", "metric"])[["mean", "min", "max"]]
        assert by_assay.loc["ATACseq"].equals(by_assay.loc["All assays"])

    def test_each_row_says_how_many_records_stand_behind_it(self, data_root: Path) -> None:
        spread = create_run_spread_summary(str(data_root), "m", "baseline", runs=(1, 2))
        assert set(spread["n_records"]) == {2}

    def test_field_types_are_chosen(self, data_root: Path) -> None:
        spread = create_run_spread_summary(str(data_root), "m", "baseline", runs=(1, 2), field_types=("all",))
        assert set(spread["field_type"]) == {"all"}

    def test_an_assay_missing_from_a_run_is_left_out(self, data_root: Path) -> None:
        spread = create_run_spread_summary(str(data_root), "m", "baseline", runs=(1, 4))
        assert spread.empty
        assert list(spread.columns) == ["assay", "field_type", "metric", "n_records", "mean", "min", "max", "range"]

    @pytest.mark.parametrize("runs", [(1,), (), (2, 2)])
    def test_fewer_than_two_distinct_runs_are_refused(self, data_root: Path, runs: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="run"):
            create_run_spread_summary(str(data_root), "m", "baseline", runs=runs)

    def test_the_printed_table_puts_the_range_beside_the_mean(
        self, data_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        table = show_run_spread(str(data_root), "m", ["baseline", "arms-agent"], runs=(1, 2, 3))
        assert list(table.columns) == [
            "assay",
            "field_type",
            "Baseline precision",
            "Baseline recall",
            "ARMS precision",
            "ARMS recall",
        ]
        row = table[(table["assay"] == "ATACseq") & (table["field_type"] == "non_ontology")].iloc[0]
        assert row["ARMS precision"] == "0.833 [0.500, 1.000]"
        assert "mean [lowest run, highest run]" in capsys.readouterr().out

    def test_the_printed_table_says_when_the_runs_are_missing(
        self, data_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        table = show_run_spread(str(data_root), "m", ["baseline"], runs=(1, 4))
        assert table.empty
        assert "no assay has predictions in every one of runs [1, 4]" in capsys.readouterr().out
