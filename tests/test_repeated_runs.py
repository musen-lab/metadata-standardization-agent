"""Tests for what reads several runs at once: the spread of the scores, and answer consistency.

The figure is captured instead of shown, by standing in for the ``_finish`` it ends with,
so a test can inspect what was drawn rather than only check that drawing did not raise.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.data_analysis import collect_run_scores, create_run_spread_summary  # noqa: E402
from notebook_utils import show_run_spread  # noqa: E402
from plots import run_spread, stability  # noqa: E402
from plots.marks import FIELD_TYPE_LABELS  # noqa: E402
from plots.run_spread import plot_run_spread  # noqa: E402
from plots.stability import plot_field_stability  # noqa: E402

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


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[plt.Figure]:
    """The figures the stability and run-spread plots finish with, kept open for inspection."""
    figures: list[plt.Figure] = []
    monkeypatch.setattr(stability, "_finish", lambda fig, _save_path: figures.append(fig))
    monkeypatch.setattr(run_spread, "_finish", lambda fig, _save_path: figures.append(fig))
    yield figures
    for fig in figures:
        plt.close(fig)


def _legend(fig: plt.Figure) -> tuple[list, list]:
    """The texts and the handles of the one axes legend of *fig*."""
    legend = fig.axes[0].get_legend()
    return legend.get_texts(), legend.legend_handles


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

    def test_a_run_not_made_yet_is_not_scored(self, data_root: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Scoring a run that is not there would warn about every one of its missing records."""
        with caplog.at_level("WARNING"):
            spread = create_run_spread_summary(str(data_root), "m", "baseline", runs=(1, 4))
        assert spread.empty
        assert caplog.records == []

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


class TestRunScores:
    def test_one_row_per_run(self, data_root: Path) -> None:
        scores = collect_run_scores(str(data_root), "m", "arms-agent", runs=(1, 2, 3))
        row = scores.query("assay == 'ATACseq' and field_type == 'non_ontology' and metric == 'precision'")
        assert row["run"].tolist() == [1, 2, 3]
        assert row["score"].tolist() == [1.0, 0.5, 1.0]

    def test_the_spread_is_taken_over_these_scores(self, data_root: Path) -> None:
        scores = collect_run_scores(str(data_root), "m", "baseline", runs=(1, 2, 3))
        spread = create_run_spread_summary(str(data_root), "m", "baseline", runs=(1, 2, 3))
        by_row = scores.groupby(["assay", "field_type", "metric"], sort=False)["score"]
        assert by_row.mean().round(3).tolist() == spread["mean"].tolist()
        assert (by_row.max() - by_row.min()).round(3).tolist() == spread["range"].tolist()

    def test_a_run_not_made_yet_leaves_it_empty(self, data_root: Path) -> None:
        assert collect_run_scores(str(data_root), "m", "baseline", runs=(1, 4)).empty


def _panel_offsets(ax: plt.Axes) -> list[float]:
    """The x of every dot in *ax*, rounded to a hundredth of a point."""
    return sorted(round(float(line.get_xdata()[0]), 2) for line in ax.get_lines() if line.get_marker() == "o")


class TestRunSpreadFigure:
    def test_a_row_per_assay_and_no_pooled_row(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_run_spread(str(data_root), "m", runs=(1, 2, 3))
        labels = [tick.get_text() for tick in captured[0].axes[0].get_yticklabels()]
        assert labels == ["ATACseq (n=2)"]

    def test_a_panel_per_field_type_and_metric(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_run_spread(str(data_root), "m", runs=(1, 2, 3))
        titles = [ax.get_title() for ax in captured[0].axes]
        assert titles == [
            f"{FIELD_TYPE_LABELS[field_type]}\n{metric}"
            for field_type in ("ontology", "non_ontology")
            for metric in ("Precision", "Recall")
        ]

    def test_each_run_is_its_distance_from_the_mean_in_points(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        plot_run_spread(str(data_root), "m", systems=("arms-agent",), runs=(1, 2, 3))
        non_ontology_precision = captured[0].axes[2]
        # Baseline: 1/1 in every run, so on 0.  ARMS: 2/2, 1/2, 2/2 against a mean of 5/6.
        assert _panel_offsets(non_ontology_precision) == [-33.33, 0.0, 0.0, 0.0, 16.67, 16.67]

    def test_the_axis_label_names_the_unit_and_can_be_replaced(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        plot_run_spread(str(data_root), "m", runs=(1, 2))
        plot_run_spread(str(data_root), "m", runs=(1, 2), x_label="Run minus mean")
        default, replaced = ({text.get_text() for text in fig.texts} for fig in captured)
        assert "Difference from the mean of 2 runs (percentage points)" in default
        assert "Run minus mean" in replaced

    def test_the_title_is_written_only_when_given(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_run_spread(str(data_root), "m", runs=(1, 2))
        plot_run_spread(str(data_root), "m", runs=(1, 2), title="Variability")
        untitled, titled = captured
        assert untitled._suptitle is None
        assert titled._suptitle.get_text() == "Variability"

    def test_runs_not_made_are_refused(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="No assay was scored"):
            plot_run_spread(str(data_root), "m", runs=(1, 4))


class TestFieldStabilityFigure:
    def test_a_bar_per_condition_for_each_assay_and_no_pooled_rows(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        plot_field_stability(str(data_root), "m", runs=(1, 2, 3))
        ax = captured[0].axes[0]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["Baseline", "ARMS"]
        names = {text.get_text() for text in ax.texts}
        assert "ATACseq" in names
        assert "All assays" not in names

    def test_the_bars_carry_their_shares_and_nothing_else(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_field_stability(str(data_root), "m", runs=(1, 2, 3))
        ax = captured[0].axes[0]
        written = {text.get_text() for text in ax.texts} - {"ATACseq"}
        assert written and all(text.endswith("%") for text in written), written
        assert list(ax.get_lines()) == [], "no dot beside a condition's name"

    def test_the_consistent_band_is_the_dark_one(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_field_stability(str(data_root), "m", runs=(1, 2, 3))
        keys = {text.get_text(): handle for text, handle in zip(*_legend(captured[0]), strict=True)}
        dark, light = (
            keys[label].get_facecolor() for label in ("same answer in every run", "answer changed between runs")
        )
        assert sum(dark[:3]) < sum(light[:3])

    def test_the_axis_says_what_is_measured(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_field_stability(str(data_root), "m", runs=(1, 2, 3))
        assert captured[0].axes[0].get_xlabel() == "Share of field instances"

    def test_a_restricted_figure_names_its_field_type(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_field_stability(str(data_root), "m", runs=(1, 2), field_type="ontology")
        assert FIELD_TYPE_LABELS["ontology"] in captured[0].axes[0].get_xlabel()

    def test_one_run_is_refused(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="two runs"):
            plot_field_stability(str(data_root), "m", runs=(1,))

    def test_no_predictions_is_refused(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="No field instances"):
            plot_field_stability(str(data_root), "m", conditions=("absent",), runs=(1, 2))
