"""Tests for the precision/recall figure that compares models under one condition.

The figure is captured instead of shown, by standing in for the ``_finish`` it ends with,
so a test can inspect what was drawn rather than only check that drawing did not raise.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib
import pytest
from matplotlib.colors import to_hex

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from plots import pr_space  # noqa: E402
from plots.marks import MODEL_COLOURS, _model_marks  # noqa: E402
from plots.pr_space import plot_pr_condition_comp, plot_pr_model_comp  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


_GOLD = {"r1": {"tissue": "lung", "title": "study"}, "r2": {"tissue": "heart", "title": "x"}}

#: model -> record -> prediction.  "good" gets everything right; "half" leaves r2's tissue
#: empty and gets r1's title wrong.
_PREDICTIONS = {
    "good": {"r1": {"tissue": "lung", "title": "study"}, "r2": {"tissue": "heart", "title": "x"}},
    "half": {"r1": {"tissue": "lung", "title": "bad"}, "r2": {"tissue": "", "title": "x"}},
}


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """One assay, two records, ARMS run once by each of two models."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
        ]
    }
    _write(tmp_path / "schemas" / "atacseq.json", schema)
    for name, gold in _GOLD.items():
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", gold)
    for model, records in _PREDICTIONS.items():
        for name, record in records.items():
            _write(tmp_path / "atacseq" / "output" / model / "arms-agent" / f"{name}.json", record)
    return tmp_path


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[plt.Figure]:
    """The figures the precision/recall plots finish with, kept open for inspection."""
    figures: list[plt.Figure] = []
    monkeypatch.setattr(pr_space, "_finish", lambda fig, _save_path: figures.append(fig))
    yield figures
    for fig in figures:
        plt.close(fig)


def _points(ax: plt.Axes) -> dict[str, list[tuple[float, float]]]:
    """Every point in *ax*, keyed by its fill colour."""
    points: dict[str, list[tuple[float, float]]] = {}
    for line in ax.get_lines():
        if line.get_linestyle() == "None" and len(line.get_xdata()) == 1:
            key = to_hex(line.get_markerfacecolor())
            points.setdefault(key, []).append((float(line.get_xdata()[0]), float(line.get_ydata()[0])))
    return points


class TestModelMarks:
    def test_each_model_takes_the_next_colour(self) -> None:
        marks = _model_marks(("a", "b", "c"), no_color=False)
        assert [mark.style["markerfacecolor"] for _model, mark in marks] == list(MODEL_COLOURS)

    def test_without_colour_the_models_are_lettered_and_alternate_fill(self) -> None:
        marks = _model_marks(("a", "b", "c"), no_color=True)
        assert [mark.letter for _model, mark in marks] == ["A", "B", "C"]
        assert [mark.fill for _model, mark in marks] == [True, False, True]

    def test_a_lone_model_has_no_letter(self) -> None:
        [(_model, mark)] = _model_marks(("a",), no_color=True)
        assert mark.letter is None

    @pytest.mark.parametrize("models", [(), ("a", "a"), ("a", "b", "c", "d")])
    def test_models_that_cannot_be_told_apart_are_refused(self, models: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="model"):
            _model_marks(models, no_color=False)


class TestModelFigure:
    def test_each_model_is_its_own_colour_at_its_own_scores(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), field_types=("ontology", "non_ontology"))
        points = _points(captured[0].axes[0])
        good, half = MODEL_COLOURS[0], MODEL_COLOURS[1]
        # good: everything right.  half: ontology 1/1 precision, 1/2 recall; non-ontology 1/2 and 1/2.
        assert sorted(points[good]) == [(1.0, 1.0), (1.0, 1.0)]
        assert sorted(points[half]) == [(0.5, 0.5), (0.5, 1.0)]

    def test_the_legend_names_the_models(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"))
        labels = [text.get_text() for legend in captured[0].legends for text in legend.get_texts()]
        assert labels[:2] == ["good", "half"]

    def test_the_assays_can_be_broken_out(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), assays=("atacseq",))
        assert captured[0].axes[0].get_title() == "ATACseq (n=2)"

    def test_a_model_without_predictions_is_refused(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="No requested assay"):
            plot_pr_model_comp(str(data_root), ("good", "absent"), assays=("atacseq",))


class TestShapeStacking:
    def test_square_behind_circle_behind_triangle(self, data_root: Path, captured: list[plt.Figure]) -> None:
        """Three field types on one spot stay three shapes: the largest at the back."""
        plot_pr_model_comp(str(data_root), ("good",))
        zorders = {line.get_marker(): line.get_zorder() for line in captured[0].axes[0].get_lines()}
        assert zorders["s"] < zorders["o"] < zorders["^"]

    def test_a_letter_sits_over_its_own_mark_and_under_the_next_shape(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), no_color=True)
        ax = captured[0].axes[0]
        marks = {line.get_marker(): line.get_zorder() for line in ax.get_lines()}
        letters = sorted(text.get_zorder() for text in ax.texts if text.get_text() in {"A", "B"})
        assert marks["s"] < letters[0] < marks["o"]
        assert letters[-1] > marks["^"]


class TestConditionLegend:
    def test_conditions_and_field_types_take_their_manuscript_names(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        for name, record in _PREDICTIONS["half"].items():
            _write(data_root / "atacseq" / "output" / "good" / "template-tool" / f"{name}.json", record)
        plot_pr_condition_comp(str(data_root), "good", baselines=("template-tool",), systems=("arms-agent",))
        labels = [text.get_text() for legend in captured[0].legends for text in legend.get_texts()]
        assert labels == [
            "GetTemplate-tool Only",
            "ARMS",
            "Ontology-constrained Fields",
            "Non-ontology-constrained Fields",
            "All Fields",
        ]
