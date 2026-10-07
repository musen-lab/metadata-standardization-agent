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
from matplotlib.patches import Patch

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from plots import pr_space  # noqa: E402
from plots.marks import MODEL_COLOURS, MODEL_HATCHES, _model_marks  # noqa: E402
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
        if line.get_gid() != pr_space.STACK_COPY_GID and line.get_linestyle() == "None" and len(line.get_xdata()) == 1:
            key = to_hex(line.get_markerfacecolor())
            points.setdefault(key, []).append((float(line.get_xdata()[0]), float(line.get_ydata()[0])))
    return points


class TestModelMarks:
    def test_each_model_takes_the_next_colour_solid(self) -> None:
        marks = _model_marks(tuple("abcdefgh"), no_color=False)
        assert [mark.style["markerfacecolor"] for _model, mark in marks] == list(MODEL_COLOURS)
        assert all(mark.style["markeredgecolor"] == "white" and mark.hatch is None for _model, mark in marks)

    def test_without_colour_each_model_takes_the_next_pattern(self) -> None:
        marks = _model_marks(tuple("abcdefgh"), no_color=True)
        assert [mark.hatch for _model, mark in marks] == list(MODEL_HATCHES)
        assert len(set(MODEL_HATCHES)) == 8
        # The first is solid ink; the rest are ink patterns on white.  None is lettered.
        assert [mark.style["markerfacecolor"] for _model, mark in marks][1:] == ["white"] * 7
        assert all(mark.halo and mark.letter is None for _model, mark in marks)

    @pytest.mark.parametrize("models", [(), ("a", "a"), tuple("abcdefghi")])
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

    def test_a_model_with_no_predictions_yet_is_left_out_quietly(
        self, data_root: Path, captured: list[plt.Figure], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            plot_pr_model_comp(str(data_root), ("good", "absent", "half"), assays=("atacseq",))
        labels = [text.get_text() for text in captured[0].legends[0].get_texts()]
        assert labels == ["good", "half"]
        assert caplog.records == []

    def test_a_model_is_drawn_in_the_assays_it_has(self, data_root: Path, captured: list[plt.Figure]) -> None:
        """ "half" has not reached RNAseq yet: that panel holds "good" alone."""
        _write(data_root / "schemas" / "rnaseq.json", json.loads((data_root / "schemas" / "atacseq.json").read_text()))
        for name, gold in _GOLD.items():
            _write(data_root / "rnaseq" / "gold" / f"{name}.json", gold)
            _write(data_root / "rnaseq" / "output" / "good" / "arms-agent" / f"{name}.json", _PREDICTIONS["good"][name])
        plot_pr_model_comp(str(data_root), ("good", "half"), assays=("atacseq", "rnaseq"))
        atacseq, rnaseq = captured[0].axes[:2]
        assert set(_points(atacseq)) == {MODEL_COLOURS[0], MODEL_COLOURS[1]}
        assert set(_points(rnaseq)) == {MODEL_COLOURS[0]}
        assert rnaseq.get_title() == "RNAseq (n=2)"

    def test_pooled_a_model_must_cover_every_assay(self, data_root: Path, captured: list[plt.Figure]) -> None:
        """Pooled over fewer assays, a model's score would not be comparable with the others'."""
        _write(data_root / "schemas" / "rnaseq.json", json.loads((data_root / "schemas" / "atacseq.json").read_text()))
        for name, gold in _GOLD.items():
            _write(data_root / "rnaseq" / "gold" / f"{name}.json", gold)
            _write(data_root / "rnaseq" / "output" / "good" / "arms-agent" / f"{name}.json", _PREDICTIONS["good"][name])
        plot_pr_model_comp(str(data_root), ("good", "half"))
        labels = [text.get_text() for text in captured[0].legends[0].get_texts()]
        assert labels[0] == "good"
        assert "half" not in labels

    def test_nothing_to_draw_is_refused(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="None of"):
            plot_pr_model_comp(str(data_root), ("absent", "missing"), assays=("atacseq",))


def _eight_models(data_root: Path) -> tuple[str, ...]:
    """Eight models that all score perfectly, so every one of their marks lands on one spot."""
    models = tuple(f"m{index}" for index in range(8))
    for model in models:
        for name, record in _PREDICTIONS["good"].items():
            _write(data_root / "atacseq" / "output" / model / "arms-agent" / f"{name}.json", record)
    return models


def _stack_boxes(ax: plt.Axes) -> list[str]:
    """The text of every stack name box in *ax*."""
    return [text.get_text() for text in ax.texts if text.get_gid() == pr_space.STACK_NAME_GID]


class TestPatternedFigure:
    def test_without_colour_every_model_is_drawn_in_its_pattern(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        models = _eight_models(data_root)
        plot_pr_model_comp(str(data_root), models, field_types=("ontology",), no_color=True)
        ax = captured[0].axes[0]
        hatches = [
            collection.get_hatch() for collection in ax.collections if collection.get_gid() != pr_space.STACK_COPY_GID
        ]
        assert hatches == [hatch for hatch in MODEL_HATCHES if hatch]
        legend_hatches = [handle.get_hatch() for handle in captured[0].legends[0].legend_handles]
        assert sorted(legend_hatches, key=str) == sorted(MODEL_HATCHES, key=str)

    def test_wrapped_keys_read_across_each_row_and_stay_inside_the_figure(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        models = _eight_models(data_root)
        plot_pr_model_comp(str(data_root), models, no_color=True)
        fig = captured[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = fig.legends[0]
        assert legend.get_window_extent(renderer).width <= fig.bbox.width
        placed = [(text.get_window_extent(renderer), text.get_text()) for text in legend.get_texts()]
        reading = [label for box, label in sorted(placed, key=lambda item: (-round(item[0].y0), item[0].x0))]
        assert reading == list(models)


class TestStackNames:
    def test_callout_threshold_uses_the_visible_shape(self) -> None:
        fig, ax, _points, marks_of = _inset_panel([(0.5, 0.5), (0.5, 0.5)])
        centre = ax.transData.transform((0.5, 0.5))

        def pair(apart_in_diameters: float) -> list[tuple[str, int, tuple[float, float]]]:
            distance = apart_in_diameters * pr_space.NO_COLOR_INK_DIAMETER * fig.dpi / 72
            second = tuple(ax.transData.inverted().transform((centre[0] + distance, centre[1])))
            return [("ontology", 0, (0.5, 0.5)), ("ontology", 1, second)]

        assert pr_space._obscured_members(ax, pair(0.8), marks_of) == {0}
        assert pr_space._obscured_members(ax, pair(1.2), marks_of) == set()
        plt.close(fig)

    def test_touching_distinct_scores_keep_their_positions_and_label_only_the_hidden_mark(self) -> None:
        scores = [(0.80, 0.80), (0.81, 0.81)]
        fig, ax, panel_points, marks_of = _inset_panel(scores)
        for index, score in enumerate(scores):
            pr_space._draw_pr_path(ax, [score], [marks_of[index]])
        pr_space._name_stacks(ax, panel_points, ["first", "second"], marks_of)
        assert _stack_boxes(ax) == ["first"]
        assert sorted(point for values in _points(ax).values() for point in values) == scores
        assert not any(line.get_gid() == pr_space.STACK_COPY_GID for line in ax.get_lines())
        plt.close(fig)

    def test_every_model_in_an_exact_tie_is_named_at_the_true_score(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        for name, record in _PREDICTIONS["good"].items():
            _write(data_root / "atacseq" / "output" / "copy" / "arms-agent" / f"{name}.json", record)
        plot_pr_model_comp(str(data_root), ("good", "half", "copy"), field_types=("ontology",))
        ax = captured[0].axes[0]
        assert set(_stack_boxes(ax)) == {"good", "copy"}
        assert not any(line.get_gid() == pr_space.STACK_COPY_GID for line in ax.get_lines())
        leaders = [line for line in ax.get_lines() if line.get_zorder() == 1.5]
        assert len(leaders) == 2
        assert all(tuple(line.get_xydata()[0]) == (1.0, 1.0) for line in leaders)

    def test_each_line_starts_at_its_own_stack(self, data_root: Path, captured: list[plt.Figure]) -> None:
        """A model's stacks sit at different spots per field type; each line leaves its own one.

        "half" scores (0.5, 1.0) on ontology fields and (0.5, 0.5) on the rest.
        """
        for name, record in _PREDICTIONS["half"].items():
            _write(data_root / "atacseq" / "output" / "twin" / "arms-agent" / f"{name}.json", record)
        plot_pr_model_comp(str(data_root), ("half", "twin"), field_types=("ontology", "non_ontology"))
        ax = captured[0].axes[0]
        leaders = [line for line in ax.get_lines() if line.get_zorder() == 1.5]
        starts = {(float(line.get_xdata()[0]), float(line.get_ydata()[0])) for line in leaders}
        assert starts == {(0.5, 1.0), (0.5, 0.5)}
        assert not any(line.get_gid() == pr_space.STACK_SCORE_LINK_GID for line in ax.get_lines())

    def test_four_perfect_models_stay_overlapped_with_names_on_leaders(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        models = _eight_models(data_root)[:4]
        plot_pr_model_comp(str(data_root), models, field_types=("ontology",), no_color=True)
        ax = captured[0].axes[0]
        assert set(_stack_boxes(ax)) == set(models)
        assert not any(line.get_gid() == pr_space.STACK_COPY_GID for line in ax.get_lines())
        assert not any(collection.get_gid() == pr_space.STACK_COPY_GID for collection in ax.collections)
        assert all(tuple(collection.get_offsets()[0]) == (1.0, 1.0) for collection in ax.collections)
        leaders = [line for line in ax.get_lines() if line.get_zorder() == 1.5]
        assert len(leaders) == len(models)
        assert all(tuple(line.get_xydata()[0]) == (1.0, 1.0) for line in leaders)

    def test_marks_apart_are_not_named(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), field_types=("ontology",))
        assert _stack_boxes(captured[0].axes[0]) == []

    def test_a_shape_hidden_by_a_different_field_type_gets_a_callout(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        """The visibility rule applies across shapes when one covers another's outline."""
        plot_pr_model_comp(str(data_root), ("good",), field_types=("ontology", "non_ontology"))
        assert _stack_boxes(captured[0].axes[0]) == ["good"]

    def test_a_name_box_covers_no_mark_and_its_line_runs_under_them(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        models = _eight_models(data_root)
        plot_pr_model_comp(str(data_root), (*models[:4], "half"), field_types=("ontology", "non_ontology"))
        fig = captured[0]
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        box = next(text for text in ax.texts if text.get_gid() == pr_space.STACK_NAME_GID)
        extent = box.get_bbox_patch().get_window_extent(renderer)
        marks = [line for line in ax.get_lines() if line.get_linestyle() == "None"]
        for mark in marks:
            x, y = ax.transData.transform((mark.get_xdata()[0], mark.get_ydata()[0]))
            assert not extent.contains(x, y)
        leaders = [line for line in ax.get_lines() if len(line.get_xdata()) == 2 and line.get_zorder() == 1.5]
        assert len(leaders) == len(_stack_boxes(ax))
        assert all(leader.get_zorder() < min(mark.get_zorder() for mark in marks) for leader in leaders)
        assert extent.x1 <= ax.get_window_extent(renderer).x1


class TestFieldTypeKey:
    def test_one_field_type_has_no_field_type_key(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), field_types=("ontology",))
        [legend] = captured[0].legends
        assert [text.get_text() for text in legend.get_texts()] == ["good", "half"]

    def test_several_field_types_keep_their_key(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), field_types=("ontology", "non_ontology"))
        assert len(captured[0].legends) == 2

    def test_the_title_is_written_over_the_figure(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_model_comp(str(data_root), ("good", "half"), field_types=("ontology",), title="Ontology fields")
        assert captured[0].get_suptitle() == "Ontology fields"


class TestShapeStacking:
    def test_square_behind_circle_behind_triangle(self, data_root: Path, captured: list[plt.Figure]) -> None:
        """Three field types on one spot stay three shapes: the largest at the back."""
        plot_pr_model_comp(str(data_root), ("good",))
        zorders = {
            line.get_marker(): line.get_zorder()
            for line in captured[0].axes[0].get_lines()
            if line.get_gid() != pr_space.STACK_COPY_GID
        }
        assert zorders["s"] < zorders["o"] < zorders["^"]


class TestConditionLegend:
    def test_tied_conditions_get_separate_marker_callouts(self, data_root: Path, captured: list[plt.Figure]) -> None:
        for name, record in _PREDICTIONS["good"].items():
            _write(data_root / "atacseq" / "output" / "good" / "baseline" / f"{name}.json", record)
        plot_pr_condition_comp(str(data_root), "good", field_types=("ontology",))
        ax = captured[0].axes[0]
        assert _stack_boxes(ax) == ["Baseline"]
        assert len([line for line in ax.get_lines() if line.get_gid() == pr_space.STACK_COPY_GID]) == 1
        assert len([line for line in ax.get_lines() if line.get_gid() == pr_space.STACK_SCORE_LINK_GID]) == 1

    def test_conditions_and_field_types_take_their_manuscript_names(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        for name, record in _PREDICTIONS["half"].items():
            _write(data_root / "atacseq" / "output" / "good" / "template-tool" / f"{name}.json", record)
        plot_pr_condition_comp(str(data_root), "good", baselines=("template-tool",), systems=("arms-agent",))
        labels = [text.get_text() for legend in captured[0].legends for text in legend.get_texts()]
        assert labels == [
            "Template-tool Only",
            "ARMS",
            "Ontology-constrained Fields",
            "Non-ontology-constrained Fields",
            "All Fields",
        ]

    @pytest.mark.parametrize("no_color", [False, True])
    def test_what_is_compared_is_keyed_by_swatch_not_by_a_field_type_s_shape(
        self, data_root: Path, captured: list[plt.Figure], no_color: bool
    ) -> None:
        """A circle in the condition key would read as the ontology-constrained fields' circle."""
        plot_pr_model_comp(str(data_root), ("good", "half"), no_color=no_color)
        compared, field_types = captured[0].legends
        assert all(isinstance(handle, Patch) for handle in compared.legend_handles)
        assert [handle.get_marker() for handle in field_types.legend_handles] == ["o", "s", "^"]


def _inset_panel(points: list[tuple[float, float]]) -> tuple[plt.Figure, plt.Axes, list, dict]:
    """A lone styled panel holding one ontology mark per point, model by model."""
    fig, ax = plt.subplots(figsize=(pr_space.SHARED_PANEL_INCHES, pr_space.SHARED_PANEL_INCHES))
    pr_space._style_pr_axes(ax)
    marks = _model_marks(tuple(f"m{index}" for index in range(len(points))), no_color=False)
    panel_points = [("ontology", index, xy) for index, xy in enumerate(points)]
    fig.canvas.draw()
    return fig, ax, panel_points, {index: mark for index, (_model, mark) in enumerate(marks)}


class TestInsets:
    def test_marks_crowded_at_different_scores_are_enlarged(self) -> None:
        fig, ax, panel_points, marks_of = _inset_panel([(0.90, 0.90), (0.92, 0.91), (0.3, 0.4)])
        [(inset, group)] = pr_space._draw_insets(ax, panel_points, marks_of)
        assert group == [0, 1]
        (x0, x1), (y0, y1) = inset.get_xlim(), inset.get_ylim()
        assert x0 < 0.90 and x1 > 0.92 and y0 < 0.90 and y1 > 0.91
        assert x1 - x0 == pytest.approx(y1 - y0)  # square: a step sideways means a step up
        # Enlarged at least MIN_ZOOM times, and drawing every mark in its window.
        assert inset.get_position().width / ax.get_position().width * 1.13 / (x1 - x0) >= pr_space.MIN_ZOOM
        assert len([line for line in inset.get_lines() if line.get_linestyle() == "None"]) == 3
        plt.close(fig)

    def test_marks_at_one_score_are_not_enlarged(self) -> None:
        """No zoom can part them; they are named instead."""
        fig, ax, panel_points, marks_of = _inset_panel([(0.9, 0.9), (0.9, 0.9)])
        assert pr_space._draw_insets(ax, panel_points, marks_of) == []
        plt.close(fig)

    def test_an_inset_covers_no_mark_and_offers_no_score_above_one(self) -> None:
        points = [(1.0, 1.0), (0.99, 0.985), (0.2, 0.2), (0.25, 0.6), (0.6, 0.25)]
        fig, ax, panel_points, marks_of = _inset_panel(points)
        [(inset, _group)] = pr_space._draw_insets(ax, panel_points, marks_of)
        assert inset.get_ylim()[1] <= pr_space.PR_WINDOW[1] and inset.get_xlim()[1] <= pr_space.PR_WINDOW[1]
        extent = inset.get_window_extent()
        for xy in points:
            assert not extent.contains(*ax.transData.transform(xy))
        plt.close(fig)

    def test_two_groups_take_two_insets_and_a_third_shares_with_its_nearest(self) -> None:
        two = [(0.90, 0.90), (0.92, 0.91), (0.30, 0.95), (0.32, 0.94)]
        fig, ax, panel_points, marks_of = _inset_panel(two)
        assert [group for _inset, group in pr_space._draw_insets(ax, panel_points, marks_of)] == [[0, 1], [2, 3]]
        plt.close(fig)
        three = [*two, (0.80, 0.80), (0.82, 0.79)]
        fig, ax, panel_points, marks_of = _inset_panel(three)
        groups = [group for _inset, group in pr_space._draw_insets(ax, panel_points, marks_of)]
        assert sorted(groups) == [[0, 1, 4, 5], [2, 3]]
        plt.close(fig)

    def test_insets_are_off_by_default_and_ties_keep_their_names(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        for name, record in _PREDICTIONS["good"].items():
            _write(data_root / "atacseq" / "output" / "copy" / "arms-agent" / f"{name}.json", record)
        plot_pr_model_comp(str(data_root), ("good", "copy"), field_types=("ontology",))
        plot_pr_model_comp(str(data_root), ("good", "copy"), field_types=("ontology",), inset=True)
        for fig in captured:
            # "good" and "copy" tie exactly: nothing to enlarge, so no inset either way.
            assert len(fig.axes) == 1
            assert set(_stack_boxes(fig.axes[0])) == {"good", "copy"}

    def test_the_condition_figure_accepts_insets(self, data_root: Path, captured: list[plt.Figure]) -> None:
        """Its marks here sit far apart, so it draws, and draws no inset: the option reaches it unharmed."""
        for name, record in _PREDICTIONS["half"].items():
            _write(data_root / "atacseq" / "output" / "good" / "baseline" / f"{name}.json", record)
        plot_pr_condition_comp(str(data_root), "good", inset=True)
        assert len(captured[0].axes) == 1
