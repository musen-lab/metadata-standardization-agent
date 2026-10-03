"""A run comparison draws each run at its own precision and recall for one model."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib
import pytest
from matplotlib.colors import to_hex

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from plots import plot_pr_run_comp, pr_space  # noqa: E402
from plots.marks import CONDITION_COLOURS, MODEL_HATCHES, NO_COLOR_INK, condition_label  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


_GOLD = {"r1": {"tissue": "lung", "title": "study"}, "r2": {"tissue": "heart", "title": "x"}}
_PREDICTIONS = {
    1: _GOLD,
    2: {"r1": {"tissue": "lung", "title": "bad"}, "r2": {"tissue": "", "title": "x"}},
}
_BASELINE_PREDICTIONS = {
    1: {"r1": {"tissue": "wrong", "title": "study"}, "r2": {"tissue": "wrong", "title": "x"}},
    2: {"r1": {"tissue": "lung", "title": "study"}, "r2": {"tissue": "wrong", "title": "x"}},
}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    schema = {"children": [{"name": "tissue", "permissible_values": [{"type": "ontology"}]}, {"name": "title"}]}
    _write(tmp_path / "schemas" / "atacseq.json", schema)
    for name, gold in _GOLD.items():
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", gold)
    for condition, predictions in (("baseline", _BASELINE_PREDICTIONS), ("arms-agent", _PREDICTIONS)):
        for run, records in predictions.items():
            for name, record in records.items():
                _write(tmp_path / "atacseq" / "output" / "m" / condition / f"run-{run}" / f"{name}.json", record)
    return tmp_path


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[plt.Figure]:
    figures: list[plt.Figure] = []
    monkeypatch.setattr(pr_space, "_finish", lambda fig, _save_path: figures.append(fig))
    yield figures
    for fig in figures:
        plt.close(fig)


def _points(ax: plt.Axes) -> dict[str, list[tuple[float, float]]]:
    points: dict[str, list[tuple[float, float]]] = {}
    for line in ax.get_lines():
        if line.get_gid() != pr_space.STACK_COPY_GID and line.get_linestyle() == "None" and len(line.get_xdata()) == 1:
            colour = to_hex(line.get_markerfacecolor())
            points.setdefault(colour, []).append((float(line.get_xdata()[0]), float(line.get_ydata()[0])))
    return points


def _legend_labels(fig: plt.Figure) -> list[str]:
    return [text.get_text() for text in fig.legends[0].get_texts()]


class TestRunComparison:
    def test_each_run_has_its_own_scores_and_numbered_markers(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        plot_pr_run_comp(
            str(data_root), "m", runs=(1, 2), condition="arms-agent", field_types=("ontology", "non_ontology")
        )
        fig = captured[0]
        assert sorted(_points(fig.axes[0])[to_hex(CONDITION_COLOURS[1])]) == sorted(
            [(1.0, 1.0), (1.0, 1.0), (0.5, 1.0), (0.5, 0.5)]
        )
        numbered = [(text.get_text(), text.xy) for text in fig.axes[0].texts if text.get_text() in {"1", "2"}]
        assert sorted(numbered) == sorted([("1", (1.0, 1.0)), ("2", (0.5, 1.0)), ("2", (0.5, 0.5))])
        assert "ARMS, 1" in [text.get_text() for text in fig.axes[0].texts if text.get_gid() == pr_space.STACK_NAME_GID]
        assert _legend_labels(fig) == ["ARMS"]

    def test_baselines_and_systems_share_each_run_index(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_run_comp(str(data_root), "m", runs=(1, 2), field_types=("ontology",))
        ax = captured[0].axes[0]
        assert _legend_labels(captured[0]) == ["Baseline", "ARMS"]
        assert sorted(_points(ax)[to_hex(CONDITION_COLOURS[0])]) == sorted([(0.0, 0.0), (0.5, 0.5)])
        assert sorted(_points(ax)[to_hex(CONDITION_COLOURS[1])]) == sorted([(1.0, 1.0), (0.5, 1.0)])
        assert sorted(text.get_text() for text in ax.texts if text.get_text() in {"1", "2"}) == ["1", "1", "2", "2"]

    def test_multiple_baselines_get_distinct_marks(self, data_root: Path, captured: list[plt.Figure]) -> None:
        for run, records in _PREDICTIONS.items():
            for name, record in records.items():
                _write(data_root / "atacseq" / "output" / "m" / "template-tool" / f"run-{run}" / f"{name}.json", record)
        plot_pr_run_comp(
            str(data_root),
            "m",
            runs=(1, 2),
            baselines=("baseline", "template-tool"),
            systems=("arms-agent",),
            field_types=("ontology",),
        )
        ax = captured[0].axes[0]
        assert _legend_labels(captured[0]) == ["Baseline", condition_label("template-tool"), "ARMS"]
        assert len(_points(ax)) == 3
        assert sum(len(points) for points in _points(ax).values()) == 6

    def test_monochrome_pair_uses_hollow_and_solid_marks(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_run_comp(str(data_root), "m", runs=(1, 2), field_types=("ontology",), no_color=True)
        legend = captured[0].legends[0]
        assert [text.get_text() for text in legend.get_texts()] == ["Baseline", "ARMS"]
        assert [to_hex(handle.get_facecolor()) for handle in legend.legend_handles] == ["#ffffff", NO_COLOR_INK]
        assert [handle.get_hatch() for handle in legend.legend_handles] == [None, None]
        assert {text.get_text() for text in captured[0].axes[0].texts} >= {"1", "2"}

    def test_monochrome_multiple_conditions_use_patterns(self, data_root: Path, captured: list[plt.Figure]) -> None:
        for run, records in _PREDICTIONS.items():
            for name, record in records.items():
                _write(data_root / "atacseq" / "output" / "m" / "template-tool" / f"run-{run}" / f"{name}.json", record)
        plot_pr_run_comp(
            str(data_root),
            "m",
            runs=(1, 2),
            baselines=("baseline", "template-tool"),
            systems=("arms-agent",),
            field_types=("ontology",),
            no_color=True,
        )
        legend = captured[0].legends[0]
        assert [handle.get_hatch() for handle in legend.legend_handles] == list(MODEL_HATCHES[:3])
        assert {text.get_text() for text in captured[0].axes[0].texts} >= {"1", "2"}

    def test_unmade_run_is_omitted_from_a_comparison(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_run_comp(str(data_root), "m", runs=(1, 3, 2), condition="arms-agent", field_types=("ontology",))
        numbered = {text.get_text() for text in captured[0].axes[0].texts if text.get_text() in {"1", "2", "3"}}
        assert numbered == {"1", "2"}

    def test_marker_uses_the_actual_run_number(self, data_root: Path, captured: list[plt.Figure]) -> None:
        for name, record in _GOLD.items():
            _write(data_root / "atacseq" / "output" / "m" / "arms-agent" / "run-5" / f"{name}.json", record)
        plot_pr_run_comp(str(data_root), "m", runs=(5, 2), condition="arms-agent", field_types=("ontology",))
        ax = captured[0].axes[0]
        assert _legend_labels(captured[0]) == ["ARMS"]
        assert {text.get_text() for text in ax.texts if text.get_text() in {"5", "2"}} == {"5", "2"}

    def test_coincident_runs_have_individual_labels_and_visible_markers(
        self, data_root: Path, captured: list[plt.Figure]
    ) -> None:
        for name, record in _GOLD.items():
            _write(data_root / "atacseq" / "output" / "m" / "arms-agent" / "run-3" / f"{name}.json", record)
        plot_pr_run_comp(str(data_root), "m", runs=(1, 3), condition="arms-agent", field_types=("ontology",))
        ax = captured[0].axes[0]
        boxes = [text.get_text() for text in ax.texts if text.get_gid() == pr_space.STACK_NAME_GID]
        assert boxes == ["ARMS, 1"]
        copies = [line for line in ax.get_lines() if line.get_gid() == pr_space.STACK_COPY_GID]
        assert len(copies) == 1
        assert (float(copies[0].get_xdata()[0]), float(copies[0].get_ydata()[0])) != (1.0, 1.0)
        assert len([line for line in ax.get_lines() if line.get_gid() == pr_space.STACK_SCORE_LINK_GID]) == 1
        assert [(text.get_text(), text.xy) for text in ax.texts if text.get_text() in {"1", "3"}] == [("3", (1.0, 1.0))]

    def test_split_panels_keep_the_field_type_titles(self, data_root: Path, captured: list[plt.Figure]) -> None:
        plot_pr_run_comp(
            str(data_root),
            "m",
            runs=(1, 2),
            condition="arms-agent",
            assays=("atacseq",),
            field_types=("ontology", "non_ontology"),
            shared_window=False,
            no_color=True,
        )
        fig = captured[0]
        assert len(fig.axes) == 2
        assert [ax.get_title() for ax in fig.axes] == ["Ontology-constrained Fields", "Non-ontology-constrained Fields"]
        assert {text.get_text() for ax in fig.axes for text in ax.texts} >= {"1", "2"}

    def test_pooled_runs_must_cover_the_same_assays(self, data_root: Path, captured: list[plt.Figure]) -> None:
        _write(data_root / "schemas" / "rnaseq.json", json.loads((data_root / "schemas" / "atacseq.json").read_text()))
        for name, gold in _GOLD.items():
            _write(data_root / "rnaseq" / "gold" / f"{name}.json", gold)
            for run, records in _PREDICTIONS.items():
                _write(
                    data_root / "rnaseq" / "output" / "m" / "arms-agent" / f"run-{run}" / f"{name}.json",
                    records[name],
                )
            _write(data_root / "atacseq" / "output" / "m" / "arms-agent" / "run-3" / f"{name}.json", gold)
        plot_pr_run_comp(str(data_root), "m", runs=(1, 2, 3), condition="arms-agent", field_types=("ontology",))
        numbered = {text.get_text() for text in captured[0].axes[0].texts if text.get_text() in {"1", "2", "3"}}
        assert numbered == {"1", "2"}
        plot_pr_run_comp(
            str(data_root),
            "m",
            runs=(1, 2, 3),
            condition="arms-agent",
            assays=("atacseq", "rnaseq"),
            field_types=("ontology",),
        )
        assert len(_points(captured[1].axes[0])[to_hex(CONDITION_COLOURS[1])]) == 3
        assert len(_points(captured[1].axes[1])[to_hex(CONDITION_COLOURS[1])]) == 2

    @pytest.mark.parametrize("runs", [(), (1,), (1, 1)])
    def test_unusable_run_lists_are_rejected(self, data_root: Path, runs: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="run"):
            plot_pr_run_comp(str(data_root), "m", runs=runs)

    def test_a_model_group_is_rejected(self, data_root: Path) -> None:
        with pytest.raises(TypeError, match="single model"):
            plot_pr_run_comp(str(data_root), ("m", "other"), runs=(1, 2))  # type: ignore[arg-type]

    def test_at_least_two_runs_must_be_available(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="Fewer than two"):
            plot_pr_run_comp(str(data_root), "m", runs=(1, 4), condition="arms-agent")

    def test_condition_alias_cannot_be_combined_with_groups(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            plot_pr_run_comp(str(data_root), "m", condition="arms-agent", baselines=())
