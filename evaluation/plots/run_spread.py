"""How far each run lands from the mean of the runs: one dot per run, per assay.

The figure of :func:`~analysis.data_analysis.create_run_spread_summary`.  That table gives
each score's mean and its lowest and highest run; this draws every run, as its distance
from the mean in percentage points, so the reader sees both how wide the spread is and
whether one run is the outlier or the runs are evenly scattered.

Only the assays are drawn, never the corpus pooled over them.  The pooled score is tight by
construction -- each assay enters it in proportion to its field instances, and their
run-to-run swings partly cancel -- so beside the assays it would read as evidence that every
assay is that stable, which it is not.  The pooled spread is in the table.

The scores are the instance-weighted ones the single-run tables print, not the deduplicated
ones the significance tests use.
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.data_analysis import POOLED_ASSAY, SPREAD_METRICS, collect_run_scores
from assays import ASSAY_ORDER
from plots.marks import FIELD_TYPE_LABELS, ConditionMark, _condition_marks, _mark_style, condition_label
from plots.pr_scores import _check_pr_arguments
from plots.theme import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LEGEND_LINE_INCHES,
    LEGEND_TEXT_SIZE,
    NO_COLOR_INK,
    PANEL_FRAME_COLOUR,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    _finish,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

#: The marks are drawn at this fraction of their size in the precision/recall figures.  Two
#: conditions share an assay's row and several runs of one condition often land within a
#: tenth of a point of each other, so a full-size mark would cover its neighbours.
RUN_MARK_SCALE = 0.62

#: How far apart, as a fraction of a row, the conditions of one assay are set: enough for two
#: marks to clear each other, little enough that the assay's conditions read as one row.
CONDITION_ROW_SPAN = 0.34

#: Height allowed per assay row, and width per panel and for the assay names left of them.
RUN_ROW_INCHES = 0.42
RUN_PANEL_INCHES = 3.0
RUN_LABEL_INCHES = 1.9

#: Height the column titles, the shared x-axis label and the legend take whatever the rows hold.
RUN_CHROME_INCHES = 1.5

#: Height kept between the tick numbers and the legend for the one x-axis label the panels share.
X_LABEL_INCHES = 0.3


def plot_run_spread(
    data_root: str,
    model: str,
    *,
    baselines: tuple[str, ...] = ("baseline",),
    systems: tuple[str, ...] = ("arms-agent",),
    runs: Sequence[int] = (1, 2, 3),
    field_types: tuple[str, ...] = ("ontology", "non_ontology"),
    no_color: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    save_path: str | None = None,
) -> None:
    """Every run's precision and recall, as its distance from the mean of *runs*, per assay.

    One panel per field type and metric, one row per assay, and within a row one dot per run
    of each condition.  A dot at 0 is a run that scored exactly the mean; a row whose dots
    all sit on 0 is a score no run moved.  Distances are in percentage points, and every
    panel shares one symmetric window so a wide spread in one panel is wide in all of them.

    The scores are :func:`~analysis.data_analysis.collect_run_scores`'s: instance-weighted,
    not deduplicated.  The corpus pooled over the assays is left out; see the module notes.

    Conditions are marked as in :func:`~plots.pr_space.plot_pr_condition_comp`, colour for the
    condition or, with *no_color*, hollow for the baselines and solid for the systems.
    *title* is written above the figure.  *x_label* replaces the axis label under the
    panels, which by default names what a dot's position means.  When *save_path* is given
    the figure is written there instead of shown.

    Raises:
        ValueError: If a group is empty, a field type is unknown, *runs* holds fewer than
            two distinct runs, or no assay was scored in every run of every condition.
    """
    _check_pr_arguments(baselines, systems, field_types)
    conditions = (*baselines, *systems)
    scores = {
        condition: _deviations(collect_run_scores(data_root, model, condition, runs=runs, field_types=field_types))
        for condition in conditions
    }
    present = [set(frame["assay"]) for frame in scores.values()]
    assays = [label for _key, label in ASSAY_ORDER if all(label in assays for assays in present)]
    if not assays:
        raise ValueError(f"No assay was scored in every one of runs {list(runs)} for every one of {conditions}")
    n_records = scores[conditions[0]].drop_duplicates("assay").set_index("assay")["n_records"]

    panels = [(field_type, metric) for field_type in field_types for metric in SPREAD_METRICS]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(RUN_LABEL_INCHES + RUN_PANEL_INCHES * len(panels), RUN_ROW_INCHES * len(assays) + RUN_CHROME_INCHES),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    marks = [mark for _condition, mark in _condition_marks(baselines, systems, no_color=no_color)]
    # Top to bottom: the first assay on top, and within an assay the conditions in legend order.
    rows = {assay: float(len(assays) - 1 - index) for index, assay in enumerate(assays)}
    offsets = (
        np.linspace(CONDITION_ROW_SPAN / 2, -CONDITION_ROW_SPAN / 2, len(conditions)) if len(conditions) > 1 else [0.0]
    )

    for ax, (field_type, metric) in zip(axes, panels, strict=True):
        for condition, mark, offset in zip(conditions, marks, offsets, strict=True):
            frame = scores[condition]
            drawn = frame[(frame["field_type"] == field_type) & (frame["metric"] == metric) & frame["assay"].isin(rows)]
            style = _run_mark_style(mark)
            for assay, deviation in zip(drawn["assay"], drawn["deviation"], strict=True):
                y = rows[assay] + offset
                ax.plot(deviation, y, "o", zorder=2, **style)
                if mark.letter:
                    _letter(ax, mark, deviation, y)
        ax.axvline(0.0, color=PANEL_FRAME_COLOUR, linewidth=0.8, zorder=1)
        ax.set_title(f"{FIELD_TYPE_LABELS[field_type].capitalize()}\n{metric.capitalize()}", fontsize=PANEL_TITLE_SIZE)
        _style_axes(ax)

    reach = max(float(frame["deviation"].abs().max()) for frame in scores.values())
    # Ticks on the even points, and the window a point past the outermost one: the panels sit
    # side by side, and a tick number on each edge would run into its neighbour's.
    ticks = 2 * max(1, ceil(reach / 2))
    for ax in axes:
        ax.set_xlim(-ticks - 1, ticks + 1)
        ax.set_xticks(range(-ticks, ticks + 1, 2))
    axes[0].set_yticks(list(rows.values()), [f"{assay} (n={n_records[assay]})" for assay in assays])
    axes[0].set_ylim(-0.6, len(assays) - 0.4)

    strip = _legend(fig, conditions, marks)
    label_height = X_LABEL_INCHES / fig.get_size_inches()[1]
    fig.tight_layout(rect=(0.0, strip + label_height, 1.0, 1.0))
    # One label for every panel, since they share the axis, centred under the panels rather
    # than under the figure, whose left part is the assay names.
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text(
        (left + right) / 2,
        strip,
        x_label if x_label is not None else f"Difference from the mean of {len(runs)} runs (percentage points)",
        ha="center",
        va="bottom",
        fontsize=AXIS_LABEL_SIZE,
    )
    if title:
        # After tight_layout, which would otherwise reserve room for a title that may not be there.
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    _finish(fig, save_path)


def _deviations(scores: pd.DataFrame) -> pd.DataFrame:
    """*scores* without the pooled rows, each run's score as percentage points from its runs' mean."""
    scores = scores[scores["assay"] != POOLED_ASSAY]
    mean = scores.groupby(["assay", "field_type", "metric"])["score"].transform("mean")
    return scores.assign(deviation=100.0 * (scores["score"] - mean))


def _run_mark_style(mark: ConditionMark) -> dict[str, object]:
    """*mark*'s circle, scaled down to :data:`RUN_MARK_SCALE`."""
    style = _mark_style(mark, "o")
    return {**style, "markersize": RUN_MARK_SCALE * float(style["markersize"])}


def _letter(ax: plt.Axes, mark: ConditionMark, x: float, y: float) -> None:
    """Write *mark*'s letter inside the mark at (*x*, *y*), as the precision/recall figures do."""
    ax.annotate(
        mark.letter,
        (x, y),
        ha="center",
        va="center",
        fontsize=4.5,
        color="white" if mark.style["markerfacecolor"] == NO_COLOR_INK else NO_COLOR_INK,
        zorder=3,
    )


def _style_axes(ax: plt.Axes) -> None:
    """Recessive furniture: a vertical grid to read distances against, the frame left and below."""
    ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(color=PANEL_FRAME_COLOUR, labelsize=TICK_LABEL_SIZE, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PANEL_FRAME_COLOUR)
        ax.spines[side].set_linewidth(0.8)


def _legend(fig: plt.Figure, conditions: tuple[str, ...], marks: list[ConditionMark]) -> float:
    """One key per condition under the panels; returns the fraction of figure height it takes."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            label=f"{mark.letter}  {condition_label(condition)}" if mark.letter else condition_label(condition),
            **_mark_style(mark, "o"),
        )
        for condition, mark in zip(conditions, marks, strict=True)
    ]
    fig.legend(
        handles=handles,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=LEGEND_TEXT_SIZE,
        handletextpad=0.5,
        columnspacing=3.0,
    )
    return 1.3 * LEGEND_LINE_INCHES / fig.get_size_inches()[1]
