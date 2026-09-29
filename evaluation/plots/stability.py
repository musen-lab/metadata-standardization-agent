"""How deterministic each condition is: a column per condition, a stacked bar per assay.

The companion to :func:`~analysis.data_analysis.create_run_spread_summary`.  That table shows
how far a condition's score moves between runs; this shows how often its answers change.  Every
field instance is either consistent -- the same answer in every run -- or not, whether or
not that answer is right.  What is counted, and why, is :mod:`analysis.data_analysis.stability`.

The figure is monochrome, like the manuscript's other figures: the two bands are two greys,
dark for the same answer in every run and light for an answer that changed.  How a share
becomes a segment is :mod:`plots.segments`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.data_analysis import STABILITY_BANDS, collect_field_stability, summarize_field_stability
from assays import ASSAY_ORDER
from plots.marks import FIELD_TYPE_LABELS, condition_label
from plots.segments import _stack_row
from plots.theme import (
    AXIS_COLOUR,
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LABEL_COLOUR,
    LEGEND_LINE_INCHES,
    LEGEND_TEXT_SIZE,
    NO_COLOR_INK,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    _finish,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: One grey per band: dark for consistent, light for inconsistent.  The dark one takes white
#: lettering and the light one the label grey, each clearing 4.5:1.
STABILITY_COLOURS = ("#2f2f2b", "#d6d6cf")

#: What each band is called in the key.
STABILITY_LABELS = {
    "consistent": "same answer in every run",
    "inconsistent": "answer changed between runs",
}

#: The figure is drawn as wide as the run-spread figure beside it, whatever the number of
#: columns.  Type is set in points and the figure in inches, so two figures shown at the same
#: width only carry the same size of type when they are drawn at the same width too: drawn
#: narrower, this one's labels would come out half as large again on the page.
STABILITY_FIGURE_INCHES = 13.9

#: Height allowed per assay row, as the run-spread figure allows, so the rows keep its
#: proportions at its width.
STABILITY_ROW_INCHES = 0.42

#: Height the column titles, the shared x-axis label and the legend take whatever the rows hold.
STABILITY_CHROME_INCHES = 1.4

#: Height kept between the tick numbers and the legend for the one x-axis label the columns share.
X_LABEL_INCHES = 0.3


def plot_field_stability(
    data_root: str,
    model: str,
    *,
    conditions: tuple[str, ...] = ("baseline", "arms-agent"),
    runs: Sequence[int] = (1, 2, 3),
    field_type: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    save_path: str | None = None,
) -> None:
    """For each assay, how often each condition gives the same answer in every run.

    One column per condition, named as :data:`~plots.marks.CONDITION_LABELS` names it, and one
    100%-wide bar per assay in each.  The bar splits the condition's field instances into those
    whose answer was the same in every run in *runs* and those whose answer changed; a short
    light end means the condition is close to deterministic.  Right or wrong does not enter: a
    condition that gives the same wrong answer every time is consistent.  The columns share
    their rows, so one assay's bars are read across.

    Shares rather than counts, because the assays differ in size by more than tenfold.  A
    segment too narrow to hold its number goes unlabelled; the exact shares, and how many
    field instances each stands on, are in
    :func:`~analysis.data_analysis.summarize_field_stability`.

    *field_type* restricts every bar to ``"ontology"`` or ``"non_ontology"`` fields, and the
    axis label says which.  *title* is written above the figure.  *x_label* replaces the axis
    label under the columns.  When *save_path* is given the figure is written there instead
    of shown.

    Raises:
        ValueError: If *runs* holds fewer than two distinct runs, or no condition has
            predictions in every one of them.
    """
    tables = {
        condition: summarize_field_stability(
            collect_field_stability(data_root, model, condition, runs=runs), field_type=field_type
        ).set_index("assay")
        for condition in conditions
    }
    present = {assay for table in tables.values() for assay in table.index if table.loc[assay, "n_instances"]}
    assays = [label for _key, label in ASSAY_ORDER if label in present]
    if not assays:
        raise ValueError(f"No field instances for any of {conditions} in runs {list(runs)} under {model!r}")

    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(
            STABILITY_FIGURE_INCHES,
            STABILITY_ROW_INCHES * len(assays) + STABILITY_CHROME_INCHES,
        ),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    colours = list(STABILITY_COLOURS)
    for ax, condition in zip(axes, conditions, strict=True):
        table = tables[condition]
        for position, assay in enumerate(assays):
            if assay not in table.index or not table.loc[assay, "n_instances"]:
                continue
            shares = {band: float(table.loc[assay, band]) for band in STABILITY_BANDS}
            _stack_row(
                ax,
                position,
                shares,
                STABILITY_BANDS,
                colours,
                label_segments=True,
                height=0.72,
                label_size=TICK_LABEL_SIZE,
            )
        ax.set_title(condition_label(condition), fontsize=PANEL_TITLE_SIZE)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS_COLOUR)
        ax.tick_params(colors=LABEL_COLOUR, labelsize=TICK_LABEL_SIZE, left=False)

    axes[0].set_yticks(range(len(assays)))
    axes[0].set_yticklabels(assays, fontsize=TICK_LABEL_SIZE, color=NO_COLOR_INK)
    axes[0].set_ylim(len(assays) - 0.4, -0.6)

    strip = _legend(fig, colours)
    label_height = X_LABEL_INCHES / fig.get_size_inches()[1]
    # Room between the columns for one's "100%" and the next one's "0" to stay apart.
    fig.tight_layout(rect=(0.0, strip + label_height, 1.0, 1.0), w_pad=3.0)
    # One label for every column, since they share the axis, centred under the bars rather
    # than under the figure, whose left part is the assay names.  A restricted figure says
    # so here, since the bars look the same either way.
    counted = "field instances" if field_type is None else FIELD_TYPE_LABELS[field_type]
    left, right = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text(
        (left + right) / 2,
        strip,
        x_label if x_label is not None else f"Share of {counted}",
        ha="center",
        va="bottom",
        fontsize=AXIS_LABEL_SIZE,
        color=LABEL_COLOUR,
    )
    if title:
        # After tight_layout, which would otherwise reserve room for a title that may not be there.
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    _finish(fig, save_path)


def _legend(fig: plt.Figure, colours: list[str]) -> float:
    """One key per band under the columns, as large as the precision/recall figures' keys.

    Returns the fraction of figure height it takes.
    """
    fig.legend(
        handles=[
            Patch(facecolor=colour, edgecolor="white", label=STABILITY_LABELS[band])
            for band, colour in zip(STABILITY_BANDS, colours, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(STABILITY_BANDS),
        frameon=False,
        fontsize=LEGEND_TEXT_SIZE,
        labelcolor=LABEL_COLOUR,
        handletextpad=0.5,
        columnspacing=3.0,
    )
    return 1.3 * LEGEND_LINE_INCHES / fig.get_size_inches()[1]
