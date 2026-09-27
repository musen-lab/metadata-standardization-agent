"""How deterministic each condition is: one stacked bar per condition, per assay.

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
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LABEL_COLOUR,
    NO_COLOR_INK,
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

#: Height allowed per bar, and the air between one assay's bars and the next assay's.
STABILITY_BAR_INCHES = 0.26
ASSAY_GAP = 0.6

#: Width of the bars and of the strip of names to their left.
STABILITY_PANEL_INCHES = 5.2
STABILITY_LABEL_INCHES = 1.9

#: How far left of the bars an assay's name sits, in points: clear of the condition names,
#: which sit between them.
ASSAY_LABEL_OFFSET = 58


def plot_field_stability(
    data_root: str,
    model: str,
    *,
    conditions: tuple[str, ...] = ("baseline", "arms-agent"),
    runs: Sequence[int] = (1, 2, 3),
    field_type: str | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """For each assay, how often each condition gives the same answer in every run.

    One 100%-wide bar per condition within each assay.  The bar splits the condition's field
    instances into those whose answer was the same in every run in *runs* and those whose
    answer changed; a short light end means the condition is close to deterministic.  Right
    or wrong does not enter: a condition that gives the same wrong answer every time is
    consistent.  Conditions are named as :data:`~plots.marks.CONDITION_LABELS` names them.

    Shares rather than counts, because the assays differ in size by more than tenfold.  A
    segment too narrow to hold its number goes unlabelled; the exact shares, and how many
    field instances each stands on, are in
    :func:`~analysis.data_analysis.summarize_field_stability`.

    *field_type* restricts every bar to ``"ontology"`` or ``"non_ontology"`` fields, and the
    axis says which.  When *save_path* is given the figure is written there instead of shown.

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

    # Positions top to bottom, each assay's bars together.
    rows: list[tuple[float, str, str]] = []
    position = 0.0
    for group_index, assay in enumerate(assays):
        if group_index:
            position += ASSAY_GAP
        for condition in conditions:
            rows.append((position, assay, condition))
            position += 1.0

    fig, ax = plt.subplots(
        figsize=(STABILITY_PANEL_INCHES + STABILITY_LABEL_INCHES, STABILITY_BAR_INCHES * position + 1.4)
    )
    colours = list(STABILITY_COLOURS)
    for position, assay, condition in rows:
        table = tables[condition]
        if assay not in table.index or not table.loc[assay, "n_instances"]:
            continue
        shares = {band: float(table.loc[assay, band]) for band in STABILITY_BANDS}
        _stack_row(ax, position, shares, STABILITY_BANDS, colours, label_segments=True, height=0.78)

    positions = [position for position, _assay, _condition in rows]
    ax.set_yticks(positions)
    ax.set_yticklabels([condition_label(condition) for _position, _assay, condition in rows], fontsize=8)
    transform = ax.get_yaxis_transform()
    for assay in assays:
        group = [position for position, name, _condition in rows if name == assay]
        ax.annotate(
            assay,
            xy=(0.0, (group[0] + group[-1]) / 2),
            xycoords=transform,
            xytext=(-ASSAY_LABEL_OFFSET, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=NO_COLOR_INK,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(max(positions) + 0.7, -0.7)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOUR)
    ax.tick_params(colors=LABEL_COLOUR, labelsize=8, left=False)
    # A restricted figure says so on its axis, since the bars look the same either way.
    counted = "field instances" if field_type is None else FIELD_TYPE_LABELS[field_type]
    ax.set_xlabel(f"Share of {counted}", fontsize=9, color=LABEL_COLOUR)

    ax.legend(
        handles=[
            Patch(facecolor=colour, edgecolor="white", label=STABILITY_LABELS[band])
            for band, colour in zip(STABILITY_BANDS, colours, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(STABILITY_BANDS),
        frameon=False,
        fontsize=8.5,
        labelcolor=LABEL_COLOUR,
        handlelength=1.4,
        columnspacing=1.4,
    )

    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    _finish(fig, save_path)
