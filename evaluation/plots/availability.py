"""Where migration is transcription and where it is interpretation.

Four panels crossing the field type against where gold's value was, each holding one mark
per assay against the group's pooled rate.  The figure exists to show that the four kinds
of work are not equally hard, so its whole design is comparison down a column: one window,
one scale, and the pooled rate drawn as the line every mark is read against.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from analysis.data_analysis import DERIVED, VERBATIM, create_availability_summary, pool_availability
from assays import ASSAY_ORDER
from plots.marks import CONDITION_COLOURS, FIELD_TYPE_LABELS
from plots.theme import (
    AXIS_COLOUR,
    CONTOUR_COLOUR,
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LABEL_COLOUR,
    PANEL_CHROME_INCHES,
    STEM_COLOUR,
    _finish,
)

#: A rate over a cell this small is a coincidence away from any number, so those marks are
#: drawn hollow.  Kept as a drawing decision rather than a filter: dropping the assay would
#: hide that it has fields of that kind at all, which is itself part of the picture.
MIN_TRUSTED_CELL = 30

#: Height allowed per assay row inside one difficulty panel, in inches.  Twelve rows of
#: text need about this much to stay legible without the panel growing taller than wide.
DIFFICULTY_ROW_INCHES = 0.24

#: Width allowed per column of panels.  Every panel spans the same 0-1 window, so they can
#: sit this close without their tick numbers running together.
DIFFICULTY_COLUMN_INCHES = 3.6

#: How far left of the first panel its field-type label sits, in points.  Measured against
#: the longest assay name rather than shared with the precision/recall figures, which put
#: their row labels outside a column of tick numbers instead of a column of names.
DIFFICULTY_ROW_LABEL_OFFSET = 104

#: Height reserved under the panels for the note explaining the two marks that carry
#: meaning of their own -- the reference line and the hollow fill.  A figure read in a
#: paper is separated from its docstring, so what the marks mean travels with it.
DIFFICULTY_NOTE_INCHES = 0.32

#: Where gold's value was, in the order the columns are drawn: the harder half first, so
#: the panel a reader looks at first is the one the figure is about.
AVAILABILITY_ORDER = (DERIVED, VERBATIM)

AVAILABILITY_TITLES = {
    DERIVED: "Gold's value not in the record",
    VERBATIM: "Gold's value in the record",
}


def _difficulty_panel(
    ax: plt.Axes,
    rates: dict[str, tuple[float, int]],
    labels: list[str],
    pooled: float,
) -> None:
    """One panel: a mark per assay against the group's pooled rate.

    *rates* maps an assay label to its ``(rate, n)``; an assay missing from it has no
    fields of this kind and gets a gap rather than a zero, which would read as a rate.
    """
    positions = range(len(labels))
    ax.axvline(pooled, color=CONTOUR_COLOUR, linestyle="--", linewidth=1.0, zorder=1)

    for position, label in zip(positions, labels, strict=True):
        if label not in rates:
            continue
        rate, count = rates[label]
        ax.plot([0.0, rate], [position, position], color=STEM_COLOUR, linewidth=1.2, zorder=2)
        # Hollow for a cell too small to carry a rate: same position, no claim of weight.
        trusted = count >= MIN_TRUSTED_CELL
        ax.plot(
            rate,
            position,
            marker="o",
            markersize=7,
            color=CONDITION_COLOURS[1] if trusted else "white",
            markeredgecolor=CONDITION_COLOURS[1],
            markeredgewidth=1.4,
            zorder=3,
        )

    ax.set_xlim(0.0, 1.04)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(list(positions))
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOUR)
    ax.tick_params(colors=LABEL_COLOUR, labelsize=8)


def plot_availability_difficulty(
    data_root: str,
    model: str,
    run_type: str = "arms-agent",
    *,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """How often the run gets a field right, by field type and by where gold's value was.

    Four panels, one per (field type, availability) group, each holding one mark per assay
    against the group's pooled rate.  The point of the crossing is that the four kinds of
    work are not equally hard: transcribing a value the record already carries is not the
    same job as producing one it does not, and the ontology-constrained half of the schema
    is not the same job as the rest.

    Read the split as difficulty, not as a bound.  A gold value the record does not carry
    is not unreachable -- the runs produce most of them -- so a low rate in those panels
    says the work is harder there, not that it was impossible.

    A hollow mark is a group of fewer than :data:`MIN_TRUSTED_CELL` values, where the rate
    is too thin to argue with; an assay with no fields of a kind is absent from its panel.
    """
    summary = create_availability_summary(data_root, model, run_type)
    if summary.empty:
        raise ValueError(f"no predictions on disk for {run_type!r} under {model!r}")

    field_types = ("ontology", "non_ontology")
    labels = [label for _key, label in ASSAY_ORDER if label in set(summary["assay"])]

    fig, axes = plt.subplots(
        len(field_types),
        len(AVAILABILITY_ORDER),
        figsize=(
            DIFFICULTY_COLUMN_INCHES * len(AVAILABILITY_ORDER) + 1.9,
            DIFFICULTY_ROW_INCHES * len(labels) * len(field_types) + PANEL_CHROME_INCHES + DIFFICULTY_NOTE_INCHES,
        ),
        squeeze=False,
    )

    for row_index, field_type in enumerate(field_types):
        for column_index, availability in enumerate(AVAILABILITY_ORDER):
            selected = summary[(summary["field_type"] == field_type) & (summary["availability"] == availability)]
            rates = {
                row["assay"]: (row["correct_rate"], int(row["n_gold_values"])) for _index, row in selected.iterrows()
            }
            pooled = pool_availability(summary, field_type, availability)
            ax = axes[row_index][column_index]
            _difficulty_panel(ax, rates, labels, pooled["correct_rate"])

            # The pooled rate names the panel: it is the number the marks are read against,
            # so it belongs where the reader is already looking rather than in a legend.
            ax.set_title(
                f"{AVAILABILITY_TITLES[availability]}\npooled {pooled['correct_rate']:.3f}"
                f"  (n={pooled['n_gold_values']:,})",
                fontsize=9,
                color=LABEL_COLOUR,
            )
            if column_index == 0:
                ax.set_yticklabels(labels, fontsize=8)
                ax.annotate(
                    FIELD_TYPE_LABELS[field_type].capitalize(),
                    xy=(0.0, 0.5),
                    xycoords="axes fraction",
                    xytext=(-DIFFICULTY_ROW_LABEL_OFFSET, 0),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            else:
                ax.set_yticklabels([])
            if row_index == len(field_types) - 1:
                ax.set_xlabel("Fields the run got right", fontsize=9)

    note = DIFFICULTY_NOTE_INCHES / fig.get_size_inches()[1]
    fig.text(
        0.5,
        note / 3,
        f"Dashed line: the panel's pooled rate.  Hollow mark: fewer than {MIN_TRUSTED_CELL} gold values, "
        "too few to read as a rate.",
        ha="center",
        va="center",
        fontsize=8,
        color=LABEL_COLOUR,
    )
    fig.tight_layout(rect=(0.0, note, 1.0, 1.0))
    if title:
        # After tight_layout, which does not know about a suptitle added later.
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    _finish(fig, save_path)
