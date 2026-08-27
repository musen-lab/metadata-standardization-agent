"""Operating points in precision/recall space: one panel per assay, or one for the corpus.

The paper's main figure.  A panel is a window on the unit square with a mark per (run,
field type), which is why this module is mostly furniture: the window, the constant-F1
contours behind the marks, the two legends under them, and the two ways of laying the
panels out.  What the marks themselves look like is :mod:`plots.marks`.
"""

from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.data_analysis import (
    create_overall_precision_recall_summary,
    create_per_assay_precision_recall_summary,
)
from assays import ASSAY_ORDER
from plots.marks import (
    FIELD_KEY_COLOUR,
    FIELD_KEY_SIZE,
    FIELD_TYPE_LABELS,
    FIELD_TYPE_MARKERS,
    RunMark,
    _mark_style,
    _run_marks,
)
from plots.pr_scores import POOLED_LABEL, _check_pr_arguments
from plots.theme import (
    AXIS_LABEL_SIZE,
    CONTOUR_COLOUR,
    CONTOUR_LABEL_COLOUR,
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LEGEND_LINE_INCHES,
    LEGEND_TEXT_SIZE,
    NO_COLOR_INK,
    PANEL_CHROME_INCHES,
    PANEL_FRAME_COLOUR,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    _finish,
)

#: Every panel is the whole unit square, whatever it plots.  Cropping to the data would
#: put the origin somewhere other than the corner, and a reader would have to check the
#: ticks of every panel before believing any distance in it: a gap that looks large is
#: only large against a scale that starts at zero.  The window reaches a little past both
#: ends of that square so a run scoring 0 or 1 -- and several score 1 -- sits inside the
#: panel rather than half under its frame.
PR_WINDOW = (-0.05, 1.08)

#: Ticked at the fifths of the scale, ends included: the two corners are what the panel
#: is read between.
PR_TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: Panels per row when the field types share a window, so a twelve-assay figure comes
#: out wider than it is tall.
PANELS_PER_ROW = 4

#: Width and height allowed per panel when the field types share a window.  The marks are
#: sized in points and the panel in inches, so the two are set together: draw the panel
#: larger and the same mark reads lighter in it.  At this size a mark is about a
#: twenty-fifth of the panel's width -- big enough to tell three shapes apart in print,
#: small enough that four marks in one corner still show four marks.
SHARED_PANEL_INCHES = 3.3

#: Height allowed per assay row in the split layout.  Every panel is the same square
#: window, so the tick numbers only need printing on the outside of the grid; without
#: them repeated between every row, the rows can sit this much closer.
SPLIT_ROW_INCHES = 2.9

#: Width allowed per field-type column in the split layout.
SPLIT_COLUMN_INCHES = 3.4

#: How far left of a panel its row label sits, in points: clear of the y-axis label
#: and its tick numbers, which is what stands between them.
ROW_LABEL_OFFSET = 44


def _draw_iso_f1(
    ax: plt.Axes,
    *,
    error_axes: bool = False,
    levels: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> None:
    """Draw the constant-F1 contours a precision/recall pair can be read against.

    Along one contour every point scores the same F1 by a different trade-off, which is
    what makes "same F1, more precision, less recall" visible rather than arithmetic.
    The same curves hold when the x axis counts misses instead of recall; they are
    simply mirrored, and the label follows the end of the curve to whichever side that
    now falls on.

    Labels sit at the end of each curve, out of the way of the marks.  They are a key to
    the curves, not a scale to read a point's height against: a point is worth more than
    every contour it sits above and less than every one it sits below, which is answered
    by following a curve, never by which label is nearest.
    """
    for level in levels:
        recall = np.linspace(level / 2 + 1e-3, 1.0, 200)
        precision = level * recall / (2 * recall - level)
        inside = precision <= 1.0
        if not inside.any():
            continue
        x = 1.0 - recall if error_axes else recall
        ax.plot(x[inside], precision[inside], color=CONTOUR_COLOUR, linewidth=0.9, linestyle=(0, (2, 3)), zorder=0)
        ax.annotate(
            f"F1={level:g}",
            (x[inside][-1], precision[inside][-1]),
            fontsize=6,
            color=CONTOUR_LABEL_COLOUR,
            ha="left" if error_axes else "right",
            va="bottom",
            # The curve passes under its own label now that the contours are dark
            # enough to see; a patch of page keeps the digits readable.
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            zorder=0,
        )


def _style_pr_axes(ax: plt.Axes) -> None:
    """The window and furniture every precision/recall panel shares.

    Both axes run from 0, and the tick there is drawn, whatever the panel holds: the
    origin is the fixed point every distance in the figure is read against.

    The panel is closed on all four sides, where the other figures leave their axes open.
    A dozen panels in a grid need each cell bounded, or a mark sitting near an edge reads
    as belonging to the gap between panels rather than to a panel; a bar chart with
    nothing beside it does not.
    """
    ax.set_xlim(*PR_WINDOW)
    ax.set_ylim(*PR_WINDOW)
    ax.set_xticks(PR_TICKS)
    ax.set_yticks(PR_TICKS)
    ax.set_aspect("equal")  # equal axes: a step sideways means what a step up means
    ax.grid(color=GRID_COLOUR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(color=PANEL_FRAME_COLOUR, labelsize=TICK_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_color(PANEL_FRAME_COLOUR)
        spine.set_linewidth(0.8)


def _draw_pr_path(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    marks: list[RunMark],
    marker: str = "o",
) -> None:
    """Draw one group's operating points.

    The points are not joined.  A line between them reads as a path something travelled,
    and these are separate systems measured once each -- nothing lies between two of them
    to trace.  The order within a group is carried by the colour ramp, which says the runs
    are ordered without claiming anything about the space between them.
    """
    for (recall, precision), mark in zip(points, marks, strict=True):
        ax.plot(recall, precision, marker, zorder=2, **_mark_style(mark, marker))
        if mark.letter:
            ax.annotate(
                mark.letter,
                (recall, precision),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if mark.style["markerfacecolor"] == NO_COLOR_INK else NO_COLOR_INK,
                zorder=3,
            )


def _pr_panel_series(
    ax: plt.Axes,
    series: list[tuple[list[tuple[float, float]], list[RunMark], str]],
    *,
    error_axes: bool = False,
    show_f1_contours: bool = True,
) -> None:
    """Draw every (points, colours, marker) series of one panel, over shared contours."""
    if show_f1_contours:
        _draw_iso_f1(ax, error_axes=error_axes)
    for points, marks, marker in series:
        _draw_pr_path(ax, points, marks, marker)
    _style_pr_axes(ax)


def _pr_legends(
    fig: plt.Figure,
    baseline_runs: tuple[str, ...],
    system_runs: tuple[str, ...],
    field_types: tuple[str, ...],
    *,
    show_field_keys: bool = True,
    no_color: bool = False,
) -> float:
    """Colour for the run, and -- when a panel holds more than one -- marker for the field type.

    Stacked rather than side by side: on one line the run names and the field-type names
    collide as soon as either list grows.  *show_field_keys* is ``False`` when every
    panel holds a single field type and says so in its own title, where a key would only
    repeat what the reader has already been told.  Returns the fraction of figure height
    to keep clear for the keys, a constant physical size however tall the figure is.
    """
    # The letter goes in the label rather than inside the key's marker: with the sizes
    # gone, three hollow circles would otherwise be three identical keys.
    run_keys = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            label=f"{mark.letter}  {run}" if mark.letter else run,
            **_mark_style(mark, "o"),
        )
        for run, mark in _run_marks(baseline_runs, system_runs, no_color=no_color)
    ]
    field_keys = (
        []
        if not show_field_keys
        else [
            Line2D(
                [],
                [],
                marker=FIELD_TYPE_MARKERS[field_type],
                linestyle="",
                markersize=FIELD_KEY_SIZE,
                # Black, not grey: the key has to show the shape, and the shape is the whole
                # message here -- colour in this legend would say something it does not mean.
                color=FIELD_KEY_COLOUR,
                # No white rim, unlike the marks in the panels: nothing overlaps a key, so a
                # rim would only make the shape read thinner than the shape it stands for.
                markeredgecolor=FIELD_KEY_COLOUR,
                label=FIELD_TYPE_LABELS[field_type],
            )
            for field_type in field_types
        ]
    )

    # The keys are set well apart: a two-column row of two names centred under a figure a
    # foot wide otherwise reads as one long label rather than as two.
    style: dict[str, object] = {"frameon": False, "handletextpad": 0.5, "loc": "lower center"}
    line = LEGEND_LINE_INCHES / fig.get_size_inches()[1]
    if not show_field_keys:
        fig.legend(handles=run_keys, ncol=len(run_keys), bbox_to_anchor=(0.5, 0.0), fontsize=LEGEND_TEXT_SIZE, **style)
        return 1.3 * line

    fig.legend(
        handles=run_keys,
        ncol=len(run_keys),
        bbox_to_anchor=(0.5, 1.05 * line),
        fontsize=LEGEND_TEXT_SIZE,
        columnspacing=3.5,
        **style,
    )
    fig.legend(
        handles=field_keys,
        ncol=len(field_keys),
        bbox_to_anchor=(0.5, 0.0),
        fontsize=LEGEND_TEXT_SIZE - 0.5,
        columnspacing=3.0,
        **style,
    )
    return 2.4 * line


def plot_pr_space(
    data_root: str,
    model: str,
    *,
    assays: tuple[str, ...] = (),
    baseline_runs: tuple[str, ...] = ("baseline",),
    system_runs: tuple[str, ...] = ("arms-agent",),
    field_types: tuple[str, ...] = ("ontology", "non_ontology", "all"),
    shared_window: bool = True,
    error_axes: bool = False,
    show_f1_contours: bool = True,
    no_color: bool = False,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Operating points in precision/recall space.

    Each run is one **operating point**, not a curve: the runs emit a value or leave a
    field empty, with no score to threshold, so there is nothing to sweep.  A group of
    several runs is joined into a path in the order given -- read it as a progression
    between discrete systems, never as a frontier that could be tuned along -- and a
    group of one is drawn as a lone coordinate, since one point has no progression.

    So ``baseline_runs=("baseline",)`` against ``system_runs=("arms-agent",)`` draws the
    head-to-head comparison, while several runs on a side -- repetitions of one
    condition, say -- draw a progression with the other beside it.

    *assays* names the assays to draw, by the keys ``ASSAY_ORDER`` uses (``"atacseq"``,
    ``"rnaseq"``, ...).  **Left empty, the corpus is pooled into one set of panels**
    rather than broken out -- pooled over every gold/prediction pair, not averaged over
    per-assay ratios, since the assays differ in size by more than tenfold.

    *error_axes* replaces recall on the x axis with the **miss rate**, ``1 - recall``:
    the share of the values gold asks for that the run did not produce.  Lower is then
    better, so the good corner moves from the right to the left, and the axis is anchored
    at 0 and labelled there -- an error axis that starts anywhere else hides how far from
    perfect a run is, and makes two runs look further apart than they are.  Precision is
    left as it is, so a panel reads up-and-left.  The F1 contours are the same curves,
    mirrored.

    *show_f1_contours* draws the constant-F1 curves behind the points.  They are what
    makes two runs with different trade-offs rankable by eye; turn them off when the
    panel is crowded enough that they compete with the marks rather than support them.

    *no_color* draws the figure without colour: the condition becomes the marker's
    fill, hollow against solid, so a pair keeps one shape and stays directly comparable;
    the field type keeps the shape it already had; and a run's place in its group becomes
    a letter written inside the mark.  What survives greyscale, print and colour-vision
    deficiency is what carries the meaning.

    *title* is written centred above the figure, a step larger than the panel titles it
    sits over so the hierarchy is legible at a glance rather than by measurement.

    *shared_window* decides what a panel holds:

    * ``True`` -- every field type shares one window, told apart by marker, so a panel
      is an assay (or the pooled corpus) and the panels wrap three to a row.
    * ``False`` -- each field type takes a column and each assay a row, so a panel holds
      one field type of one assay.  Wider, but nothing overlaps.

    Colour is the run -- one hue per group, in monotone lightness steps when a group has
    several runs -- and the marker is the field type.  Constant-F1 contours sit behind
    the points: two runs on the same contour reached the same F1 by a different
    trade-off.  Every panel shares one window, so they stay comparable rather than each
    rescaling to its own data.  When *save_path* is given the figure is written there
    (PNG/PDF inferred from the extension) instead of shown interactively.
    """
    _check_pr_arguments(baseline_runs, system_runs, field_types)
    runs = (*baseline_runs, *system_runs)
    labels = dict(ASSAY_ORDER)
    unknown = [key for key in assays if key not in labels]
    if unknown:
        raise ValueError(f"Unknown assay key(s): {unknown}")

    if assays:
        frames = {
            (run, field_type): create_per_assay_precision_recall_summary(
                data_root, model, run, category=field_type
            ).set_index("assay")
            for run in runs
            for field_type in field_types
        }
        wanted = {labels[key] for key in assays}
        rows = [
            label
            for _key, label in ASSAY_ORDER
            if label in wanted and all(label in frame.index for frame in frames.values())
        ]
        if not rows:
            raise ValueError(f"No requested assay has predictions for every one of {runs}")

        def score(run: str, field_type: str, row: str) -> tuple[float, float]:
            frame = frames[(run, field_type)]
            return frame.loc[row, "recall"], frame.loc[row, "precision"]

        first = frames[(runs[0], field_types[0])]
        n_records = {row: int(first.loc[row, "n_records"]) for row in rows}
    else:
        summaries = {
            run: create_overall_precision_recall_summary(data_root, model, run).set_index("category") for run in runs
        }
        rows = [POOLED_LABEL]

        def score(run: str, field_type: str, _row: str) -> tuple[float, float]:
            summary = summaries[run]
            return summary.loc[field_type, "recall"], summary.loc[field_type, "precision"]

        n_records = {POOLED_LABEL: int(summaries[runs[0]].loc[field_types[0], "n_records"])}

    def panel_title(row: str) -> str:
        """The row's name with the records standing behind it.

        A panel drawn from 15 records and one drawn from 100 look identical otherwise,
        and the difference decides how much either is worth reading into.
        """
        return f"{row} (n={n_records[row]})"

    marks = [mark for _run, mark in _run_marks(baseline_runs, system_runs, no_color=no_color)]
    groups = ((baseline_runs, marks[: len(baseline_runs)]), (system_runs, marks[len(baseline_runs) :]))

    def placed(run: str, field_type: str, row: str) -> tuple[float, float]:
        """One run's point, with recall turned into its miss rate when asked for."""
        recall, precision = score(run, field_type, row)
        return (1.0 - recall if error_axes else recall, precision)

    def series(row: str, drawn: tuple[str, ...]) -> list[tuple[list[tuple[float, float]], list[str], str]]:
        """The (points, colours, marker) series of one panel.

        Group before field type, so a whole condition is laid down before the next one
        starts.  Where two conditions land on the same spot the system group is then the
        one left legible, rather than whichever field type happened to be drawn last.
        """
        return [
            ([placed(run, field_type, row) for run in group], group_marks, FIELD_TYPE_MARKERS[field_type])
            for group, group_marks in groups
            for field_type in drawn
        ]

    if shared_window:
        columns = min(PANELS_PER_ROW, len(rows))
        grid_rows = ceil(len(rows) / columns)
        fig, axes = plt.subplots(
            grid_rows,
            columns,
            figsize=(SHARED_PANEL_INCHES * columns, SHARED_PANEL_INCHES * grid_rows + PANEL_CHROME_INCHES),
            squeeze=False,
            # Every panel is the same window, so the tick numbers belong on the outside of
            # the grid only.  Without them repeated down every column the panels sit closer
            # and each one is drawn larger for the same page.
            sharex=True,
            sharey=True,
        )
        flat = [ax for grid_row in axes for ax in grid_row]
        for ax, row in zip(flat, rows, strict=False):
            _pr_panel_series(ax, series(row, field_types), error_axes=error_axes, show_f1_contours=show_f1_contours)
            ax.set_title(panel_title(row), fontsize=PANEL_TITLE_SIZE)
        for ax in flat[len(rows) :]:
            ax.set_visible(False)
        for grid_row in axes:
            grid_row[0].set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE)
    else:
        columns = len(field_types)
        grid_rows = len(rows)
        fig, axes = plt.subplots(
            grid_rows,
            columns,
            figsize=(SPLIT_COLUMN_INCHES * columns, SPLIT_ROW_INCHES * grid_rows + PANEL_CHROME_INCHES),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row_index, row in enumerate(rows):
            for column, field_type in enumerate(field_types):
                ax = axes[row_index][column]
                _pr_panel_series(
                    ax, series(row, (field_type,)), error_axes=error_axes, show_f1_contours=show_f1_contours
                )
                if row_index == 0:
                    ax.set_title(FIELD_TYPE_LABELS[field_type].capitalize(), fontsize=PANEL_TITLE_SIZE)
            # "Precision" belongs against the axis it measures; the assay names the whole
            # row, so it sits further out, turned to run with the row's height.
            first = axes[row_index][0]
            first.set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE)
            first.annotate(
                panel_title(row),
                xy=(0.0, 0.5),
                xycoords="axes fraction",
                xytext=(-ROW_LABEL_OFFSET, 0),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="center",
                fontsize=PANEL_TITLE_SIZE,
            )

    x_label = "Misses (1 - Recall)" if error_axes else "Recall"
    # The bottom *visible* axis of each column carries the label: a short last row
    # leaves some columns without an axis in axes[-1] at all.
    for column in range(columns):
        for grid_row in reversed(range(len(axes))):
            ax = axes[grid_row][column]
            if ax.get_visible():
                ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
                # Sharing the x axis prints the tick numbers on the bottom row alone.  A
                # short last row leaves this column ending a row higher, so they go back on
                # -- an axis labelled "Recall" with no numbers under it is unreadable.
                ax.tick_params(labelbottom=True)
                break

    # With one field type per panel, named in the column title, a marker key would only
    # repeat it.
    strip = _pr_legends(fig, baseline_runs, system_runs, field_types, show_field_keys=shared_window, no_color=no_color)
    fig.tight_layout(rect=(0.0, strip, 1.0, 1.0))
    if title:
        # After tight_layout, which does not know about a suptitle added later: adding it
        # first would have the layout reserve the space and then leave a gap when there
        # is no title.
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    _finish(fig, save_path)
