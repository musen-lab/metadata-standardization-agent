"""Operating points in precision/recall space: one panel per assay, or one for the corpus.

The paper's main figure.  A panel is a window on the unit square with a mark per (condition,
field type) -- or, in :func:`plot_pr_model_comp`, per (model, field type) under one condition --
which is why this module is mostly furniture: the window, the constant-F1 contours behind
the marks, the two legends under them, and the two ways of laying the panels out.  What the
marks themselves look like is :mod:`plots.marks`.
"""

from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.inset import InsetIndicator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

from analysis.corpus import get_assay, iter_assays
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
    MARKER_ZORDER,
    MODEL_HATCH_LINE_WIDTH,
    NO_COLOR_EDGE_WIDTH,
    ConditionMark,
    _condition_marks,
    _mark_style,
    _model_marks,
    condition_label,
)
from plots.pr_scores import POOLED_LABEL, _check_pr_arguments
from plots.theme import (
    AXIS_LABEL_SIZE,
    CONTOUR_COLOUR,
    CONTOUR_LABEL_COLOUR,
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LABEL_COLOUR,
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
#: ends of that square so a condition scoring 0 or 1 -- and several score 1 -- sits inside the
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

#: Keys per row in the legend of what the points compare: four names fit under the narrowest
#: figure without crowding, and eight models wrap to two even rows.
RUN_KEYS_PER_ROW = 4

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
    marks: list[ConditionMark],
    marker: str = "o",
) -> None:
    """Draw one group's operating points.

    The points are not joined.  A line between them reads as a path something travelled,
    and these are separate systems measured once each -- nothing lies between two of them
    to trace.  The order within a group is carried by the colour ramp, which says the conditions
    are ordered without claiming anything about the space between them.
    """
    zorder = MARKER_ZORDER.get(marker, 2.0)
    for (recall, precision), mark in zip(points, marks, strict=True):
        style = _mark_style(mark, marker)
        if mark.halo:
            # A ring of page under the mark, at the mark's own depth and drawn just before it, so
            # it parts this mark from the ones beneath without covering any drawn after it.
            ax.plot(
                recall,
                precision,
                marker,
                markersize=style["markersize"] + 2 * NO_COLOR_EDGE_WIDTH,
                color="white",
                markeredgewidth=0,
                zorder=zorder,
            )
        if mark.hatch:
            # A line cannot carry a pattern; a one-point scatter can.  Its size is an area.
            dot = ax.scatter(
                [recall],
                [precision],
                s=style["markersize"] ** 2,
                marker=marker,
                facecolors=style["markerfacecolor"],
                edgecolors=style["markeredgecolor"],
                linewidths=style["markeredgewidth"],
                hatch=mark.hatch,
                zorder=zorder,
            )
            dot.set_hatch_linewidth(MODEL_HATCH_LINE_WIDTH)
        else:
            ax.plot(recall, precision, marker, zorder=zorder, **style)
        if mark.letter:
            ax.annotate(
                mark.letter,
                (recall, precision),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if mark.style["markerfacecolor"] == NO_COLOR_INK else NO_COLOR_INK,
                # Just over its own mark and under the next shape forward, so a letter is
                # covered along with the mark it names rather than showing through.
                zorder=zorder + 0.05,
            )


def _pr_panel_series(
    ax: plt.Axes,
    series: list[tuple[list[tuple[float, float]], list[ConditionMark], str]],
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
    keys: list[tuple[str, ConditionMark]],
    field_types: tuple[str, ...],
    *,
    show_field_keys: bool = True,
) -> float:
    """Colour for what the points compare, and -- when a panel holds more than one -- marker for the field type.

    *keys* is one (label, mark) per thing compared: a condition in :func:`plot_pr_condition_comp`, a
    model in :func:`plot_pr_model_comp`.

    Stacked rather than side by side: on one line the condition names and the field-type names
    collide as soon as either list grows.  *show_field_keys* is ``False`` when every
    panel holds a single field type and says so in its own title, where a key would only
    repeat what the reader has already been told.  Returns the fraction of figure height
    to keep clear for the keys, a constant physical size however tall the figure is.
    """
    # A swatch, not a marker: every marker shape in the panels stands for a field type, so a
    # circle here would read as "ontology-constrained" before it read as a colour.  The
    # swatch keeps what the key does mean -- the colour, or with colour off the fill, hollow,
    # solid or patterned -- and the letter goes in the label, where a swatch has no room for it.
    run_keys = [
        Patch(
            facecolor=mark.style["markerfacecolor"],
            # A hollow key keeps its dark rim; a solid one is bordered in its own fill,
            # since the white seam its mark carries in a panel would thin it here.
            edgecolor=mark.style["markerfacecolor"] if mark.fill is not False else mark.style["markeredgecolor"],
            linewidth=NO_COLOR_EDGE_WIDTH,
            # With colour off a model is its pattern, so its key carries the same one.
            hatch=mark.hatch,
            hatch_linewidth=MODEL_HATCH_LINE_WIDTH,
            label=f"{mark.letter}  {label}" if mark.letter else label,
        )
        for label, mark in keys
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
    style: dict[str, object] = {
        "frameon": False,
        "handletextpad": 0.5,
        "loc": "lower center",
    }
    line = LEGEND_LINE_INCHES / fig.get_size_inches()[1]
    if not show_field_keys:
        rows = _run_legend(fig, run_keys, 0.0, {"fontsize": LEGEND_TEXT_SIZE, **style})
        return (0.3 + rows) * line

    rows = _run_legend(fig, run_keys, 1.05 * line, {"fontsize": LEGEND_TEXT_SIZE, "columnspacing": 3.5, **style})
    fig.legend(
        handles=field_keys,
        ncol=len(field_keys),
        bbox_to_anchor=(0.5, 0.0),
        fontsize=LEGEND_TEXT_SIZE - 0.5,
        columnspacing=3.0,
        **style,
    )
    return (1.4 + rows) * line


def _run_legend(fig: plt.Figure, keys: list[Patch], bottom: float, style: dict[str, object]) -> int:
    """Lay the keys of what is compared out in rows under the figure, and return how many rows.

    Up to :data:`RUN_KEYS_PER_ROW` keys to a row, and fewer when a row that long would run
    past the figure's edges -- a narrow figure of eight models would otherwise lose the
    names at both ends.  The keys read across each row in the order given.
    """
    renderer = fig.canvas.get_renderer()
    for columns in range(min(len(keys), RUN_KEYS_PER_ROW), 0, -1):
        rows = ceil(len(keys) / columns)
        # A legend fills its columns top to bottom; handed over column by column, the keys
        # read across each row instead.
        ordered = [
            keys[row * columns + column]
            for column in range(columns)
            for row in range(rows)
            if row * columns + column < len(keys)
        ]
        legend = fig.legend(handles=ordered, ncol=columns, bbox_to_anchor=(0.5, bottom), **style)
        if columns == 1 or legend.get_window_extent(renderer).width <= fig.bbox.width:
            return rows
        legend.remove()
    raise AssertionError("unreachable: one column always fits")


def plot_pr_condition_comp(
    data_root: str,
    model: str,
    *,
    assays: tuple[str, ...] = (),
    baselines: tuple[str, ...] = ("baseline",),
    systems: tuple[str, ...] = ("arms-agent",),
    field_types: tuple[str, ...] = ("ontology", "non_ontology", "all"),
    shared_window: bool = True,
    error_axes: bool = False,
    show_f1_contours: bool = True,
    no_color: bool = False,
    inset: bool = False,
    title: str | None = None,
    save_path: str | None = None,
    run: int = 1,
) -> None:
    """Operating points in precision/recall space.

    Each condition is one **operating point**, not a curve: the conditions emit a value or leave a
    field empty, with no score to threshold, so there is nothing to sweep.  A group of
    several conditions is joined into a path in the order given -- read it as a progression
    between discrete systems, never as a frontier that could be tuned along -- and a
    group of one is drawn as a lone coordinate, since one point has no progression.

    So ``baselines=("baseline",)`` against ``systems=("arms-agent",)`` draws the
    head-to-head comparison, while several conditions on a side -- the ablations
    leading up to ARMS, say -- draw a progression with the other beside it.

    *assays* names the assays to draw, by the keys ``ASSAY_ORDER`` uses (``"atacseq"``,
    ``"rnaseq"``, ...).  **Left empty, the corpus is pooled into one set of panels**
    rather than broken out -- pooled over every gold/prediction pair, not averaged over
    per-assay ratios, since the assays differ in size by more than tenfold.

    *error_axes* replaces recall on the x axis with the **miss rate**, ``1 - recall``:
    the share of the values gold asks for that the condition did not produce.  Lower is then
    better, so the good corner moves from the right to the left, and the axis is anchored
    at 0 and labelled there -- an error axis that starts anywhere else hides how far from
    perfect a condition is, and makes two conditions look further apart than they are.  Precision is
    left as it is, so a panel reads up-and-left.  The F1 contours are the same curves,
    mirrored.

    *show_f1_contours* draws the constant-F1 curves behind the points.  They are what
    makes two conditions with different trade-offs rankable by eye; turn them off when the
    panel is crowded enough that they compete with the marks rather than support them.

    *no_color* draws the figure without colour: the condition becomes the marker's
    fill, hollow against solid, so a pair keeps one shape and stays directly comparable;
    the field type keeps the shape it already had; and a condition's place in its group becomes
    a letter written inside the mark.  What survives greyscale, print and colour-vision
    deficiency is what carries the meaning.

    *inset* draws marks that crowd together at different scores again, enlarged, in an inset of
    their panel, with a frame round the patch it enlarges and dashed lines joining the two: a
    difference of a point or two of precision is real but hides under marks a dozen points
    wide.  Marks at exactly one score are not enlarged, since no zoom can part them.

    *title* is written centred above the figure, a step larger than the panel titles it
    sits over so the hierarchy is legible at a glance rather than by measurement.

    *shared_window* decides what a panel holds:

    * ``True`` -- every field type shares one window, told apart by marker, so a panel
      is an assay (or the pooled corpus) and the panels wrap three to a row.
    * ``False`` -- each field type takes a column and each assay a row, so a panel holds
      one field type of one assay.  Wider, but nothing overlaps.

    Colour is the condition -- one hue per group, in monotone lightness steps when a group has
    several conditions -- and the marker is the field type.  Constant-F1 contours sit behind
    the points: two conditions on the same contour reached the same F1 by a different
    trade-off.  Every panel shares one window, so they stay comparable rather than each
    rescaling to its own data.  When *save_path* is given the figure is written there
    (PNG/PDF inferred from the extension) instead of shown interactively.
    """
    _check_pr_arguments(baselines, systems, field_types)
    conditions = (*baselines, *systems)
    marked = _condition_marks(baselines, systems, no_color=no_color)
    marks = [mark for _condition, mark in marked]
    indices = list(range(len(conditions)))
    _plot_pr_points(
        data_root,
        [(model, condition) for condition in conditions],
        [
            (indices[: len(baselines)], marks[: len(baselines)]),
            (indices[len(baselines) :], marks[len(baselines) :]),
        ],
        [(condition_label(condition), mark) for condition, mark in marked],
        assays=assays,
        field_types=field_types,
        shared_window=shared_window,
        error_axes=error_axes,
        show_f1_contours=show_f1_contours,
        title=title,
        save_path=save_path,
        run=run,
        inset=inset,
    )


def plot_pr_model_comp(
    data_root: str,
    models: tuple[str, ...],
    *,
    condition: str = "arms-agent",
    run: int = 1,
    assays: tuple[str, ...] = (),
    field_types: tuple[str, ...] = ("ontology", "non_ontology", "all"),
    shared_window: bool = True,
    error_axes: bool = False,
    show_f1_contours: bool = True,
    no_color: bool = False,
    inset: bool = False,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """One condition's operating points in precision/recall space, one colour per model.

    The same figure as :func:`plot_pr_condition_comp`, with the model in place of the condition:
    every point is *condition* in run *run*, and what changes from one colour to the next is
    the model that produced it.  Everything else -- *assays*, *field_types*,
    *shared_window*, *error_axes*, *show_f1_contours*, *inset*, *title* and *save_path* -- means
    what it means there.  In an inset, a stack still at one spot is named inside the inset.

    A model is drawn where it has predictions, so *models* can name a model whose run has
    not been made, or not finished, yet.  With *assays*, each model is drawn in the panels of
    the assays it has predictions for, and a model with none of them is left out of the
    figure and its legend.  Pooled, a model is drawn only when it has predictions for every
    assay: pooled over fewer, its score would stand on different assays from the others'
    and could not be compared with them.

    Up to eight models can be drawn.  The marker shape is the field type, as everywhere, and
    the model is the colour -- one of :data:`~plots.marks.MODEL_COLOURS`, a colour-blind-safe
    set, given to the models drawn in the order *models* lists them.  With *no_color* the model
    is the pattern its mark is filled with instead, from :data:`~plots.marks.MODEL_HATCHES`:
    solid ink, then lines and grids in ink on white.

    Where marks of one field type land on the same spot, only the top one can be seen, so the
    stack is named: a box beside it lists every model in it, top first, joined to it by a faint
    line.  The box goes as close as it can without covering a mark or another box, and the line
    runs under the marks.  With a single field type the field-type key is left off, since one
    shape has nothing to tell apart; *title* can name the field type instead.

    Raises:
        ValueError: If *models* is empty or names a model twice, more models are drawn than
            there are colours, a field type or assay key is unknown, or no model has
            anything to draw.
    """
    _check_pr_arguments((condition,), (condition,), field_types)
    if not models:
        raise ValueError("models needs at least one model")
    if len(set(models)) != len(models):
        raise ValueError(f"Each model may be named once, got {list(models)}")
    _check_assay_keys(assays)
    # Pooled, a model has to cover the whole corpus; broken out, one requested assay will do.
    if assays:
        wanted = [get_assay(data_root, key) for key in assays]
        covers = any
    else:
        wanted = [assay for assay in iter_assays(data_root) if assay.has_gold]
        covers = all
    drawn = tuple(
        model for model in models if covers(assay.has_predictions(model, condition, run=run) for assay in wanted)
    )
    if not drawn:
        where = f"assays {list(assays)}" if assays else "every assay"
        raise ValueError(f"None of {list(models)} has {condition} predictions in run {run} for {where}")
    marked = _model_marks(drawn, no_color=no_color)
    _plot_pr_points(
        data_root,
        [(model, condition) for model in drawn],
        # One group per model, in the order listed: a later model is drawn over an earlier one.
        [([index], [mark]) for index, (_model, mark) in enumerate(marked)],
        marked,
        assays=assays,
        field_types=field_types,
        shared_window=shared_window,
        error_axes=error_axes,
        show_f1_contours=show_f1_contours,
        title=title,
        save_path=save_path,
        run=run,
        partial=True,
        name_stacks=True,
        inset=inset,
    )


def _check_assay_keys(assays: tuple[str, ...]) -> None:
    """Reject an assay key ``ASSAY_ORDER`` does not know before anything is read."""
    known = dict(ASSAY_ORDER)
    unknown = [key for key in assays if key not in known]
    if unknown:
        raise ValueError(f"Unknown assay key(s): {unknown}")


def _plot_pr_points(
    data_root: str,
    points: list[tuple[str, str]],
    groups: list[tuple[list[int], list[ConditionMark]]],
    keys: list[tuple[str, ConditionMark]],
    *,
    assays: tuple[str, ...],
    field_types: tuple[str, ...],
    shared_window: bool,
    error_axes: bool,
    show_f1_contours: bool,
    title: str | None,
    save_path: str | None,
    run: int,
    partial: bool = False,
    name_stacks: bool = False,
    inset: bool = False,
) -> None:
    """Draw and finish a precision/recall figure: the layout both public figures share.

    *points* is one (model, condition) per operating point.  *groups* splits them, by index
    into *points*, into the groups drawn one after another, each with its marks.  *keys* is
    the legend, one (label, mark) per thing compared.

    An assay is drawn only when every point has predictions for it, unless *partial*: then
    an assay is drawn when any point has, and each panel holds the points it has.  Only the
    assays asked for are scored, and with *partial* only those a point has predictions for,
    so an assay no run reached is never scored and never warns about its missing records.

    With *name_stacks*, marks of one field type that land on one spot are named beside it,
    by their labels in *keys* -- see :func:`_name_stacks`.  With *inset*, marks crowded together
    are drawn again, enlarged, in an inset of their panel -- see :func:`_draw_insets`.
    """
    _check_assay_keys(assays)
    labels = dict(ASSAY_ORDER)

    if assays:
        frames = {
            (index, field_type): create_per_assay_precision_recall_summary(
                data_root,
                model,
                condition,
                category=field_type,
                run=run,
                assays=[
                    key
                    for key in assays
                    if not partial or get_assay(data_root, key).has_predictions(model, condition, run=run)
                ],
            ).set_index("assay")
            for index, (model, condition) in enumerate(points)
            for field_type in field_types
        }
        wanted = {labels[key] for key in assays}
        covered = any if partial else all
        rows = [
            label
            for _key, label in ASSAY_ORDER
            if label in wanted and covered(label in frame.index for frame in frames.values())
        ]
        if not rows:
            raise ValueError(f"No requested assay has predictions for every one of {points}")

        def present(index: int, row: str) -> bool:
            return row in frames[(index, field_types[0])].index

        def score(index: int, field_type: str, row: str) -> tuple[float, float]:
            frame = frames[(index, field_type)]
            return frame.loc[row, "recall"], frame.loc[row, "precision"]

        # The records behind the row: the most any point was scored on, which is the whole
        # assay once every run behind the panel has finished it.
        n_records = {
            row: max(
                int(frames[(index, field_types[0])].loc[row, "n_records"])
                for index in range(len(points))
                if present(index, row)
            )
            for row in rows
        }
    else:
        summaries = [
            create_overall_precision_recall_summary(data_root, model, condition, run=run).set_index("category")
            for model, condition in points
        ]
        rows = [POOLED_LABEL]

        def score(index: int, field_type: str, _row: str) -> tuple[float, float]:
            summary = summaries[index]
            return summary.loc[field_type, "recall"], summary.loc[field_type, "precision"]

        n_records = {POOLED_LABEL: int(summaries[0].loc[field_types[0], "n_records"])}

        def present(_index: int, _row: str) -> bool:
            return True

    def panel_title(row: str) -> str:
        """The row's name with the records standing behind it.

        A panel drawn from 15 records and one drawn from 100 look identical otherwise,
        and the difference decides how much either is worth reading into.
        """
        return f"{row} (n={n_records[row]})"

    def placed(index: int, field_type: str, row: str) -> tuple[float, float]:
        """One point, with recall turned into its miss rate when asked for."""
        recall, precision = score(index, field_type, row)
        return (1.0 - recall if error_axes else recall, precision)

    def series(row: str, drawn: tuple[str, ...]) -> list[tuple[list[tuple[float, float]], list[str], str]]:
        """The (points, colours, marker) series of one panel.

        Group before field type, so a whole condition is laid down before the next one
        starts.  The shapes stack by :data:`~plots.marks.MARKER_ZORDER` whatever the order,
        so two field types on one spot both stay visible; where two conditions put the same
        shape on the same spot, the later group -- the system -- is the one left legible.
        """
        drawn_series = []
        for group, group_marks in groups:
            # A point with nothing for this row is left out of it, with its mark.
            kept = [(index, mark) for index, mark in zip(group, group_marks, strict=True) if present(index, row)]
            if not kept:
                continue
            for field_type in drawn:
                drawn_series.append(
                    (
                        [placed(index, field_type, row) for index, _mark in kept],
                        [mark for _index, mark in kept],
                        FIELD_TYPE_MARKERS[field_type],
                    )
                )
        return drawn_series

    def drawn_points(row: str, drawn: tuple[str, ...]) -> list[tuple[str, int, tuple[float, float]]]:
        """Every (field type, point index, position) of one panel, in the order :func:`series` draws them."""
        return [
            (field_type, index, placed(index, field_type, row))
            for group, _marks in groups
            for index in group
            if present(index, row)
            for field_type in drawn
        ]

    # The panels whose stacks are named once the layout is final, with what each one holds.
    panels: list[tuple[plt.Axes, list[tuple[str, int, tuple[float, float]]]]] = []

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
            panels.append((ax, drawn_points(row, field_types)))
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
                panels.append((ax, drawn_points(row, (field_type,))))
                if row_index == 0:
                    ax.set_title(FIELD_TYPE_LABELS[field_type], fontsize=PANEL_TITLE_SIZE)
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
    # repeat it; with one field type in all, there is only one shape to key.
    strip = _pr_legends(fig, keys, field_types, show_field_keys=shared_window and len(field_types) > 1)
    fig.tight_layout(rect=(0.0, strip, 1.0, 1.0))
    if title:
        # After tight_layout, which does not know about a suptitle added later: adding it
        # first would have the layout reserve the space and then leave a gap when there
        # is no title.
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    # Last, so every inset and box is placed against the panels where they will finally sit.
    names = [label for label, _mark in keys]
    marks_of = {index: mark for group, group_marks in groups for index, mark in zip(group, group_marks, strict=True)}
    insets = [_draw_insets(ax, panel_points, marks_of) if inset else [] for ax, panel_points in panels]
    if name_stacks:
        # Drawn once first: an inset decides which of its connectors show only as it is drawn,
        # and a name box has to keep off the ones that do.
        fig.canvas.draw()
        for (ax, panel_points), panel_insets in zip(panels, insets, strict=True):
            zoomed = frozenset(member for _inset, members in panel_insets for member in members)
            extents = tuple(ax_inset.get_window_extent() for ax_inset, _members in panel_insets)
            _name_stacks(ax, panel_points, names, avoid=extents, avoid_lines=_connector_lines(ax), skip=zoomed)
            for ax_inset, _members in panel_insets:
                _name_stacks(ax_inset, _in_view(ax_inset, panel_points), names)
    _finish(fig, save_path)


#: How close two marks of one field type sit, in points, before the lower one counts as hidden
#: and the stack is named.  Two thirds of a mark's width: closer than that, what shows of the
#: lower mark is a sliver of rim, too little to name the model by its colour or pattern.
STACK_DISTANCE_POINTS = 8.0

#: How far a stack's name box may sit from it, nearest first, in points.  The first is far
#: enough that the line to the box still shows past the mark it starts under.
STACK_LABEL_DISTANCES = (22, 26, 30, 35, 40, 46, 53, 60, 68, 77, 87, 100, 115, 130, 150)

#: The directions a name box is tried in, in degrees.  Never straight sideways or up and down:
#: a line along a gridline disappears into it, and every point at a score of 1 sits on one.
STACK_LABEL_ANGLES = tuple(angle for angle in range(0, 360, 15) if angle % 90)

#: How much clear page a name box keeps around every mark, in points: a mark's radius and a
#: little air.
STACK_LABEL_CLEARANCE = 9.0

#: The line from a stack to its name: a guide, quieter than the gridlines it crosses.
STACK_LINE_COLOUR = "#cfcfca"
STACK_BOX_EDGE_COLOUR = "#c9c9c4"
STACK_LABEL_SIZE = 6.5

#: The id every stack name box carries, which tells it apart from the contour labels.
STACK_NAME_GID = "stack-name"


def _name_stacks(
    ax: plt.Axes,
    panel_points: list[tuple[str, int, tuple[float, float]]],
    names: list[str],
    *,
    avoid: tuple[Bbox, ...] = (),
    avoid_lines: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
    skip: frozenset[int] = frozenset(),
) -> None:
    """Name every stack of marks in *ax* that hides one mark under another.

    *panel_points* is every mark in the panel as (field type, index into *names*, position), in
    the order they were drawn.  A stack is marks of one field type within
    :data:`STACK_DISTANCE_POINTS` of each other; shapes of different field types stack by
    :data:`~plots.marks.MARKER_ZORDER` and stay visible, so they are never a stack.  The box
    lists the stack's names top first -- the mark the reader can see heads the list.

    The box goes to the nearest of the spots tried that covers no mark and no box already
    placed, and stays inside the panel.  A line that has to pass a mark costs a little, and is
    drawn under the marks so that it runs behind them rather than over them.  Where every spot
    covers something, the one covering fewest marks is taken: a crowded panel still says what
    is in the stack.

    *avoid* is pixel extents a box must keep off as it keeps off other boxes -- the panel's
    insets -- and *avoid_lines* pixel segments it must not sit across, the insets' connectors.
    *skip* is positions in *panel_points* whose stacks are named elsewhere, in an inset; they
    still count as marks a box must not cover.
    """
    renderer = ax.figure.canvas.get_renderer()
    to_pixels = ax.figure.dpi / 72
    marks = [ax.transData.transform(xy) for _field_type, _index, xy in panel_points]
    frame = ax.get_window_extent(renderer)
    placed: list[Bbox] = list(avoid)

    for stack in _stacks(panel_points, marks, STACK_DISTANCE_POINTS * to_pixels):
        if skip.issuperset(stack):
            continue
        top_first = [names[panel_points[member][1]] for member in reversed(stack)]
        # The bottom mark's own position: the stack's first member, in this field type.
        anchor = panel_points[stack[0]][2]
        start = ax.transData.transform(anchor)
        best: tuple[float, plt.Annotation, tuple[float, float]] | None = None
        for distance in STACK_LABEL_DISTANCES:
            for angle in np.deg2rad(STACK_LABEL_ANGLES):
                offset = (distance * np.cos(angle), distance * np.sin(angle))
                box = ax.annotate(
                    "\n".join(top_first),
                    anchor,
                    xytext=offset,
                    textcoords="offset points",
                    ha="right" if offset[0] < 0 else "left",
                    va="top" if offset[1] < 0 else "bottom",
                    fontsize=STACK_LABEL_SIZE,
                    color=LABEL_COLOUR,
                    linespacing=1.15,
                    zorder=5,
                    gid=STACK_NAME_GID,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": STACK_BOX_EDGE_COLOUR,
                        "linewidth": 0.6,
                    },
                )
                box.draw(renderer)  # the box's extent is only known once it has been laid out
                extent = box.get_bbox_patch().get_window_extent(renderer)
                end = (min(max(start[0], extent.x0), extent.x1), min(max(start[1], extent.y0), extent.y1))
                clearance = STACK_LABEL_CLEARANCE * to_pixels
                near = extent.expanded(1.0, 1.0).padded(clearance)
                covered = sum(near.contains(x, y) for x, y in marks) + sum(extent.overlaps(o) for o in placed)
                # A line behind another box would seem to lead to that box instead.
                covered += sum(_runs_through(start, end, other) for other in placed)
                covered += sum(_runs_through(head, tail, extent) for head, tail in avoid_lines)
                crossed = sum(_passes(start, end, mark, clearance * 0.75) for mark in marks)
                inside = frame.contains(extent.x0, extent.y0) and frame.contains(extent.x1, extent.y1)
                cost = (0 if inside else 1e6) + covered * 1e3 + crossed * 2 + distance / 100
                if best is None or cost < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (cost, box, end)
                else:
                    box.remove()
        _cost, box, end = best
        placed.append(box.get_bbox_patch().get_window_extent(renderer))
        x_end, y_end = ax.transData.inverted().transform(end)
        ax.plot(
            [anchor[0], x_end],
            [anchor[1], y_end],
            color=STACK_LINE_COLOUR,
            linewidth=0.6,
            solid_capstyle="butt",
            zorder=1.5,  # over the contours, under every mark
        )


def _stacks(
    panel_points: list[tuple[str, int, tuple[float, float]]],
    marks: list[np.ndarray],
    reach: float,
) -> list[list[int]]:
    """The groups, two or more each, of marks of one field type within *reach* pixels of each other.

    Each group lists positions in *panel_points*, in drawing order, bottom first.
    """
    stacks = []
    taken: set[int] = set()
    for first, (field_type, _index, _xy) in enumerate(panel_points):
        if first in taken:
            continue
        members = [first] + [
            other
            for other in range(first + 1, len(panel_points))
            if other not in taken
            and panel_points[other][0] == field_type
            and np.hypot(*(marks[other] - marks[first])) < reach
        ]
        taken.update(members)
        if len(members) > 1:
            stacks.append(members)
    return stacks


def _passes(start: np.ndarray, end: tuple[float, float], mark: np.ndarray, reach: float) -> bool:
    """Whether the line from *start* to *end* passes within *reach* of *mark*, other than the mark it starts at."""
    if np.hypot(*(mark - start)) < 2:
        return False
    run = np.subtract(end, start)
    along = np.clip(np.dot(mark - start, run) / max(np.dot(run, run), 1e-9), 0.0, 1.0)
    return bool(np.hypot(*(mark - (start + along * run))) < reach)


def _runs_through(start: np.ndarray, end: tuple[float, float], box: Bbox) -> bool:
    """Whether the line from *start* to *end* passes through *box*, a pixel extent."""
    return any(box.contains(*(start + step * np.subtract(end, start))) for step in np.linspace(0.0, 1.0, 25))


#: How close marks sit, in points, before they count as crowded and are enlarged in an inset:
#: a mark and a half apart, near enough that the gap between them is hard to read.
ZOOM_REACH_POINTS = 18.0

#: How much an inset must enlarge its patch to be worth the room it takes.
MIN_ZOOM = 2.0

#: How much room an inset's window leaves round its marks, as a share of their spread, and the
#: least it leaves, in score units, so a pair one hundredth apart is not drawn edge to edge.
ZOOM_MARGIN = 0.5
ZOOM_MIN_HALF_SPAN = 0.012

#: The sizes an inset is tried at, as a share of its panel's width, largest first: one inset
#: to a panel, or each of two.  A smaller inset that covers nothing beats a larger one that
#: covers a mark.
INSET_SIZES = (0.52, 0.46, 0.40)
INSET_PAIR_SIZES = (0.44, 0.38, 0.32)

#: How many places along each side of the panel an inset is tried at, and how far it keeps
#: from the panel's frame, as a share of the panel: further on the left and at the bottom,
#: where its own tick numbers sit and would otherwise land on the frame.
INSET_POSITIONS = 6
INSET_MARGIN = 0.03
INSET_TICK_MARGIN = (0.10, 0.08)

#: The inset's frame, the frame round the patch it enlarges, and the dashed lines joining them:
#: one grey, quieter than the panel frame, so the inset reads as a detail of its panel.
ZOOM_FRAME_COLOUR = "#8a8a84"
ZOOM_CONNECTOR_COLOUR = "#b5b5ae"
ZOOM_TICK_SIZE = 6


def _draw_insets(
    ax: plt.Axes,
    panel_points: list[tuple[str, int, tuple[float, float]]],
    marks_of: dict[int, ConditionMark],
) -> list[tuple[plt.Axes, list[int]]]:
    """Enlarge each crowded group of *ax*'s marks in an inset, and return the insets with their groups.

    A group is marks of any field type chained within :data:`ZOOM_REACH_POINTS` of one another
    that do not all sit at one score -- marks at one score stay one mark however far they are
    enlarged, and are named instead.  Each group takes an inset of its own, up to two to a
    panel; past two, the groups nearest each other share an inset until two are left.  An inset that would not
    enlarge its patch :data:`MIN_ZOOM` times is not drawn.

    The inset holds every mark inside its window, drawn as in the panel and in the same
    order, on a square window so a step sideways still means what a step up means.  Each inset
    goes to the spot, of a grid of :data:`INSET_POSITIONS` a side at the sizes of
    :data:`INSET_SIZES` or :data:`INSET_PAIR_SIZES`, that covers fewest marks, zoomed patches
    and insets already placed -- the largest and lowest of those that cover none.
    """
    groups = _crowds(ax, panel_points)
    while len(groups) > 2:
        # Two insets is what a panel has room for: the two groups nearest each other share one.
        centres = [np.mean([panel_points[member][2] for member in group], axis=0) for group in groups]
        first, second = min(
            ((i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))),
            key=lambda pair: np.hypot(*(centres[pair[0]] - centres[pair[1]])),
        )
        groups[first] = sorted(groups[first] + groups.pop(second))
    windows = [_zoom_window(panel_points, group) for group in groups]
    sizes = INSET_SIZES if len(groups) == 1 else INSET_PAIR_SIZES
    panel_span = PR_WINDOW[1] - PR_WINDOW[0]
    # Judged at the largest size: a group too spread out to gain from that gains less from less.
    kept = [
        (group, window)
        for group, window in zip(groups, windows, strict=True)
        if sizes[0] * panel_span / (window[1] - window[0]) >= MIN_ZOOM
    ]
    if not kept:
        return []

    reach = STACK_LABEL_CLEARANCE * ax.figure.dpi / 72
    marks = [ax.transData.transform(xy) for _field_type, _index, xy in panel_points]
    obstacles = [Bbox(ax.transData.transform([(x0, y0), (x1, y1)])) for _group, (x0, x1, y0, y1) in kept]

    def cost(slot: tuple[float, float, float, float]) -> float:
        """What *slot* would cover: marks (any of one), the frames of zoomed patches, other insets."""
        x, y, w, h = slot
        extent = Bbox(ax.transAxes.transform([(x, y), (x + w, y + h)]))
        covered = sum(extent.padded(reach).contains(*mark) for mark in marks)
        covered += sum(extent.overlaps(obstacle) for obstacle in obstacles)
        # Among clear spots, the larger and the lower: the marks of these panels sit high.
        return covered * 1e3 + (sizes[0] - w) * 10 + y

    layout = []
    for _group in kept:
        slot = min(
            (
                (x, y, size, size)
                for size in sizes
                for x in np.linspace(INSET_TICK_MARGIN[0], 1 - INSET_MARGIN - size, INSET_POSITIONS)
                for y in np.linspace(INSET_TICK_MARGIN[1], 1 - INSET_MARGIN - size, INSET_POSITIONS)
            ),
            key=cost,
        )
        layout.append(slot)
        obstacles.append(Bbox(ax.transAxes.transform([slot[:2], (slot[0] + slot[2], slot[1] + slot[3])])))

    insets = []
    for slot, (group, (x0, x1, y0, y1)) in zip(layout, kept, strict=True):
        ax_inset = ax.inset_axes(slot)
        for field_type, index, xy in panel_points:
            _draw_pr_path(ax_inset, [xy], [marks_of[index]], FIELD_TYPE_MARKERS[field_type])
        ax_inset.set_xlim(x0, x1)
        ax_inset.set_ylim(y0, y1)
        ax_inset.set_aspect("equal")
        ax_inset.set_facecolor("white")
        ax_inset.grid(color=GRID_COLOUR, linewidth=0.6)
        ax_inset.set_axisbelow(True)
        ax_inset.tick_params(labelsize=ZOOM_TICK_SIZE, length=2, pad=1.5, color=PANEL_FRAME_COLOUR)
        for axis in (ax_inset.xaxis, ax_inset.yaxis):
            axis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
        for spine in ax_inset.spines.values():
            spine.set_color(ZOOM_FRAME_COLOUR)
            spine.set_linewidth(0.8)
        indicator = ax.indicate_inset_zoom(ax_inset, edgecolor=ZOOM_FRAME_COLOUR, linewidth=0.7, alpha=1.0)
        for connector in indicator.connectors:
            connector.set_linestyle((0, (2, 2)))
            connector.set_linewidth(0.6)
            connector.set_color(ZOOM_CONNECTOR_COLOUR)
        insets.append((ax_inset, group))
    return insets


def _crowds(ax: plt.Axes, panel_points: list[tuple[str, int, tuple[float, float]]]) -> list[list[int]]:
    """The groups of positions in *panel_points* worth enlarging -- see :func:`_draw_insets`."""
    reach = ZOOM_REACH_POINTS * ax.figure.dpi / 72
    marks = [ax.transData.transform(xy) for _field_type, _index, xy in panel_points]
    groups: list[list[int]] = []
    seen: set[int] = set()
    for first in range(len(panel_points)):
        if first in seen:
            continue
        group, frontier = {first}, [first]
        while frontier:
            current = frontier.pop()
            for other in range(len(panel_points)):
                if other not in group and np.hypot(*(marks[other] - marks[current])) < reach:
                    group.add(other)
                    frontier.append(other)
        seen |= group
        if len({panel_points[member][2] for member in group}) > 1:
            groups.append(sorted(group))
    return groups


def _zoom_window(
    panel_points: list[tuple[str, int, tuple[float, float]]], group: list[int]
) -> tuple[float, float, float, float]:
    """The square window, (x0, x1, y0, y1), an inset shows *group* through.

    It reaches past the panel's edge no further than the panel's own window does, so an inset
    never offers a score above 1 that no condition could reach.
    """
    xs = [panel_points[member][2][0] for member in group]
    ys = [panel_points[member][2][1] for member in group]
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * (1 + ZOOM_MARGIN) + ZOOM_MIN_HALF_SPAN
    low, high = PR_WINDOW[0], 1.0 + 0.25 * half

    def span(values: list[float]) -> tuple[float, float]:
        centre = min(max((min(values) + max(values)) / 2, low + half), high - half)
        return centre - half, centre + half

    return (*span(xs), *span(ys))


def _in_view(
    ax: plt.Axes, panel_points: list[tuple[str, int, tuple[float, float]]]
) -> list[tuple[str, int, tuple[float, float]]]:
    """The points of *panel_points* that fall inside *ax*'s window."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return [point for point in panel_points if x0 <= point[2][0] <= x1 and y0 <= point[2][1] <= y1]


def _connector_lines(ax: plt.Axes) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """The visible zoom connectors of *ax*'s insets, as pixel segments, once the figure is drawn."""
    lines = []
    for indicator in ax.findobj(InsetIndicator):
        for connector in indicator.connectors:
            if connector.get_visible():
                # An inset's connectors give both ends as transforms: the inset's corner, the frame's.
                lines.append((connector.coords1.transform(connector.xy1), connector.coords2.transform(connector.xy2)))
    return tuple(lines)
