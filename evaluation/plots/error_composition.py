"""What the errors are made of: one bar for the whole corpus, and one bar per assay.

Two readings of the same taxonomy.  The corpus bar is the headline -- every counted error
once, at the sub-category level, with the categories braced above it so the figure says
both what the run did and what it cost in the metric's own notation.  The per-assay figure
is the same measurement repeated down the page, to show whether the corpus bar is one
story or twelve different ones averaged.

Both are shares rather than counts, because the assays differ in size by more than
tenfold.  How a share becomes a segment is :mod:`plots.segments`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.data_analysis import (
    CATEGORIES,
    CATEGORY_BY_SUBCATEGORY,
    CONFUSION_CELLS_BY_CATEGORY,
    POOLED_ASSAY,
    SUBCATEGORIES,
    category_shares,
    collect_field_errors,
    deduplicate_errors,
    subcategory_shares,
)
from assays import ASSAY_ORDER
from plots.segments import _label_narrow_segments, _segment_colours, _segment_fills, _stack_row
from plots.theme import (
    AXIS_COLOUR,
    FIGURE_TITLE_SIZE,
    GRID_COLOUR,
    LABEL_COLOUR,
    NO_COLOR_INK,
    PANEL_CHROME_INCHES,
    _finish,
    _legend_columns,
)

if TYPE_CHECKING:
    import pandas as pd

#: Height allowed per assay bar, in inches, and the gap the pooled row sits behind.
COMPOSITION_BAR_INCHES = 0.26
POOLED_ROW_GAP = 0.6

#: Width of one composition panel, and the strip of assay names to its left.  Kept apart so
#: a figure of one panel comes out the width of a panel plus its names rather than half of
#: a two-panel figure, names and all.
COMPOSITION_PANEL_INCHES = 4.6
COMPOSITION_LABEL_INCHES = 1.8

#: What the one composition panel is called.  "Once" is the load-bearing word: the category
#: is the confusion case, so a substitution carries one label rather than one per side.
COMPOSITION_TITLE = "Every counted error, once"

#: Where the category braces sit above the bar, in axes fractions: the rule, the little
#: hooks that turn it into a bracket, and the label that names what the span costs.
BRACE_LINE = 1.10
BRACE_HOOK = 1.02
BRACE_LABEL = 1.16

#: Air between a label and the rule it sits in, in points.  The label masks the rule with
#: its own background rather than the rule being drawn around it, so this is the whole
#: margin however long the label is -- no guessing a gap from the character count.
BRACE_LABEL_PAD = 1.5

#: The shortest run of rule that still reads as a rule rather than as a tick, in points.
#: A label sits in the rule only when it leaves at least this much on both sides; below
#: that it goes above the rule, where it costs the span nothing.
#:
#: Measured against the label rather than against the span, because the labels differ in
#: width by threefold -- "FP" is a third of "FP + FN" -- so no one span can decide for
#: both.  Judging by the span alone pushed "FP" above a bracket it fitted inside five
#: times over.
BRACE_MIN_RULE_STUB = 6.0

#: How far each brace is drawn inside its span.  Categories are adjacent by construction,
#: so without it two neighbouring braces meet at a shared edge and read as one long rule
#: with a stray tick in the middle.
BRACE_INSET = 0.004


def _composition_legend(ax: plt.Axes, categories: tuple[str, ...], *, no_color: bool = False, y: float = -0.10) -> None:
    """The key, under the panel it describes: the panels do not share categories."""
    colours, hatches = _segment_fills(categories, no_color=no_color)
    handles = [
        Patch(
            facecolor=colour,
            edgecolor=NO_COLOR_INK if no_color else "white",
            hatch=hatch,
            label=category,
        )
        for category, colour, hatch in zip(categories, colours, hatches, strict=True)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=_legend_columns(len(categories)),
        frameon=False,
        fontsize=8,
        labelcolor=LABEL_COLOUR,
        handlelength=1.4,
        columnspacing=1.2,
    )


def _composition_panel(
    ax: plt.Axes,
    shares: pd.DataFrame,
    labels: list[str],
    categories: tuple[str, ...],
    *,
    label_segments: bool,
) -> list[str]:
    """A stacked bar per assay, with the pooled row set apart below them.

    Every assay gets a row whether or not it has errors of its own, so an assay that is
    clean reads as ``n=0`` rather than dropping out of the axis.  Returns the tick labels
    in drawing order.
    """
    colours = _segment_colours(categories)
    by_assay = {assay: frame.set_index("category") for assay, frame in shares.groupby("assay", observed=True)}

    # The pooled bar is set apart by a gap rather than a rule: it is the same measurement
    # over a different set of records, not a total of the bars above it.
    rows = [(float(index), label) for index, label in enumerate(labels)]
    rows.append((len(labels) - 1 + POOLED_ROW_GAP + 0.7, POOLED_ASSAY))

    for position, label in rows:
        frame = by_assay.get(label)
        if frame is not None:
            _stack_row(
                ax,
                position,
                frame["share"].to_dict(),
                categories,
                colours,
                label_segments=label_segments,
                height=0.8 if label == POOLED_ASSAY else 0.7,
            )
        count = 0 if frame is None else int(frame["n_errors"].iloc[0])
        ax.text(1.01, position, f"n={count:,}", va="center", fontsize=7.5, color=LABEL_COLOUR)

    positions = [position for position, _label in rows]
    drawn = [label for _position, label in rows]

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(max(positions) + 0.7, -0.7)
    ax.set_yticks(positions)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOUR)
    ax.tick_params(colors=LABEL_COLOUR, labelsize=8, left=False)
    return drawn


def _label_fits_inline(ax: plt.Axes, text: plt.Text, left: float, right: float) -> bool:
    """Whether *text* can break the rule between *left* and *right* and leave a rule.

    Everything is compared in display pixels, which is the one space the label's width and
    the span are both already in: the label is set in points and the span in data units,
    and converting either into the other's terms would depend on the figure's width.
    """
    span = abs(ax.transData.transform((right, 0))[0] - ax.transData.transform((left, 0))[0])
    per_point = ax.get_figure().dpi / 72.0
    margin = 2 * (BRACE_LABEL_PAD + BRACE_MIN_RULE_STUB) * per_point
    return text.get_window_extent().width + margin <= span


def _category_brace(ax: plt.Axes, left: float, right: float, label: str) -> None:
    """A bracket over ``[left, right]``, naming the confusion cells that span costs.

    Drawn in the x-axis transform, so the ends land on data coordinates and the height
    stays put whatever the bar's own scale is.  The rule breaks around the label rather
    than running under it, and a label with no room to break it sets above the rule
    instead -- otherwise the rule would come out as two stubs with a name between them.

    Which of the two happens is settled by measuring the label after it is placed, since
    that is the only way to know how wide it came out.  It is set inline first because
    that is the common case, and moved up only when the measurement says it does not fit.
    """
    transform = ax.get_xaxis_transform()
    middle = (left + right) / 2
    left, right = left + BRACE_INSET, right - BRACE_INSET
    ink = {"color": LABEL_COLOUR, "linewidth": 0.9, "clip_on": False, "transform": transform}

    for edge in (left, right):
        ax.plot([edge, edge], [BRACE_HOOK, BRACE_LINE], **ink)
    ax.plot([left, right], [BRACE_LINE, BRACE_LINE], **ink)

    text = ax.text(
        middle,
        BRACE_LINE,
        label,
        transform=transform,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=LABEL_COLOUR,
        clip_on=False,
        zorder=3,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": BRACE_LABEL_PAD},
    )
    if not _label_fits_inline(ax, text, left, right):
        # Above the rule, and with no patch of white: up there it masks nothing, and a
        # white patch would only punch a hole in whatever the label happens to sit over.
        text.set_y(BRACE_LABEL)
        text.set_va("bottom")
        text.set_bbox(None)


def plot_error_composition(
    data_root: str,
    model: str,
    run_type: str = "arms-agent",
    *,
    field_type: str | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """What each assay's errors are made of.

    One 100%-wide bar per assay over the categories, with the pooled corpus set apart at the
    bottom.  Shares rather than counts, because the assays differ in size by more than
    tenfold and a stack of absolute counts would show only that ATACseq is large; the count
    each share is taken over is printed at the end of its bar, so a share standing on very
    little says so.

    The bars are the categories, not the sub-categories: a stacked bar reads at three
    segments and not at seven, and the category is the confusion case, which is the split a
    reader already knows from the precision/recall tables.
    """
    errors = collect_field_errors(data_root, model, run_type)
    if errors.empty:
        raise ValueError(f"no predictions on disk for {run_type!r} under {model!r}")

    labels = [label for _key, label in ASSAY_ORDER if label in set(errors["assay"])]
    shares = category_shares(errors, field_type=field_type)

    fig, ax = plt.subplots(
        figsize=(
            COMPOSITION_PANEL_INCHES + COMPOSITION_LABEL_INCHES,
            COMPOSITION_BAR_INCHES * (len(labels) + 2) + PANEL_CHROME_INCHES + 0.6,
        )
    )
    drawn = _composition_panel(ax, shares, labels, CATEGORIES, label_segments=False)
    ax.set_title(COMPOSITION_TITLE, fontsize=10, color=LABEL_COLOUR)
    ax.set_yticklabels(drawn, fontsize=8)
    _composition_legend(ax, CATEGORIES)

    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=1.0, va="bottom")
    _finish(fig, save_path)


def plot_corpus_error_composition(
    data_root: str,
    model: str,
    run_type: str = "arms-agent",
    *,
    field_type: str | None = None,
    apply_dedup: bool = False,
    show_subcategories: bool = True,
    show_confusion: bool = True,
    no_color: bool = False,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """The whole corpus in one bar: every counted error, at one level of the taxonomy.

    The headline reading.  The bar carries the sub-categories, which is as fine as the
    analysis goes; the categories are the braces above it, each naming the confusion cells
    its span counts toward -- so the figure says what the run did and what it cost in the
    metric's own notation, without a reader having to hold the mapping.

    *show_subcategories* set to ``False`` stacks the three categories instead, for a reading
    that has to carry from across a room.  The braces are unchanged either way: they still
    span a category, which at that level is a single segment.

    *apply_dedup* counts each distinct error once instead of once per record it occurs in --
    how many different things the run gets wrong, rather than how much work they cost.  The
    two readings differ sharply, this corpus repeating itself as it does, and the bar itself
    does not say which of them it is drawing: whatever carries the figure has to.

    *show_confusion* set to ``False`` drops the braces.  Worth doing where the figure sits
    beside the precision/recall tables and the mapping is already in the reader's hands, or
    where the space above the bar is wanted for something else.

    *no_color* draws the segments as light greys told apart by hatching rather than by hue,
    which is what survives a greyscale print, a photocopy, and a reader who cannot separate
    the hues.  The hatches differ in direction and not only in density, since density is the
    thing a photocopy flattens.

    The sub-categories run in their categories' order, which is what makes each category a
    single unbroken span to brace.  Every error appears once: the category is the confusion
    case, so a substitution carries one label rather than one per side.
    """
    errors = collect_field_errors(data_root, model, run_type)
    if errors.empty:
        raise ValueError(f"no predictions on disk for {run_type!r} under {model!r}")
    if apply_dedup:
        errors = deduplicate_errors(errors)

    if show_subcategories:
        level, order = "subcategory", SUBCATEGORIES
        shares = subcategory_shares(errors, field_type=field_type)
    else:
        level, order = "category", CATEGORIES
        shares = category_shares(errors, field_type=field_type)
    pooled = shares[shares["assay"] == POOLED_ASSAY].set_index(level)
    by_segment = pooled["share"].to_dict()
    category_of = CATEGORY_BY_SUBCATEGORY if show_subcategories else {name: name for name in CATEGORIES}

    colours, hatches = _segment_fills(order, no_color=no_color)
    fig, ax = plt.subplots(figsize=(9.6, 2.5 if show_confusion else 2.1))
    narrow = _stack_row(ax, 0.0, by_segment, order, colours, label_segments=True, hatches=hatches, no_color=no_color)
    _label_narrow_segments(ax, narrow)

    # The braces are spans of the bar, so they are measured off the same running total the
    # segments were drawn from rather than recomputed from the category shares.
    if show_confusion:
        left = 0.0
        for category in CATEGORIES:
            width = sum(by_segment.get(name, 0.0) for name in order if category_of[name] == category)
            if width:
                _category_brace(ax, left, left + width, CONFUSION_CELLS_BY_CATEGORY[category])
            left += width

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.6, -0.6)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _composition_legend(ax, order, no_color=no_color, y=-0.24 if narrow else -0.10)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    if title:
        fig.suptitle(title, fontsize=FIGURE_TITLE_SIZE, y=0.99, va="top")
    _finish(fig, save_path)
