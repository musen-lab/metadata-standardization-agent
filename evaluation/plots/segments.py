"""How a share becomes a segment of a stacked bar, and how a segment gets its number.

The segments are error categories, which are ordered by how much the error costs to fix,
so they take a *scale* rather than the pair of hues :mod:`plots.marks` gives the runs: one
hue, light to dark, however many categories there are.  With colour off the scale is spent
and the pattern carries the distinction instead.

The numbering is the awkward part and most of what is here.  A share of two per cent has
no room for "2%" inside it, and the smallest categories are neighbours, so their numbers
cluster where there is least space for them.  They are set beneath the bar on leaders and
nudged apart, which is what :func:`_spread_labels` and :func:`_label_narrow_segments` do
between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex

from plots.theme import (
    AXIS_COLOUR,
    LABEL_COLOUR,
    LABEL_ON_DARK_BELOW,
    NO_COLOR_INK,
    _relative_luminance,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

#: One monotone ramp for the error categories, light to dark in step with how much
#: the error costs to fix.  A third hue on purpose: blue and orange name *runs* in every
#: other figure, and a reader who learned that would read a coloured segment here as
#: a condition.  The dark end clears 4.5:1 on white, so a label can sit on it in white.
CATEGORY_RAMP = ("#BBD5D1", "#7FB0AA", "#4A8A83", "#255F59")

#: The fill a hatched segment sits on when colour is off.  White, so the hatch and the seam
#: sit at full contrast against it -- the hatch is what tells the segments apart, and a grey
#: behind it spends contrast on a distinction the hatch is already making.  It also gives
#: the percentages the same white to sit on wherever they land.
NO_COLOR_SEGMENT_FILL = "white"

#: How the segments are told apart when colour is off, as ``(fill, hatch)`` in the order
#: they are drawn: open, one hatch, filled, then the rest of the hatches.
#:
#: Open and filled are the two a reader never has to decode, and they are the furthest apart
#: anything on the page can be.  Open leads.  Filled waits until the third place, because it
#: is the heaviest mark the page can carry and the second segment is the widest category on
#: this corpus -- solid ink that wide takes the figure over.  The hatches differ in
#: *direction* as well as density, since density is the thing a photocopy flattens and
#: direction is what it leaves.
NO_COLOR_SEGMENT_PATTERNS = (
    (NO_COLOR_SEGMENT_FILL, ""),
    (NO_COLOR_SEGMENT_FILL, "|||"),
    (NO_COLOR_INK, ""),
    (NO_COLOR_SEGMENT_FILL, "+++"),
    (NO_COLOR_SEGMENT_FILL, "..."),
    (NO_COLOR_SEGMENT_FILL, "///"),
    (NO_COLOR_SEGMENT_FILL, "xxx"),
    (NO_COLOR_SEGMENT_FILL, "\\\\\\"),
    (NO_COLOR_SEGMENT_FILL, "ooo"),
    (NO_COLOR_SEGMENT_FILL, "---"),
)

#: Below this share a segment is too narrow to hold its own number at 8pt, so the number is
#: set beneath the bar on a leader line instead of squeezed inside or dropped.
MIN_LABELLED_SHARE = 0.05

#: Where a narrow segment's number goes, in axes fractions: the leader leaves the bar's
#: underside, and the number sits just below the axis.
NARROW_LABEL_LEADER_TOP = 0.20
NARROW_LABEL_LEADER_FOOT = 0.04
NARROW_LABEL_Y = 0.0

#: How far apart two of those numbers must sit.  Narrow segments cluster -- the smallest
#: categories are neighbours -- so without this their labels would land on top of each other
#: and the leaders would be the only thing telling them apart.
NARROW_LABEL_MIN_GAP = 0.055

#: The rightmost a number may be centred.  Its own width sits either side of that centre,
#: so allowing 1.0 would hang half of it off the end of the bar.
NARROW_LABEL_RIGHT_LIMIT = 0.98


def _segment_colours(categories: tuple[str, ...]) -> list[str]:
    """One ramp step per category, spread across the ramp when there are fewer than steps."""
    if len(categories) == len(CATEGORY_RAMP):
        return list(CATEGORY_RAMP)
    if len(categories) > len(CATEGORY_RAMP):
        # More categories than steps: interpolate within the same hue rather than reach for
        # another, so the scale still reads as one ramp however many segments it carries.
        ramp = LinearSegmentedColormap.from_list("category", CATEGORY_RAMP)
        return [to_hex(ramp(step)) for step in np.linspace(0.0, 1.0, len(categories))]
    # Keep the darkest step for the last category whatever the count, so the far end of the
    # scale reads the same wherever it is drawn.
    picked = [0, *range(len(CATEGORY_RAMP) - len(categories) + 1, len(CATEGORY_RAMP))]
    return [CATEGORY_RAMP[index] for index in picked]


def _segment_fills(categories: tuple[str, ...], *, no_color: bool) -> tuple[list[str], list[str]]:
    """The fill and the hatch each segment takes.

    In colour the hatch is empty and the ramp carries the whole distinction.  With colour off
    the pair carries it: :data:`NO_COLOR_SEGMENT_PATTERNS` in order, the two plain patterns
    near the front and hatches around them, all in ink at full contrast -- which is what
    survives a greyscale print, a photocopy, and a reader who cannot separate the hues.
    """
    if not no_color:
        return _segment_colours(categories), [""] * len(categories)
    patterns = [NO_COLOR_SEGMENT_PATTERNS[index % len(NO_COLOR_SEGMENT_PATTERNS)] for index in range(len(categories))]
    return [fill for fill, _hatch in patterns], [hatch for _fill, hatch in patterns]


def _stack_row(
    ax: plt.Axes,
    position: float,
    shares: dict[str, float],
    categories: tuple[str, ...],
    colours: list[str],
    *,
    label_segments: bool,
    height: float = 0.7,
    hatches: list[str] | None = None,
    no_color: bool = False,
) -> list[tuple[float, str]]:
    """One 100%-wide bar, drawn left to right in *categories* order.

    Returns the ``(centre, number)`` of every segment too narrow to hold its own number, for
    a caller that wants to set them outside; an empty list when nothing was too narrow or
    when *label_segments* is off.
    """
    narrow: list[tuple[float, str]] = []
    left = 0.0
    hatches = hatches or [""] * len(categories)
    # A white seam separates coloured segments; without colour the fills are white and ink,
    # so the seam is ink.  matplotlib draws the hatch in the edge colour too, which is what
    # puts the pattern at full contrast against the fill behind it.
    seam = NO_COLOR_INK if no_color else "white"
    for category, colour, hatch in zip(categories, colours, hatches, strict=True):
        share = shares.get(category, 0.0)
        if not share:
            continue
        ax.barh(
            position,
            share,
            left=left,
            height=height,
            color=colour,
            edgecolor=seam,
            linewidth=0.6,
            hatch=hatch,
            zorder=2,
            # The first segment starts at 0 and the last ends at 1, which are the axis
            # limits, so their outer edges straddle the clip boundary and the bar comes out
            # open at one end.  Nothing here needs clipping -- every segment is inside the
            # axis by construction -- so the outline is drawn whole instead.
            clip_on=False,
        )
        if label_segments and share < MIN_LABELLED_SHARE:
            narrow.append((left + share / 2, f"{share:.0%}"))
        elif label_segments:
            # The ramp crosses the contrast threshold partway along, so one lettering colour
            # cannot serve the whole row.  Measured per segment rather than by position, so
            # an interpolated ramp is handled as correctly as the four fixed steps.
            dark = _relative_luminance(colour) < LABEL_ON_DARK_BELOW
            ax.text(
                left + share / 2,
                position,
                f"{share:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                # Measured against the fill the number lands on, which is what makes the
                # filled pattern legible: white lettering there, ink on every other segment
                # of a colourless row, and the usual grey when there is colour to sit on.
                color="white" if dark else (NO_COLOR_INK if no_color else LABEL_COLOUR),
                zorder=3,
                # A hatch runs straight through lettering and leaves it unreadable, so the
                # label clears itself a patch of the segment's own fill to sit on.  The fill
                # being white is what makes that patch read as a gap in the pattern.
                bbox={"facecolor": colour, "edgecolor": "none", "pad": 2.0} if hatch else None,
            )
        left += share
    return narrow


def _spread_labels(positions: list[float], *, min_gap: float, upper: float = 1.0) -> list[float]:
    """Nudge *positions* apart to *min_gap*, keeping their order and the right-hand edge.

    A forward pass opens the gaps, which can push the last label past *upper*; the whole run
    then slides back by the overshoot and a reverse pass reopens any gap the slide closed.
    Order is preserved throughout, so a label never crosses the leader of its neighbour --
    which is the thing that would make the leaders unreadable rather than merely tight.

    Enough room is assumed: labels needing more than *upper* between them all come back
    evenly spaced from the left instead, which is as good as the space allows.
    """
    if not positions:
        return []
    placed = list(positions)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + min_gap)

    overshoot = placed[-1] - upper
    if overshoot > 0:
        placed = [position - overshoot for position in placed]
        for index in range(len(placed) - 2, -1, -1):
            placed[index] = min(placed[index], placed[index + 1] - min_gap)
    if placed[0] < 0.0:
        return [index * min_gap for index in range(len(placed))]
    return placed


def _label_narrow_segments(ax: plt.Axes, narrow: list[tuple[float, str]]) -> None:
    """Set the numbers of the too-narrow segments below the bar, each on a leader.

    The leader runs from the segment's own centre to wherever its number ended up, so a
    number nudged sideways still says which sliver it belongs to.
    """
    placed = _spread_labels(
        [centre for centre, _text in narrow], min_gap=NARROW_LABEL_MIN_GAP, upper=NARROW_LABEL_RIGHT_LIMIT
    )
    transform = ax.get_xaxis_transform()
    for (centre, text), position in zip(narrow, placed, strict=True):
        ax.plot(
            [centre, position],
            [NARROW_LABEL_LEADER_TOP, NARROW_LABEL_LEADER_FOOT],
            color=AXIS_COLOUR,
            linewidth=0.8,
            transform=transform,
            clip_on=False,
            zorder=3,
        )
        ax.text(
            position,
            NARROW_LABEL_Y,
            text,
            transform=transform,
            ha="center",
            va="top",
            fontsize=8,
            color=LABEL_COLOUR,
            clip_on=False,
        )
