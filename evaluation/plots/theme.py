"""The page every figure is drawn on: its greys, its type, and how it is finished.

Nothing here says what a figure *means*.  The encodings live beside the marks they make --
:mod:`plots.marks` for the runs, :mod:`plots.segments` for the stacked bars -- and what is
collected here is everything a reader is not supposed to notice: the grid, the frame, the
tick colours, the sizes of type, and the one place a finished figure is written or shown.

The greys are one ramp, listed light to dark, and each is placed by how far forward its
job belongs.  Keeping them in one list is the point: a colour chosen against its
neighbours stays legible as a choice, where a hex code written where it is used only
looks arbitrary.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

#: Recessive chart furniture: the grid and axes sit behind the data, not beside it.
GRID_COLOUR = "#e6e6e3"

#: The stem a mark sits on, drawn from a panel's left edge in the difficulty figure.
#: Lighter than the mark and heavier than the grid: it carries the reading distance, not
#: the reading.
STEM_COLOUR = "#d0d0ca"

AXIS_COLOUR = "#c9c9c4"

#: The F1 contours are not furniture -- they are a scale the reader measures against --
#: so they sit a step darker than the grid while staying well behind the marks.
CONTOUR_COLOUR = "#a8a8a1"
CONTOUR_LABEL_COLOUR = "#7d7d76"

#: The frame and ticks of a precision/recall panel.  Darker than the grid, which is
#: furniture, and lighter than the marks, which are the reading.
PANEL_FRAME_COLOUR = "#59595a"

LABEL_COLOUR = "#52514e"

#: Ink for every mark when colour is off.
NO_COLOR_INK = "#0b0b0b"

#: The figure's four levels of type, each a clear step from the one under it, so they are
#: told apart by size rather than by position alone.  Set for a figure printed at about
#: half a page rather than read on screen: the tick numbers are the floor, and everything
#: else is placed above them.
FIGURE_TITLE_SIZE = 14
PANEL_TITLE_SIZE = 12
LEGEND_TEXT_SIZE = 12
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9.5

#: Height the furniture takes whatever the grid holds: the column titles above it, the
#: x-axis label below, and the legend strip under that.  Added on top of the rows'
#: allowance rather than taken out of it, so one row of panels comes out the size of a
#: row in a grid of twelve instead of being crowded out by the chrome around it.
PANEL_CHROME_INCHES = 1.4

#: Height of one legend row, in inches.  Kept physical rather than fractional so a tall
#: figure does not reserve a proportionally huge strip for the same two lines of keys.
#: Enough for a key at :data:`LEGEND_TEXT_SIZE` beside a mark drawn full size.
LEGEND_LINE_INCHES = 0.36

#: How many keys one line of the legend may hold.  Four: names this long run into each
#: other past that at a readable size, and the line count follows from the key count rather
#: than being fixed -- a key squeezed to fit a line budget is worse than an extra line.
LEGEND_MAX_PER_LINE = 4

#: Background luminance at which white lettering overtakes :data:`LABEL_COLOUR` ink for
#: contrast.  Derived rather than eyeballed: it is where the two contrast ratios against a
#: segment fill cross, so each segment's label takes whichever of the two reads better.
LABEL_ON_DARK_BELOW = 0.33


def _relative_luminance(colour: str) -> float:
    """The WCAG relative luminance of *colour*."""
    channels = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in to_rgb(colour)
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _legend_columns(n_keys: int) -> int:
    """Columns enough to keep a legend line to :data:`LEGEND_MAX_PER_LINE` keys.

    matplotlib fills a legend column by column, so a row holds one key from each column and
    the column count *is* the keys per line; the rows then follow from how many keys there
    are.  Seven keys come out four and three, three keys come out on one line.
    """
    return max(1, min(LEGEND_MAX_PER_LINE, n_keys))


def _finish(fig: plt.Figure, save_path: str | None) -> None:
    """Write the figure to *save_path*, or show it when no path is given."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
