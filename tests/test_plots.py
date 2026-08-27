"""Tests for the drawing decisions that fail silently or late.

Most of :mod:`plots` is checked by looking at what it draws.  These two are not:
a group count the ramp cannot serve raises only when the figure is drawn, and a label whose
lettering is the wrong side of the contrast threshold is legible enough on screen to pass a
glance and unreadable in print.
"""

from __future__ import annotations

from math import ceil

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.data_analysis import CATEGORIES, SUBCATEGORIES  # noqa: E402
from plots.error_composition import BRACE_LABEL, BRACE_LINE, _category_brace  # noqa: E402
from plots.marks import (  # noqa: E402
    FIELD_TYPE_MARKERS,
    NO_COLOR_EDGE_WIDTH,
    NO_COLOR_INK_DIAMETER,
    NO_COLOR_OUTLINE_REACH,
    _mark_style,
    _no_color_marker_size,
    _run_marks,
)
from plots.segments import (  # noqa: E402
    CATEGORY_RAMP,
    NO_COLOR_SEGMENT_FILL,
    NO_COLOR_SEGMENT_PATTERNS,
    _segment_colours,
    _segment_fills,
    _spread_labels,
    _stack_row,
)
from plots.theme import (  # noqa: E402
    LABEL_ON_DARK_BELOW,
    LEGEND_MAX_PER_LINE,
    NO_COLOR_INK,
    _legend_columns,
    _relative_luminance,
)


class TestSegmentColours:
    def test_every_group_gets_a_colour_at_any_count(self) -> None:
        # _stack_row zips groups against colours with strict=True, so a short list raises
        # at draw time rather than here.  The combined order already needs five.
        for count in range(1, 9):
            groups = tuple(f"group {index}" for index in range(count))
            assert len(_segment_colours(groups)) == count

    def test_the_orders_the_figures_use_are_served(self) -> None:
        # The corpus bar stacks the seven sub-categories, past the ramp's four fixed steps.
        for order in (CATEGORIES, SUBCATEGORIES):
            assert len(_segment_colours(order)) == len(order)

    def test_the_colours_darken_monotonically(self) -> None:
        # The ramp is the severity scale, so a step that got lighter would say the opposite
        # of what it means.  Checked for the interpolated counts as well as the fixed ones.
        for count in (3, 4, 5, 6):
            groups = tuple(f"group {index}" for index in range(count))
            luminances = [_relative_luminance(colour) for colour in _segment_colours(groups)]
            assert luminances == sorted(luminances, reverse=True), (count, luminances)

    def test_the_darkest_step_ends_every_ramp(self) -> None:
        for count in (2, 3, 4, 5):
            groups = tuple(f"group {index}" for index in range(count))
            assert _segment_colours(groups)[-1].lower() == CATEGORY_RAMP[-1].lower()


def _drawn_extent(marker: str, style: dict[str, object]) -> tuple[float, float]:
    """The width and height of the ink *style* puts on the page, in points."""
    import numpy as np
    from matplotlib import pyplot as plt

    dpi = 600
    fig = plt.figure(figsize=(1.4, 1.4), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot(0.5, 0.5, marker, **style)
    fig.canvas.draw()
    grey = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].mean(axis=2)
    plt.close(fig)

    ink = grey < 250
    rows = np.where(ink.any(axis=1))[0]
    columns = np.where(ink.any(axis=0))[0]
    per_point = dpi / 72.0
    return (columns[-1] - columns[0] + 1) / per_point, (rows[-1] - rows[0] + 1) / per_point


class TestSegmentFills:
    def test_colour_on_leaves_the_hatches_empty(self) -> None:
        colours, hatches = _segment_fills(SUBCATEGORIES, no_color=False)
        assert colours == _segment_colours(SUBCATEGORIES)
        assert set(hatches) == {""}

    def test_colour_off_gives_every_segment_a_fill_and_a_hatch(self) -> None:
        for order in (CATEGORIES, SUBCATEGORIES):
            colours, hatches = _segment_fills(order, no_color=True)
            assert len(colours) == len(hatches) == len(order)

    def test_no_two_neighbours_share_a_pattern(self) -> None:
        # Open and filled both carry no hatch, so it is the pair that has to differ -- the
        # hatch alone would call those two the same.
        colours, hatches = _segment_fills(SUBCATEGORIES, no_color=True)
        pairs = list(zip(colours, hatches, strict=True))
        assert all(a != b for a, b in zip(pairs, pairs[1:], strict=False))

    def test_the_two_plain_patterns_lead_with_one_hatch_between_them(self) -> None:
        # Open first and filled third are the two a reader never has to decode.  Filled
        # waits a place because it is the heaviest mark on the page and the second segment
        # is the widest category on this corpus.
        colours, hatches = _segment_fills(SUBCATEGORIES, no_color=True)
        assert (colours[0], hatches[0]) == (NO_COLOR_SEGMENT_FILL, "")
        assert colours[1] == NO_COLOR_SEGMENT_FILL and hatches[1] != ""
        assert (colours[2], hatches[2]) == (NO_COLOR_INK, "")

    def test_a_hatched_segment_is_white_behind_its_hatch(self) -> None:
        # The hatch is drawn in the edge colour, so a fill of anything but white spends
        # contrast on a distinction the hatch is already making.  The filled pattern is the
        # exception, and carries no hatch to compete with.
        colours, hatches = _segment_fills(SUBCATEGORIES, no_color=True)
        assert all(colour == NO_COLOR_SEGMENT_FILL for colour, hatch in zip(colours, hatches, strict=True) if hatch)

    def test_every_fill_takes_lettering_that_reads_on_it(self) -> None:
        # White on the filled pattern, ink on the rest: the number is placed by measuring
        # the fill it lands on, so every fill has to sit clearly one side of the threshold.
        colours, _hatches = _segment_fills(SUBCATEGORIES, no_color=True)
        for colour in colours:
            assert abs(_relative_luminance(colour) - LABEL_ON_DARK_BELOW) > 0.1

    def test_the_pattern_table_is_long_enough_for_the_taxonomy(self) -> None:
        assert len(NO_COLOR_SEGMENT_PATTERNS) >= len(SUBCATEGORIES)


class TestNoColorMarkerSize:
    def test_the_two_fills_are_corrected_by_the_same_reach(self) -> None:
        # matplotlib centres a border on the marker's path, so it lands half inside the
        # black and half outside.  The hollow mark's border is the ink and gains that outer
        # half; the solid mark's is white and loses the inner half.  Left at one size, the
        # pair would read as two weights rather than as two conditions.
        for marker, reach in NO_COLOR_OUTLINE_REACH.items():
            solid = _no_color_marker_size(marker, filled=True)
            hollow = _no_color_marker_size(marker, filled=False)
            assert solid - NO_COLOR_INK_DIAMETER == pytest.approx(reach * NO_COLOR_EDGE_WIDTH)
            assert NO_COLOR_INK_DIAMETER - hollow == pytest.approx(reach * NO_COLOR_EDGE_WIDTH)

    def test_the_hollow_mark_stays_large_enough_to_letter(self) -> None:
        # The correction shrinks the mark, and a letter is written inside it at 6.5pt.
        for marker in FIELD_TYPE_MARKERS.values():
            assert _no_color_marker_size(marker, filled=False) > 8.0

    def test_every_shape_and_fill_draws_to_the_ink_diameter(self) -> None:
        # The point of the correction is a property of the page, not of the arithmetic:
        # measured off a render, so a change in how matplotlib strokes a border is caught
        # here rather than in a figure nobody re-measures.  The solid mark's white border
        # is invisible against the page, so what is measured is the black either way.
        marks = dict(_run_marks(("baseline",), ("arms-agent",), no_color=True))
        widths = []
        for marker in FIELD_TYPE_MARKERS.values():
            for run in ("baseline", "arms-agent"):
                widths.extend(_drawn_extent(marker, _mark_style(marks[run], marker)))
        # Within a third of a point of each other and of what they are aiming at: below
        # what a reader can see, and above the pixel the measurement is quantised to.
        assert max(widths) - min(widths) < 0.34, sorted(widths)
        assert max(abs(width - NO_COLOR_INK_DIAMETER) for width in widths) < 0.34, sorted(widths)

    def test_the_two_groups_take_the_two_fills(self) -> None:
        # Two runs in the baseline group -- two repetitions of it, say -- so the fill is
        # shown to belong to the group rather than to the run.
        marks = dict(_run_marks(("baseline", "baseline-2"), ("arms-agent",), no_color=True))
        assert marks["baseline"].fill is False
        assert marks["baseline-2"].fill is False
        assert marks["arms-agent"].fill is True

    def test_colour_on_leaves_one_size_for_every_run(self) -> None:
        # The giveback is a no-colour concern only: in colour both groups are filled, so
        # neither is spending a border on ink and the sizes have nothing to reconcile.
        marks = dict(_run_marks(("baseline", "baseline-2"), ("arms-agent",), no_color=False))
        assert {mark.fill for mark in marks.values()} == {None}
        assert len({mark.style["markersize"] for mark in marks.values()}) == 1


class TestLegendColumns:
    def test_no_line_holds_more_than_the_maximum(self) -> None:
        # A row holds one key from each column, so the column count is the keys per line.
        for n_keys in range(1, 20):
            assert _legend_columns(n_keys) <= LEGEND_MAX_PER_LINE

    def test_the_taxonomy_s_keys_come_out_as_expected(self) -> None:
        assert _legend_columns(len(CATEGORIES)) == 3  # three keys, one line
        assert _legend_columns(len(SUBCATEGORIES)) == 4  # seven keys, four then three

    def test_few_keys_take_one_line(self) -> None:
        for n_keys in range(1, LEGEND_MAX_PER_LINE + 1):
            assert ceil(n_keys / _legend_columns(n_keys)) == 1

    def test_lines_grow_with_the_key_count(self) -> None:
        # The line budget is not fixed: squeezing a key to save a line is the worse trade.
        assert ceil(9 / _legend_columns(9)) == 3

    def test_no_count_asks_for_zero_columns(self) -> None:
        assert _legend_columns(0) == 1


class TestBarOutline:
    def _edge_ink(self, x: float) -> int:
        """Ink in the two pixel columns at data *x*, over a bar drawn from 0 to 1."""
        fig, ax = plt.subplots(figsize=(6, 1), dpi=100)
        _stack_row(
            ax,
            0.0,
            {"a": 0.5, "b": 0.5},
            ("a", "b"),
            ["white", "white"],
            label_segments=False,
            hatches=["", "///"],
            no_color=True,  # an ink seam, so the outline is there to be measured
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.6, -0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())
        column = int(ax.transData.transform((x, 0))[0])
        plt.close(fig)
        dark = pixels[:, :, :3].mean(axis=2) < 128
        return int(dark[:, max(column - 1, 0) : column + 1].sum())

    def test_both_outer_edges_of_the_bar_are_drawn(self) -> None:
        # The first segment starts at the axis's left limit and the last ends at its right,
        # so the outer edges straddle the clip boundary.  matplotlib's clip box keeps the
        # left one and drops the right, which leaves the bar visibly open at one end.
        assert self._edge_ink(0.0) > 0
        assert self._edge_ink(1.0) > 0

    def test_neither_outer_edge_is_a_sliver_of_the_other(self) -> None:
        # Not pixel-exact: the two limits land on different sub-pixel offsets, so the
        # antialiased line spreads differently across the columns sampled.  What the clipped
        # bar looked like was 2 against 48, which this catches and rounding does not trip.
        left, right = self._edge_ink(0.0), self._edge_ink(1.0)
        assert min(left, right) > 0.5 * max(left, right), (left, right)


class TestCategoryBrace:
    def _placed(self, left: float, right: float, label: str) -> tuple[float, bool]:
        """Where the brace put its label, on the bar the corpus figure actually draws."""
        fig, ax = plt.subplots(figsize=(9.6, 2.5))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.6, -0.6)
        _category_brace(ax, left, right, label)
        text = ax.texts[-1]
        placement = (text.get_position()[1], text.get_bbox_patch() is not None)
        plt.close(fig)
        return placement

    def test_a_label_with_room_breaks_the_rule(self) -> None:
        y, has_patch = self._placed(0.0, 0.5, "FN")
        assert y == BRACE_LINE
        # The patch is what masks the rule the label is standing in.
        assert has_patch

    def test_a_label_with_no_room_sits_above_the_rule(self) -> None:
        # No patch up there: it would punch a hole in whatever the label sits over.
        y, has_patch = self._placed(0.0, 0.01, "FP + FN")
        assert y == BRACE_LABEL
        assert not has_patch

    def test_the_narrowest_span_the_corpus_draws_keeps_its_label_inline(self) -> None:
        # Insertions are 12.7% of the deduplicated corpus and carry the shortest label of
        # the three.  Judged on the span alone, that went above the rule while its
        # neighbours stayed in theirs -- three braces, one of them inexplicably different.
        assert self._placed(0.0, 0.127, "FP")[0] == BRACE_LINE

    def test_the_longest_label_still_fits_its_own_span(self) -> None:
        # "FP + FN" is three times the width of "FP", so it is the one that decides whether
        # a single rule can serve both.  Its span is 21% of the same bar.
        assert self._placed(0.0, 0.212, "FP + FN")[0] == BRACE_LINE


class TestSpreadLabels:
    def test_positions_already_apart_are_left_alone(self) -> None:
        assert _spread_labels([0.1, 0.5, 0.9], min_gap=0.05) == [0.1, 0.5, 0.9]

    def test_crowded_positions_are_opened_to_the_gap(self) -> None:
        placed = _spread_labels([0.40, 0.41, 0.42], min_gap=0.05)
        assert all(b - a >= 0.05 - 1e-9 for a, b in zip(placed, placed[1:], strict=False))

    def test_the_order_is_never_crossed(self) -> None:
        # A label crossing its neighbour's leader is what makes leaders unreadable, rather
        # than merely tight.
        placed = _spread_labels([0.94, 0.95, 0.96], min_gap=0.06)
        assert placed == sorted(placed)

    def test_nothing_is_pushed_past_the_right_edge(self) -> None:
        placed = _spread_labels([0.94, 0.97, 0.99], min_gap=0.06, upper=0.98)
        assert placed[-1] <= 0.98 + 1e-9
        assert all(b - a >= 0.06 - 1e-9 for a, b in zip(placed, placed[1:], strict=False))

    def test_more_labels_than_room_come_back_evenly_spaced(self) -> None:
        # Not a case this figure reaches, but the fallback has to be defined: sliding the
        # run left far enough would otherwise take it off the left edge instead.
        placed = _spread_labels([0.9] * 12, min_gap=0.1, upper=1.0)
        assert placed[0] >= 0.0
        assert all(b - a >= 0.1 - 1e-9 for a, b in zip(placed, placed[1:], strict=False))

    def test_no_labels_is_no_positions(self) -> None:
        assert _spread_labels([], min_gap=0.05) == []


class TestLabelContrast:
    def test_the_threshold_splits_the_fixed_ramp_where_it_always_did(self) -> None:
        # The rule was positional -- the last two steps took white -- before it was measured.
        # The measurement has to agree on the ramp it replaced, or every existing figure
        # would change without anyone asking for it.
        on_dark = [_relative_luminance(colour) < LABEL_ON_DARK_BELOW for colour in CATEGORY_RAMP]
        assert on_dark == [False, False, True, True]

    def test_black_and_white_land_on_the_right_sides(self) -> None:
        assert _relative_luminance("#ffffff") > LABEL_ON_DARK_BELOW
        assert _relative_luminance("#000000") < LABEL_ON_DARK_BELOW
