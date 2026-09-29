"""How a condition and a field type become a mark.

Two variables have to share one panel of the precision/recall figures, and they are split
across channels that do not interfere: the **condition** takes the colour (or, with colour off,
the fill and a letter), and the **field type** takes the marker shape.  Keeping both here
is what stops a figure inventing its own vocabulary -- a condition is the same colour, and a
field type the same shape, in every figure either appears in.

:mod:`plots.segments` is the same idea for the stacked bars, whose segments are categories
rather than conditions and so need a scale rather than a pair of hues.
"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from plots.theme import NO_COLOR_INK

#: One colour per condition, in the order the conditions are drawn, so a condition keeps
#: its colour whichever figure it appears in.  The pair clears colour-vision-deficiency
#: separation; the orange falls below 3:1 against a white page, which is why the bars
#: carry their value as text rather than relying on the fill alone.
CONDITION_COLOURS = ("#4472C4", "#ED7D31")

#: What each condition is called wherever a figure names it: legend keys and axis labels.
#: The keys are the directory names the runs are written under, which read as code; the
#: manuscript names the two arms "Baseline" and "ARMS", and each ablation after the one tool
#: it keeps.  A condition missing here is shown
#: under its own name.
CONDITION_LABELS = {
    "baseline": "Baseline",
    "template-tool": "GetTemplate-tool Only",
    "term-tool": "TermSearch-tool Only",
    "arms-agent": "ARMS",
}


def condition_label(condition: str) -> str:
    """The name a figure shows for *condition*."""
    return CONDITION_LABELS.get(condition, condition)


#: A group holding several conditions is *ordinal* -- swapping two of them would change what
#: they mean -- so the group takes one hue in monotone lightness steps rather than
#: unrelated hues.  The light end clears 2:1 on a white page.  The other group is a
#: different mechanism, not a further step, so it keeps the hue it has in every figure.
LADDER_BLUES = ("#8FB5E2", "#5A8CCB", "#2E5FA3")

#: The same, in the system group's hue, for when several system conditions are compared.
LADDER_ORANGES = ("#ED7D31", "#C25A11", "#8C4109")

#: Marker per field type, so field types can share one panel without spending the
#: colour channel, which belongs to the condition.
FIELD_TYPE_MARKERS = {"ontology": "o", "non_ontology": "s", "all": "^"}

#: What each field type is called wherever it is written -- legend keys, column titles
#: and axis labels alike, so one name is learned rather than three.
FIELD_TYPE_LABELS = {
    "all": "All Fields",
    "ontology": "Ontology-constrained Fields",
    "non_ontology": "Non-ontology-constrained Fields",
}

#: The field-type keys, which carry shape rather than identity.  Dark grey rather than
#: black: dark enough to read the marker's outline, quiet enough not to outrank the
#: coloured condition keys beside it.  Drawn a little under the marks in the panels, since a key
#: has only to be identified and not compared.
FIELD_KEY_COLOUR = "#3f3f3a"
FIELD_KEY_SIZE = 9


def _condition_colours(n_conditions: int, ramp: tuple[str, ...], solo: str) -> list[str]:
    """One colour per condition in a group: the solid hue for a lone condition, the ramp for several.

    A group of one is a coordinate, not a progression, so it keeps the flat hue it has
    in every other figure rather than borrowing a step from an ordinal ramp.
    """
    if n_conditions == 1:
        return [solo]
    if n_conditions <= len(ramp):
        return list(ramp[len(ramp) - n_conditions :])
    cmap = plt.get_cmap("Blues" if ramp is LADDER_BLUES else "Oranges")
    return [cmap(step) for step in np.linspace(0.42, 0.85, n_conditions)]


class ConditionMark(NamedTuple):
    """How one condition is drawn: matplotlib keywords, the letter naming its place, and -- with
    colour off -- whether the mark is solid, which its size hangs on.

    *fill* is ``None`` when the size sits in *style* itself; every mark the figures draw
    sets it, in colour as well as without, so that each is sized for its shape.
    """

    style: dict[str, object]
    letter: str | None
    fill: bool | None = None


#: The black width every mark comes out, whichever condition it stands for and whatever
#: shape it takes.  Order is carried by the letter rather than by area, so marks of one
#: size are left to be compared shape for shape.
NO_COLOR_INK_DIAMETER = 12.4

#: The border both marks carry: the hollow mark's is the ink it is made of, the solid
#: mark's is white and is what gives two overlapping solid marks a seam.  Wide enough to
#: read as a mark rather than as a hairline at print size, narrow enough that the hollow
#: one stays a ring rather than filling in.
#:
#: The white border is drawn on the mark rather than under it, so half of it falls inside
#: the black and only half is left to separate with.  That is the whole of the seam, and
#: it is deliberately little: everything a wider one covers is some other condition's mark, so
#: it would erase a mark that lands beside this one rather than merely separating it.
#: Two conditions at the same point collapse to one mark either way -- nothing drawn in the
#: same place can show both.
NO_COLOR_EDGE_WIDTH = 1.4

#: How far a stroked border reaches past the marker's nominal size, in border widths.
#: matplotlib centres a border on the marker's path, so half of it falls outside -- but
#: only where the outline is everywhere the same distance from the middle.  A mitred
#: corner reaches further, and the sharper the corner the further it goes.  Measured:
#: 1.01 for the circle and 0.99 for the square against 1.62 for the triangle, whose apex
#: is the sharpest corner of the three.
NO_COLOR_OUTLINE_REACH = {"o": 1.0, "s": 1.0, "^": 1.62}


def _no_color_marker_size(marker: str, *, filled: bool) -> float:
    """The nominal size that draws *marker* :data:`NO_COLOR_INK_DIAMETER` wide.

    Both marks carry a border, and matplotlib strokes it centred on the marker's path, so
    it lands half inside the black and half outside.  On the hollow mark the border *is*
    the black, so its outer half adds to the width; on the solid mark the border is white,
    so its inner half takes width away.  One nominal size would therefore draw two
    different widths, and a pair that differs only in which condition it stands for would
    read as differing in weight.  So the hollow mark is drawn that much smaller and the
    solid one that much larger, and the two come out matching.

    The correction is the same distance either way and holds for all three shapes, because
    offsetting a marker's outline scales it about its incentre rather than moving it a
    fixed distance: what differs between the shapes is only how many border widths that
    scale is worth, which is what :data:`NO_COLOR_OUTLINE_REACH` records.
    """
    reach = NO_COLOR_OUTLINE_REACH.get(marker, 1.0) * NO_COLOR_EDGE_WIDTH
    return NO_COLOR_INK_DIAMETER + reach if filled else NO_COLOR_INK_DIAMETER - reach


def _colour_mark(colour: str) -> ConditionMark:
    """A solid mark in *colour*, drawn to the same width as a solid mark with colour off.

    The same white border and the same size: colour changes what a mark says, not how much
    of the panel it takes, so a figure keeps its weight whether it is printed in colour or not.
    """
    style = {
        "color": colour,
        "markerfacecolor": colour,
        "markeredgecolor": "white",
        "markeredgewidth": NO_COLOR_EDGE_WIDTH,
    }
    return ConditionMark(style, None, fill=True)


#: How far forward each marker shape is drawn: the square at the back, the circle over it,
#: the triangle in front.  Largest to smallest, so three field types landing on one point
#: still show as three shapes -- each one's corners or rim stand out behind the next.  Every
#: value sits between the contours (0) and the letters written on the marks (3).
MARKER_ZORDER = {"s": 2.0, "o": 2.1, "^": 2.2}


def _mark_style(mark: ConditionMark, marker: str) -> dict[str, object]:
    """*mark*'s keywords, sized for the shape it is about to be drawn in."""
    if mark.fill is None:
        return mark.style
    return {**mark.style, "markersize": _no_color_marker_size(marker, filled=mark.fill)}


def _condition_marks(
    baselines: tuple[str, ...],
    systems: tuple[str, ...],
    *,
    no_color: bool,
) -> list[tuple[str, ConditionMark]]:
    """Every condition in the order it is drawn, with the marker it takes.

    Three things need telling apart in one panel -- which condition, which field type,
    and where a condition sits in the sequence -- and shape can carry only one of them.  With
    colour off they are split across the channels that survive greyscale, print and
    colour-vision deficiency:

    * **field type** keeps the marker shape, as it has in colour;
    * **condition** is the fill, the baseline group hollow against the system group
      solid.  The two keep one shape, so a pair reads as a pair;
    * **order** is a capital letter written inside the mark, running ``A``, ``B``, ``C``
      across the groups -- a letter states a condition's place exactly where a size only
      suggests it.  Only a group holding several conditions is lettered: a group of one has no
      order to report, and a lone ``A`` beside it would invite the reader to look for a
      sequence that is not there.

    A solid mark takes a white border and a hollow one a dark border: two solid marks that
    overlap would otherwise merge into a blob with no seam to read, and a white border on a
    white fill would leave nothing to see at all.  The two then need different nominal
    sizes to come out the same width, and each shape a different size again -- see
    :func:`_no_color_marker_size`.
    """
    marks: list[tuple[str, ConditionMark]] = []
    lettered = 0
    for conditions, ramp, solo, filled in (
        (baselines, LADDER_BLUES, CONDITION_COLOURS[0], False),
        (systems, LADDER_ORANGES, CONDITION_COLOURS[1], True),
    ):
        if not no_color:
            for condition, colour in zip(conditions, _condition_colours(len(conditions), ramp, solo), strict=True):
                marks.append((condition, _colour_mark(colour)))
            continue

        for condition in conditions:
            style = {
                "color": NO_COLOR_INK,
                "markerfacecolor": NO_COLOR_INK if filled else "white",
                "markeredgecolor": "white" if filled else NO_COLOR_INK,
                "markeredgewidth": NO_COLOR_EDGE_WIDTH,
            }
            # Only the lettered conditions take a place in the sequence, so a lettered group
            # reads A, B, C whether or not the other group is a single condition.
            letter = chr(ord("A") + lettered) if len(conditions) > 1 else None
            lettered += bool(letter)
            marks.append((condition, ConditionMark(style, letter, fill=filled)))
    return marks


#: One colour per model, in the order the models are listed, for the figures that compare
#: models under one condition.  None of the three is a condition's hue, so a model is never
#: mistaken for the baseline or ARMS.  Violet, aqua and yellow clear colour-vision-deficiency
#: separation between every pair (the check a scatter needs, not only neighbours); the aqua
#: and the yellow fall below 3:1 against a white page, so a model is always named in the
#: legend rather than left to its colour.  A fourth model would need a fourth hue validated
#: with them, not one generated.
MODEL_COLOURS = ("#4a3aa7", "#1baf7a", "#eda100")


def _model_marks(models: tuple[str, ...], *, no_color: bool) -> list[tuple[str, ConditionMark]]:
    """Every model in the order given, with the mark it takes.

    In colour, each model takes its hue from :data:`MODEL_COLOURS`.  With colour off the model
    is the letter written inside its mark, ``A``, ``B``, ``C``, and the marks alternate solid
    and hollow so that overlapping ones stay apart.  A lone model has no letter, as a lone
    condition has none.

    Raises:
        ValueError: If *models* is empty, names a model twice, or holds more models than
            :data:`MODEL_COLOURS` has colours.
    """
    if not models:
        raise ValueError("models needs at least one model")
    if len(set(models)) != len(models):
        raise ValueError(f"Each model may be named once, got {list(models)}")
    if len(models) > len(MODEL_COLOURS):
        raise ValueError(f"At most {len(MODEL_COLOURS)} models can be told apart by colour, got {len(models)}")
    if not no_color:
        return [(model, _colour_mark(colour)) for model, colour in zip(models, MODEL_COLOURS, strict=False)]
    marks = []
    for index, model in enumerate(models):
        # Solid and hollow in turn, as the baseline and system groups are, so two models
        # that land on one spot still show as two marks rather than one black blob.
        filled = index % 2 == 0
        style = {
            "color": NO_COLOR_INK,
            "markerfacecolor": NO_COLOR_INK if filled else "white",
            "markeredgecolor": "white" if filled else NO_COLOR_INK,
            "markeredgewidth": NO_COLOR_EDGE_WIDTH,
        }
        letter = chr(ord("A") + index) if len(models) > 1 else None
        marks.append((model, ConditionMark(style, letter, fill=filled)))
    return marks
