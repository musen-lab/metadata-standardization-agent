"""What counts as blank, what counts as equal, and how the two combine.

Every metric in this package routes its decision through :func:`_classify_field`,
which is the single place a gold field and its prediction are compared.
:mod:`analysis.metrics.accuracy` collapses its five cases to a boolean;
:mod:`analysis.metrics.confusion` maps them to confusion-matrix cells.  Because both
derive from one classification rather than restating it, accuracy and
precision/recall cannot disagree about what a blank is or what a match is -- a
divergence that would be silent, since it changes the numbers without raising.
"""

from __future__ import annotations

from typing import Any

# The five ways one gold field and its prediction can relate.  These are comparison
# outcomes, not metric cells: :mod:`analysis.metrics.confusion` maps them to
# ``TP``/``FP``/``FN``/``TN`` (a substitution lands in two cells at once), and
# :mod:`analysis.metrics.accuracy` collapses them to a boolean.
BOTH_BLANK = "both_blank"
MATCH = "match"
SUBSTITUTION = "substitution"
INSERTION = "insertion"
DELETION = "deletion"

#: The cases in which the prediction agrees with gold.
CORRECT_CASES = (BOTH_BLANK, MATCH)


def _normalize_doi(value: str) -> str:
    """Normalize DOI URLs so that doi.org, dx.doi.org, and bare DOIs are equivalent."""
    return value.replace("https://doi.org/", "https://dx.doi.org/")


def _values_match(
    predicted_val: Any,
    gold_val: Any,
    *,
    match_case: bool,
    match_whole_word: bool,
    field_name: str = "",
) -> bool:
    """Return ``True`` if *predicted_val* matches *gold_val* under the given flags.

    When either value is not a ``str``, exact equality is used regardless of
    flags.  For string values:

    * *match_case=False* lowercases both strings before comparison.
    * *match_whole_word=False* checks whether the gold value is a **substring
      of** the predicted value (rather than requiring equality).
    * When *field_name* ends with ``_doi``, DOI URLs are normalised so that
      ``doi.org`` and ``dx.doi.org`` are treated as equivalent.
    """
    if not isinstance(predicted_val, str) or not isinstance(gold_val, str):
        return predicted_val == gold_val

    predicted_str = predicted_val if match_case else predicted_val.lower()
    gold_str = gold_val if match_case else gold_val.lower()

    if field_name.endswith("_doi"):
        predicted_str = _normalize_doi(predicted_str)
        gold_str = _normalize_doi(gold_str)

    if match_whole_word:
        return predicted_str == gold_str
    return gold_str in predicted_str


def _is_missing(value: Any) -> bool:
    """Return ``True`` if *value* is considered missing.

    A field value is missing when it is ``None`` or a string that is empty or
    contains only whitespace.  ``null`` and ``""`` both mean "no value" in this
    corpus -- the gold standard uses ``""`` for blank fields while the agent is
    prompted to emit ``null`` -- so treating them as distinct would score a
    correct blank as a mismatch in both directions.

    Other falsy-but-not-``None`` values (``0``, ``False``, empty lists) are
    considered present, since they carry meaning as field values.
    """
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _classify_field(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    field_name: str,
    *,
    match_case: bool,
    match_whole_word: bool,
) -> str:
    """Return which of the five cases describes *field_name* in this record pair.

    The positive action is *asserting a value* -- writing something rather than
    leaving the field blank -- so the cases are named for what the prediction did
    relative to gold:

    ===================  ====================  ==================
    gold                 prediction            case
    ===================  ====================  ==================
    blank                blank                 ``BOTH_BLANK``
    blank                has a value           ``INSERTION``
    has a value          blank                 ``DELETION``
    has a value          matches               ``MATCH``
    has a value          a different value     ``SUBSTITUTION``
    ===================  ====================  ==================

    A field absent from *predicted* is read as blank, so an omitted key and an
    explicit ``null`` score identically.
    """
    gold_val = gold[field_name]
    pred_val = predicted.get(field_name)
    gold_missing = _is_missing(gold_val)
    pred_missing = _is_missing(pred_val)

    if gold_missing and pred_missing:
        return BOTH_BLANK
    if gold_missing:
        return INSERTION
    if pred_missing:
        return DELETION
    if _values_match(
        pred_val, gold_val, match_case=match_case, match_whole_word=match_whole_word, field_name=field_name
    ):
        return MATCH
    return SUBSTITUTION


def _is_field_correct(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    field_name: str,
    *,
    match_case: bool,
    match_whole_word: bool,
) -> bool:
    """Return ``True`` when the prediction agrees with gold on *field_name*.

    Agreement means both are blank or both are present and matching.  Accuracy
    credits the both-blank case; precision and recall do not, which is why they
    read :func:`_classify_field` directly instead of going through here.
    """
    return (
        _classify_field(predicted, gold, field_name, match_case=match_case, match_whole_word=match_whole_word)
        in CORRECT_CASES
    )
