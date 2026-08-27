"""Precision, recall and F1 over asserted values.

The positive action is asserting a value, so with ``TP`` right assertions, ``FP``
wrong ones and ``FN`` gold values never asserted::

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = 2 * precision * recall / (precision + recall)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from analysis.metrics.field_roles import _get_ontology_constrained_fields
from analysis.metrics.matching import (
    BOTH_BLANK,
    DELETION,
    INSERTION,
    MATCH,
    SUBSTITUTION,
    _classify_field,
)

if TYPE_CHECKING:
    from pathlib import Path

CONFUSION_CATEGORIES = ("ontology", "non_ontology", "all")

#: The four confusion-matrix cells, then the three error shapes.
CONFUSION_KEYS = ("TP", "FP", "FN", "TN", "insertions", "deletions", "substitutions")

#: Which counters each of the five cases increments.  A substitution lands in three
#: of them: the value asserted is wrong (``FP``), gold's value was not produced
#: (``FN``), and the shape of the error is a substitution rather than an insertion
#: or a deletion.
_CELLS_BY_CASE: dict[str, tuple[str, ...]] = {
    BOTH_BLANK: ("TN",),
    INSERTION: ("FP", "insertions"),
    DELETION: ("FN", "deletions"),
    MATCH: ("TP",),
    SUBSTITUTION: ("FP", "FN", "substitutions"),
}


def compute_field_confusion(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    schema_path: Path,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> dict[str, dict[str, int]]:
    """Classify every gold field into a confusion-matrix cell.

    The unit of evaluation is one field of one record, and the positive action is
    *asserting a value* -- writing something rather than leaving the field blank.
    Which of the five cases a field falls into is decided by
    :func:`~analysis.metrics.matching._classify_field`, and defined in that module
    beside the comparison rules; this function only maps each case onto counters:

    ================  ===============  =================  =======
    case              cell             error shape        MUC
    ================  ===============  =================  =======
    ``BOTH_BLANK``    ``TN``           --                 ``NON``
    ``INSERTION``     ``FP``           ``insertions``     ``SPU``
    ``DELETION``      ``FN``           ``deletions``      ``MIS``
    ``MATCH``         ``TP``           --                 ``COR``
    ``SUBSTITUTION``  ``FP`` & ``FN``  ``substitutions``  ``INC``
    ================  ===============  =================  =======

    The last column names the same five cases in the vocabulary of the Message
    Understanding Conference template-filling evaluations -- correct, incorrect,
    missing, spurious, noncommittal -- which is what the slot-filling literature
    reports (Chinchor, *MUC-4 Evaluation Metrics*, 1992).  ``NON`` is the both-blank
    case, and MUC leaves it out of the precision and recall denominators exactly as
    ``TN`` is left out here.  The mapping is recorded only so the two can be lined up;
    ``TP``/``FP``/``FN``/``TN`` are the names used in the code and in the numbers this
    returns.  MUC's partial-credit category has no counterpart here: a comparison
    either matches or it does not.

    A wrong value counts as both a false positive and a false negative: the value
    asserted is wrong, and the value gold holds was not produced.  Precision and
    recall have different denominators and the two counters are never summed, so
    this double-counts nothing.

    Alongside the four cells, the three error shapes are tracked separately, so
    over-filling and under-filling stay distinguishable::

        FP == insertions + substitutions
        FN == deletions  + substitutions

    The accuracy metrics collapse the same call to a boolean, so the two families can
    never disagree about what a blank is or what a match is.

    Returns a dict keyed by :data:`CONFUSION_CATEGORIES`, each holding the seven
    counters in :data:`CONFUSION_KEYS`.  The ``"all"`` entry is the sum of the
    other two.
    """
    ontology_fields = set(_get_ontology_constrained_fields(schema_path))
    counts = {category: dict.fromkeys(CONFUSION_KEYS, 0) for category in CONFUSION_CATEGORIES}

    for field_name in gold:
        case = _classify_field(predicted, gold, field_name, match_case=match_case, match_whole_word=match_whole_word)
        category = "ontology" if field_name in ontology_fields else "non_ontology"
        for target in (category, "all"):
            for cell in _CELLS_BY_CASE[case]:
                counts[target][cell] += 1

    return counts


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    """Return ``{"precision", "recall", "f1"}`` from a TP/FP/FN counter.

    ``precision`` is the fraction of asserted values that were right,
    ``recall`` the fraction of gold's values that were produced, and ``f1`` their
    harmonic mean.  A zero denominator yields ``0.0``, matching how the accuracy
    functions report empty categories -- so a prediction that asserts nothing
    scores 0.0 on all three rather than raising.
    """
    true_positives = counts.get("TP", 0)
    false_positives = counts.get("FP", 0)
    false_negatives = counts.get("FN", 0)

    asserted = true_positives + false_positives
    expected = true_positives + false_negatives
    precision = true_positives / asserted if asserted else 0.0
    recall = true_positives / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
