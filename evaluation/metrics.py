"""Evaluation metrics for comparing agent output against gold standard.

Metrics:
    - Overall accuracy: overall record-level agreement across all fields in the
      gold standard.  Both-null counts as a match; any difference in value or
      presence counts as a mismatch.
"""

from __future__ import annotations

from typing import Any


def compute_accuracy(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> float:
    """Compute accuracy of *predicted* metadata against *gold*.

    Accuracy measures overall record-level agreement: the fraction of gold
    fields where both records agree.  Two fields agree when:

    * both values are missing (``None``), or
    * both values are non-missing and match via ``_values_match()``.

    The denominator is all keys present in *gold*.

    Returns 0.0 when *gold* has no fields.
    """
    if not gold:
        return 0.0
    matches = 0
    for k in gold:
        gold_val = gold[k]
        pred_val = predicted.get(k)
        gold_missing = _is_missing(gold_val)
        pred_missing = _is_missing(pred_val)
        if (gold_missing and pred_missing) or (
            not gold_missing
            and not pred_missing
            and _values_match(pred_val, gold_val, match_case=match_case, match_whole_word=match_whole_word)
        ):
            matches += 1
    return matches / len(gold)


def _values_match(
    predicted_val: Any,
    gold_val: Any,
    *,
    match_case: bool,
    match_whole_word: bool,
) -> bool:
    """Return ``True`` if *predicted_val* matches *gold_val* under the given flags.

    When either value is not a ``str``, exact equality is used regardless of
    flags.  For string values:

    * *match_case=False* lowercases both strings before comparison.
    * *match_whole_word=False* checks whether the gold value is a **substring
      of** the predicted value (rather than requiring equality).
    """
    if not isinstance(predicted_val, str) or not isinstance(gold_val, str):
        return predicted_val == gold_val

    predicted_str = predicted_val if match_case else predicted_val.lower()
    gold_str = gold_val if match_case else gold_val.lower()

    if match_whole_word:
        return predicted_str == gold_str
    return gold_str in predicted_str


def _is_missing(value: Any) -> bool:
    """Return ``True`` if *value* is considered missing.

    A field value is missing when it is ``None``.  Empty strings, empty lists,
    and other falsy-but-not-None values are considered present.
    """
    return value is None
