"""What each counted error looks like, one error at a time.

:mod:`analysis.data_analysis.errors` labels a mismatch by its *shape* -- a formatting
difference, a boolean written the other way.  This module asks the next question: what did
gold hold, what did the run assert, and what does the legacy record say about how the run
got there?  It reaches for the evidence that can settle it -- the record the run was given,
the template's permissible values, and, when the run kept one, the run's own decision log.

Every error is labelled at two levels, and carries both:

* its **category** is the confusion case, named for what the run did:
  :data:`SUBSTITUTIONS`, :data:`DELETIONS`, :data:`INSERTIONS`.
* its **sub-category** says which of the several ways that came about.

Under :data:`DELETIONS` and :data:`INSERTIONS` the sub-categories split on one question --
whether the legacy record held the value at issue -- so the two mirror each other:
:data:`UNDERESTIMATE_LEGACY_VALUE` left behind what was there, :data:`OVERESTIMATE_LEGACY_VALUE`
took what was not wanted, and :data:`ENTIRELY_DONT_KNOW` and :data:`TOO_OPTIMISTIC_ANSWER`
are the pair where the record said nothing.  A substitution splits on the same question
-- :data:`MISLOCATE_LEGACY_VALUE` against :data:`COMPLETELY_WRONG` -- after
:data:`CLOSE_MATCH` is taken out of it first.

**One row per error.**  The category *is* the confusion case, so a substitution has one
label rather than one per side, and the frame holds one row per (record, field).  Which
side each row costs is the ``costs`` column, and ``FP`` and ``FN`` are still recoverable
from it: :func:`reconcile_with_confusion` checks that they are.

Every row carries a *pointer*: the gold file, the prediction file and the field, so any
label can be opened and argued with.  Nothing here changes what counts as wrong -- that
stays :func:`~analysis.metrics.matching._classify_field`, the rule the scores use.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, load_record
from analysis.metrics import _is_missing
from analysis.metrics.matching import DELETION, INSERTION, SUBSTITUTION, _classify_field

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from analysis.corpus import Assay

#: Which side of the score a row costs.  Read off the confusion case rather than stored
#: twice: an insertion can only cost precision, a deletion only recall, a substitution both.
COSTS_PRECISION = "precision"
COSTS_RECALL = "recall"
COSTS_BOTH = "precision and recall"

COSTS_BY_CASE = {INSERTION: COSTS_PRECISION, DELETION: COSTS_RECALL, SUBSTITUTION: COSTS_BOTH}

#: The whole corpus, where a per-assay frame carries a pooled row beside its assays.
POOLED_ASSAY = "All assays"

# ---------------------------------------------------------------------------
# The categories: the confusion case, named for what the run did.
# ---------------------------------------------------------------------------

#: Gold holds a value and the run asserted a different one.
SUBSTITUTIONS = "substitutions"

#: Gold holds a value and the run asserted nothing.
DELETIONS = "deletions"

#: Gold leaves the field blank and the run asserted something.
INSERTIONS = "insertions"

CATEGORIES = (SUBSTITUTIONS, DELETIONS, INSERTIONS)

CATEGORY_BY_CASE = {SUBSTITUTION: SUBSTITUTIONS, DELETION: DELETIONS, INSERTION: INSERTIONS}

#: Which confusion cells a category lands in, in the metric's own notation.  The same fact
#: the ``costs`` column carries in words, kept separately because a figure has room for
#: ``FP + FN`` where it has none for "precision and recall", and a reader coming from the
#: precision/recall tables is already holding the short form.
CONFUSION_CELLS_BY_CATEGORY = {
    SUBSTITUTIONS: "FP + FN",
    DELETIONS: "FN",
    INSERTIONS: "FP",
}

# ---------------------------------------------------------------------------
# Sub-categories of a substitution.
# ---------------------------------------------------------------------------

#: Some part of the assertion is right without being gold's value exactly.  Four ways in,
#: and they overlap heavily -- 62% of these rows satisfy more than one -- so this is one
#: label rather than four.  :func:`close_match_reasons` returns every reason that applies,
#: for a reader who wants to break it down; a break-down has to pick an order between them,
#: which is the thing this label avoids having to defend.
CLOSE_MATCH = "close_match"

#: The asserted value is a legacy value, carried by a *different* field of the record: the
#: instinct to copy was right and the source field was wrong.
MISLOCATE_LEGACY_VALUE = "mislocate_legacy_value"

#: Neither near gold nor carried by the record: nothing in the input accounts for it.
COMPLETELY_WRONG = "completely_wrong"

# ---------------------------------------------------------------------------
# Sub-categories of a deletion, and their mirrors under an insertion.
# ---------------------------------------------------------------------------

#: The run left the field blank and the record held gold's value: it was there to be copied
#: and was left behind.
UNDERESTIMATE_LEGACY_VALUE = "underestimate_legacy_value"

#: The run left the field blank and the record held nothing to go on -- but gold has a
#: value, so something was there to be known.  Read as a field that took more than
#: transcription, not as one that could not be done.
ENTIRELY_DONT_KNOW = "entirely_dont_know"

#: Gold left the field blank and the run wrote a value the record carries: it read more
#: into the record than the curator did.  The mirror of :data:`UNDERESTIMATE_LEGACY_VALUE`.
OVERESTIMATE_LEGACY_VALUE = "overestimate_legacy_value"

#: Gold left the field blank and the run answered from its own knowledge, the record
#: holding nothing.  The mirror of :data:`ENTIRELY_DONT_KNOW`.
TOO_OPTIMISTIC_ANSWER = "too_optimistic_answer"

#: Reporting order, grouped so each sub-category sits under its category.
SUBCATEGORIES = (
    CLOSE_MATCH,
    MISLOCATE_LEGACY_VALUE,
    COMPLETELY_WRONG,
    UNDERESTIMATE_LEGACY_VALUE,
    ENTIRELY_DONT_KNOW,
    OVERESTIMATE_LEGACY_VALUE,
    TOO_OPTIMISTIC_ANSWER,
)

CATEGORY_BY_SUBCATEGORY = {
    CLOSE_MATCH: SUBSTITUTIONS,
    MISLOCATE_LEGACY_VALUE: SUBSTITUTIONS,
    COMPLETELY_WRONG: SUBSTITUTIONS,
    UNDERESTIMATE_LEGACY_VALUE: DELETIONS,
    ENTIRELY_DONT_KNOW: DELETIONS,
    OVERESTIMATE_LEGACY_VALUE: INSERTIONS,
    TOO_OPTIMISTIC_ANSWER: INSERTIONS,
}

#: The reasons a substitution counts as a close match, in the order
#: :func:`close_match_reasons` returns them.
SAME_VALUE_OTHER_SHAPE = "same value, other shape"
ONE_CONTAINS_THE_OTHER = "one contains the other"
KEPT_THIS_FIELD_S_VALUE = "kept this field's record value"
USED_THE_VOCABULARY = "used the vocabulary, gold did not"

CLOSE_MATCH_REASONS = (
    SAME_VALUE_OTHER_SHAPE,
    ONE_CONTAINS_THE_OTHER,
    KEPT_THIS_FIELD_S_VALUE,
    USED_THE_VOCABULARY,
)

ERROR_COLUMNS = [
    "assay",
    "field",
    "field_type",
    "case",
    "costs",
    "category",
    "subcategory",
    "close_match_reasons",
    "gold_value",
    "predicted_value",
    "legacy_value",
    "resolution",
    "reasoning",
    "pointer",
]

_WHITESPACE = re.compile(r"\s+")

#: Stripped before comparing, so a bare DOI and the same DOI as a URL are one value.
#: The legacy records store them bare; the templates ask for the resolver URL.
_DOI_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/")

#: Answers that commit to nothing.  Kept small and literal: a longer list would start
#: swallowing real vocabulary terms.
_PLACEHOLDERS = frozenset(
    {"custom", "in-house", "in house", "unknown", "other", "none", "n/a", "na", "not applicable", "not specified"}
)


def _flatten(value: Any) -> str:
    """One comparable string for a value of any JSON type."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _loose(value: Any) -> str:
    """Case-folded, whitespace-collapsed, punctuation-trimmed form for near-miss tests."""
    return _WHITESPACE.sub(" ", _flatten(value).strip().casefold()).strip(" .,;:_-/")


def _loose_identifier(value: Any) -> str:
    """:func:`_loose`, with any DOI resolver prefix removed.

    Without this, a DOI copied out of the legacy record looks unsupported: the record
    holds ``10.17504/protocols.io.abc`` and the run writes
    ``https://dx.doi.org/10.17504/protocols.io.abc``, which is the same identifier
    wearing the prefix the template asks for.
    """
    return _DOI_PREFIX.sub("", _loose(value))


def _as_number(value: Any) -> float | None:
    """*value* as a float when it reads as one, else ``None``."""
    try:
        return float(_flatten(value).strip())
    except (TypeError, ValueError):
        return None


def _equivalent_to_gold(predicted_value: Any, gold_value: Any, predicted_loose: str) -> bool:
    """Whether the asserted value is gold's value in another shape.

    Two shapes to relax, not one: text, where case and whitespace differ, and number,
    where ``10`` and ``"10.0"`` are the same quantity written differently.  Both are the
    same finding -- the answer was right and the matcher could not see it -- so they are
    one category rather than two, and the numeric arm is kept even though this corpus
    happens to hold no instance of it: without it such a pair falls through to the
    containment test and is reported as a truncation, which it is not.
    """
    if predicted_loose == _loose(gold_value):
        return True
    predicted_number = _as_number(predicted_value)
    return predicted_number is not None and predicted_number == _as_number(gold_value)


def close_match_reasons(
    gold_value: Any,
    predicted_value: Any,
    field: str,
    legacy: dict[str, Any],
    permissible: set[str] | None,
) -> list[str]:
    """Every way this substitution is still partly right, not merely the first.

    Four tests, and a row can satisfy several: on this corpus 62% of close matches satisfy
    more than one, and "same value, other shape" is contained in "one contains the other"
    but for values that loosen away to nothing.  They are returned together rather than
    resolved into one, because resolving them means choosing an order between overlapping
    definitions, and a share reported against such a choice describes the order as much as
    the run.  :data:`CLOSE_MATCH` is the one label; this is for reading underneath it.
    """
    reasons: list[str] = []
    predicted_loose, gold_loose = _loose(predicted_value), _loose(gold_value)

    if _equivalent_to_gold(predicted_value, gold_value, predicted_loose):
        reasons.append(SAME_VALUE_OTHER_SHAPE)
    if predicted_loose and gold_loose and (predicted_loose in gold_loose or gold_loose in predicted_loose):
        reasons.append(ONE_CONTAINS_THE_OTHER)

    same_field = legacy.get(field)
    kept_it = not _is_missing(same_field) and _loose_identifier(same_field) == _loose_identifier(predicted_value)
    if kept_it:
        reasons.append(KEPT_THIS_FIELD_S_VALUE)

    # The curator kept a value the template does not permit and the run picked a term it
    # does: the two disagree about whether to normalise, which is not the run being wrong.
    if (
        permissible
        and not _is_missing(same_field)
        and _loose_identifier(same_field) == _loose_identifier(gold_value)
        and _loose_identifier(gold_value) not in permissible
        and _loose_identifier(predicted_value) in permissible
    ):
        reasons.append(USED_THE_VOCABULARY)

    return reasons


def _classify_error(
    case: str,
    gold_value: Any,
    predicted_value: Any,
    field: str,
    legacy: dict[str, Any],
    permissible: set[str] | None,
) -> tuple[str, list[str]]:
    """The sub-category for one error, with the close-match reasons behind it.

    An insertion and a deletion split on the same question, asked of the value each is
    about: did the record hold it?  A substitution asks that too -- of the value the run
    wrote -- but only after the close-match tests, which take out the rows where the run's
    answer is defensibly near gold's.  That order is safe rather than merely conventional:
    every row it takes from :data:`MISLOCATE_LEGACY_VALUE` is one whose asserted value is
    equal to gold's once shape is relaxed.
    """
    if case == INSERTION:
        return (OVERESTIMATE_LEGACY_VALUE if _appears_in_legacy(predicted_value, legacy) else TOO_OPTIMISTIC_ANSWER), []
    if case == DELETION:
        return (UNDERESTIMATE_LEGACY_VALUE if _appears_in_legacy(gold_value, legacy) else ENTIRELY_DONT_KNOW), []

    reasons = close_match_reasons(gold_value, predicted_value, field, legacy, permissible)
    if reasons:
        return CLOSE_MATCH, reasons
    if _legacy_fields_carrying(predicted_value, legacy) - {field}:
        return MISLOCATE_LEGACY_VALUE, []
    return COMPLETELY_WRONG, []


def _legacy_fields_carrying(value: Any, legacy: dict[str, Any]) -> set[str]:
    """Which legacy fields hold *value*, comparing loosely and ignoring a DOI prefix.

    The one definition of "the record contains this", so "the run copied the same field"
    and "the run copied a different field" cannot disagree about what counts as carried.
    A value with no comparable form -- one that loosens away to nothing, such as ``"."``
    -- is carried by nothing, since matching it against another empty form would report
    provenance where there is no value to have a provenance.
    """
    target = _loose_identifier(value)
    if not target:
        return set()
    return {
        legacy_field
        for legacy_field, legacy_value in legacy.items()
        if not _is_missing(legacy_value) and _loose_identifier(legacy_value) == target
    }


def _appears_in_legacy(value: Any, legacy: dict[str, Any]) -> bool:
    """Whether any legacy field holds *value*."""
    return bool(_legacy_fields_carrying(value, legacy))


def _decision_log(assay: Assay, model: str, run_type: str, record_name: str) -> dict[str, dict[str, Any]]:
    """The run's own account of each field, keyed by field, when it kept one.

    ARMS writes a decision per field under ``decisions/``; the prompt-only conditions do
    not, so this is empty for them and the two columns it fills stay blank.
    """
    path = assay.output_dir(model, run_type) / "decisions" / record_name
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {entry["key"]: entry for entry in entries if isinstance(entry, dict) and "key" in entry}


def _require_legacy(legacy_path: Path, predicted_path: Path) -> dict[str, Any]:
    """The legacy record at *legacy_path*, or a refusal to categorise without it.

    Every category here except the two shape tests is decided by asking what the legacy
    record holds, so a missing input is not a gap in one column -- it silently moves
    errors into :data:`SPURIOUS_INVENTED`, :data:`UNSUPPORTED_VALUE` and
    :data:`ABSTAINED_ABSENT`, the three categories that mean "the record could not
    account for this".  The reconciliation check would still pass, because the count is
    right and only the attribution is wrong, so nothing downstream would notice.

    A record with a prediction had an input when the run read it, which is what makes
    this a broken corpus rather than a partial one: an assay missing its gold or its
    schema is skipped, but a prediction without its input cannot be scored honestly.
    """
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"no legacy record at {legacy_path}, but {predicted_path} exists to categorise. "
            "Provenance is decided against the legacy record, so continuing would report "
            "these errors as invented, unsupported or unavailable when they may be none of those."
        )
    return load_record(legacy_path)


def collect_field_errors(
    data_root: str | Path,
    model: str,
    run_type: str,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> pd.DataFrame:
    """One row per counted error, labelled at both levels of the taxonomy.

    One row per (record, field), not per side of the score: the category *is* the confusion
    case, so a substitution has one label rather than one per side.  ``costs`` says which
    side it lands on, and :func:`reconcile_with_confusion` checks that ``FP`` and ``FN``
    still come back out.  ``pointer`` is ``<assay>/<record>#<field>``, and ``resolution``
    and ``reasoning`` carry the run's own account of the field where it kept one.

    Returns an empty frame with :data:`ERROR_COLUMNS` when the run has no predictions.
    Raises :class:`FileNotFoundError` when a record has a prediction but no legacy input,
    which :func:`_require_legacy` explains.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()
        permissible = {
            field: {_loose_identifier(option) for option in options}
            for field, options in assay.permissible_values().items()
        }
        output_dir = assay.output_dir(model, run_type)

        for gold_path in sorted(assay.gold_dir.glob("*.json")):
            predicted_path = output_dir / gold_path.name
            if not predicted_path.exists():
                continue
            gold = load_record(gold_path)
            predicted = load_record(predicted_path)
            legacy = _require_legacy(assay.input_dir / gold_path.name, predicted_path)
            decisions = _decision_log(assay, model, run_type, gold_path.name)

            for field in gold:
                case = _classify_field(predicted, gold, field, match_case=match_case, match_whole_word=match_whole_word)
                if case not in (INSERTION, DELETION, SUBSTITUTION):
                    continue

                subcategory, reasons = _classify_error(
                    case, gold.get(field), predicted.get(field), field, legacy, permissible.get(field)
                )
                decision = decisions.get(field, {})
                rows.append(
                    {
                        "assay": assay.label,
                        "field": field,
                        "field_type": "ontology" if field in ontology_fields else "non_ontology",
                        "case": case,
                        "costs": COSTS_BY_CASE[case],
                        "category": CATEGORY_BY_CASE[case],
                        "subcategory": subcategory,
                        "close_match_reasons": ", ".join(reasons),
                        "gold_value": gold.get(field),
                        "predicted_value": predicted.get(field),
                        "legacy_value": legacy.get(field),
                        "resolution": decision.get("resolution"),
                        "reasoning": decision.get("reasoning"),
                        "pointer": f"{assay.key}/{gold_path.stem}#{field}",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=ERROR_COLUMNS)
    frame = pd.DataFrame(rows)[ERROR_COLUMNS]
    _check_levels_agree(frame)
    return frame


def _check_levels_agree(errors: pd.DataFrame) -> None:
    """Refuse a frame whose two levels contradict each other.

    The category comes from the confusion case and the sub-category from the classifier, by
    two separate routes, so they could drift apart without anything raising -- and a
    sub-category filed under the wrong category would make one table's rows sum to
    something no other table agrees with.
    """
    wrong = {
        (subcategory, category)
        for subcategory, category in zip(errors["subcategory"], errors["category"], strict=True)
        if CATEGORY_BY_SUBCATEGORY.get(subcategory) != category
    }
    if wrong:
        raise ValueError(f"sub-category filed under the wrong category: {sorted(wrong)}")


def deduplicate_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """One row per *distinct* error, rather than one per record the error occurs in.

    The corpus repeats itself: the same field of the same assay is wrong the same way in
    dozens of records, so an instance-weighted count is carried by a handful of recurring
    values.  This answers the other question -- how many different things the run gets
    wrong -- the way :mod:`analysis.data_analysis.repetition` does for the headline
    numbers, and against the same notion of a distinct value.

    Two errors are the same when they agree on assay, field, what gold held, what the run
    asserted, and what the taxonomy calls it.  The sub-category is in the key because it is
    not implied by the rest: the same gold and asserted values can be labelled differently
    in two records when the *legacy* records differ -- a value absent from one record and
    present in another is a different mistake, and the taxonomy already says so.  It costs
    two rows on this corpus, and keeps every table below a partition of these rows.

    ``n_instances`` counts the records each distinct error covered, so nothing about the
    repetition is lost by collapsing it.
    """
    import pandas as pd

    if errors.empty:
        return errors.assign(n_instances=pd.Series(dtype=int))

    keys = ["assay", "field", "subcategory"]
    # Values of any JSON type reach this frame, and an unhashable one -- a list, say --
    # would make groupby raise rather than group.  Serialised, they compare as they read.
    serialised = errors.assign(
        _gold=errors["gold_value"].map(lambda value: json.dumps(value, sort_keys=True, default=str)),
        _predicted=errors["predicted_value"].map(lambda value: json.dumps(value, sort_keys=True, default=str)),
    )
    grouped = serialised.groupby([*keys, "_gold", "_predicted"], dropna=False, sort=False)
    distinct = grouped.head(1).copy()
    distinct["n_instances"] = (
        grouped.size().reindex(pd.MultiIndex.from_frame(distinct[[*keys, "_gold", "_predicted"]])).to_numpy()
    )
    return distinct.drop(columns=["_gold", "_predicted"]).reset_index(drop=True)


def _summarize_level(
    errors: pd.DataFrame,
    *,
    level: str,
    order: tuple[str, ...],
    field_type: str | None,
) -> pd.DataFrame:
    """Counts of *level* per assay, in *order*, over every counted error."""
    import pandas as pd

    selected = errors if field_type is None else errors[errors["field_type"] == field_type]
    if selected.empty:
        return pd.DataFrame(columns=[level, "total", "share"])

    table = (
        pd.crosstab(selected[level], selected["assay"])
        .reindex([name for name in order if name in set(selected[level])])
        .fillna(0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    table["share"] = (table["total"] / len(selected)).round(3)
    return table.reset_index()


def summarize_error_categories(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """Frequency of each **category** per assay, over every counted error.

    One row per category, one column per assay, plus a ``total`` and a ``share``.  The
    coarse level: what the figures draw and what a headline can carry.  *field_type*
    narrows to ``"ontology"`` or ``"non_ontology"``.
    """
    return _summarize_level(errors, level="category", order=CATEGORIES, field_type=field_type)


def summarize_error_subcategories(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """Frequency of each **sub-category** per assay, over every counted error.

    Same shape as :func:`summarize_error_categories` one level down, in an order that runs
    through the categories in theirs -- so printed underneath it, each block of rows sits
    below the category it rolls into.  A sub-category that dominates one assay and is absent
    elsewhere is visible here rather than averaged away.
    """
    return _summarize_level(errors, level="subcategory", order=SUBCATEGORIES, field_type=field_type)


def _level_shares(errors: pd.DataFrame, *, level: str, order: tuple[str, ...], field_type: str | None) -> pd.DataFrame:
    """Counts and within-assay shares of *level*, per assay and pooled over the corpus."""
    import pandas as pd

    selected = errors if field_type is None else errors[errors["field_type"] == field_type]
    if selected.empty:
        return pd.DataFrame(columns=["assay", level, "n", "n_errors", "share"])

    rows: list[dict[str, Any]] = []
    for assay_label, frame in [*selected.groupby("assay", observed=True), (POOLED_ASSAY, selected)]:
        counts = frame[level].value_counts()
        for name in order:
            rows.append(
                {
                    "assay": assay_label,
                    level: name,
                    "n": int(counts.get(name, 0)),
                    "n_errors": len(frame),
                    "share": int(counts.get(name, 0)) / len(frame),
                }
            )
    return pd.DataFrame(rows)


def category_shares(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """The categories per assay, as counts and as shares of that assay's errors.

    One row per (assay, category) plus one per category for the pooled corpus under
    :data:`POOLED_ASSAY`, so a figure can draw both from one frame.  ``share`` is within the
    assay, which is what makes composition comparable across assays that differ in size by
    more than tenfold; ``n_errors`` is carried alongside so the reader can see how much a
    share is standing on.  Every category is present even at zero, so a figure's segments
    keep their order between rows.
    """
    return _level_shares(errors, level="category", order=CATEGORIES, field_type=field_type)


def subcategory_shares(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """The same, one level down.

    The sub-categories run in their categories' order, so a bar stacked from this frame has
    each category as a single unbroken span -- which is what lets a figure brace the
    categories over the top of the sub-category segments.
    """
    return _level_shares(errors, level="subcategory", order=SUBCATEGORIES, field_type=field_type)


def reconcile_with_confusion(
    errors: pd.DataFrame,
    data_root: str | Path,
    model: str,
    run_type: str,
) -> dict[str, dict[str, int]]:
    """Check the labelled rows against the confusion counts they claim to explain.

    With one row per error rather than one per side, ``FP`` and ``FN`` are recovered from
    the confusion case: an insertion costs precision, a deletion recall, a substitution
    both.  Returns ``{"FP": {"counted": ..., "categorised": ...}, "FN": {...}}``.  A
    mismatch means an error was missed or counted twice, which would make every share in
    the tables above wrong -- so this is worth asserting rather than assuming.
    """
    from analysis.data_analysis.precision_recall_tables import _accumulate_confusion

    counts, _n_pairs, _skipped = _accumulate_confusion(iter_assays(data_root), model, run_type)
    return {
        "FP": {
            "counted": counts["all"]["FP"],
            "categorised": int(errors["case"].isin((INSERTION, SUBSTITUTION)).sum()),
        },
        "FN": {
            "counted": counts["all"]["FN"],
            "categorised": int(errors["case"].isin((DELETION, SUBSTITUTION)).sum()),
        },
    }
