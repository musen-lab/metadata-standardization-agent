"""Whether gold's value was already written in the input, and whether that predicted success.

:mod:`analysis.data_analysis.error_taxonomy` asks where the value the run *asserted* came
from.  This module asks the mirror question of the value gold *wanted*: is it carried
verbatim by some field of the legacy record, or would producing it have taken a
normalisation, an inference, or knowledge from outside the record?

The distinction is descriptive, and deliberately not called recoverability.  A gold value
absent from the record is not therefore unreachable -- on this corpus the runs produce
most of them anyway -- so the split bounds nothing.  What it does is separate the fields
where migration is transcription from the fields where it is interpretation, and crossed
with the ontology/non-ontology split it says which of the four kinds of work a run is
actually bad at.  That is a claim about difficulty, not about a ceiling.

The comparison is the same one the taxonomy uses -- :func:`_appears_in_legacy`, whole
values compared loosely with a DOI prefix ignored -- so "the record carries this" means
one thing across both modules.  Correctness is
:func:`~analysis.metrics.matching._classify_field`, the rule every other table scores on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, load_record
from analysis.data_analysis.error_taxonomy import _appears_in_legacy, _require_legacy
from analysis.metrics import _is_missing
from analysis.metrics.matching import MATCH, _classify_field

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

#: Whether the legacy record carries gold's value, as it is written in the tables.
VERBATIM = "in the record"
DERIVED = "not in the record"

AVAILABILITY_COLUMNS = ["assay", "field_type", "availability", "n_gold_values", "n_correct", "correct_rate"]


def create_availability_summary(
    data_root: str | Path,
    model: str,
    run_type: str,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
) -> pd.DataFrame:
    """How often *run_type* gets a field right, by field type and by where gold's value was.

    One row per (assay, field type, availability), holding the number of gold values of
    that kind, how many the run reproduced, and the rate.  Only fields gold actually asks
    for are counted -- a blank gold field asks for nothing -- so ``n_gold_values`` sums to
    recall's denominator, and the four ``availability`` x ``field_type`` groups partition
    it.

    The counts are returned alongside the rate so a group can be pooled across assays by
    summing them, which averaging the per-assay rates would get wrong: the assays differ
    in size by more than tenfold.
    """
    import pandas as pd

    tally: dict[tuple[str, str, str], dict[str, int]] = {}
    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()
        output_dir = assay.output_dir(model, run_type)

        for gold_path in sorted(assay.gold_dir.glob("*.json")):
            predicted_path = output_dir / gold_path.name
            if not predicted_path.exists():
                continue
            gold = load_record(gold_path)
            predicted = load_record(predicted_path)
            legacy = _require_legacy(assay.input_dir / gold_path.name, predicted_path)

            for field, gold_value in gold.items():
                if _is_missing(gold_value):
                    continue
                key = (
                    assay.label,
                    "ontology" if field in ontology_fields else "non_ontology",
                    VERBATIM if _appears_in_legacy(gold_value, legacy) else DERIVED,
                )
                case = _classify_field(predicted, gold, field, match_case=match_case, match_whole_word=match_whole_word)
                cell = tally.setdefault(key, {"n_gold_values": 0, "n_correct": 0})
                cell["n_gold_values"] += 1
                cell["n_correct"] += case == MATCH

    if not tally:
        return pd.DataFrame(columns=AVAILABILITY_COLUMNS)

    rows: list[dict[str, Any]] = [
        {
            "assay": assay_label,
            "field_type": field_type,
            "availability": availability,
            **counts,
            "correct_rate": counts["n_correct"] / counts["n_gold_values"],
        }
        for (assay_label, field_type, availability), counts in tally.items()
    ]
    return pd.DataFrame(rows)[AVAILABILITY_COLUMNS]


def pool_availability(summary: pd.DataFrame, field_type: str, availability: str) -> dict[str, float]:
    """One group's rate over the whole corpus, pooled over values rather than assays.

    Returns ``{"n_gold_values", "n_correct", "correct_rate"}``, with a rate of ``0.0`` for
    an empty group so a caller drawing a reference line has a number rather than a NaN.
    """
    group = summary[(summary["field_type"] == field_type) & (summary["availability"] == availability)]
    n_values = int(group["n_gold_values"].sum())
    n_correct = int(group["n_correct"].sum())
    return {
        "n_gold_values": n_values,
        "n_correct": n_correct,
        "correct_rate": n_correct / n_values if n_values else 0.0,
    }
