"""How deterministic a condition is across repeated runs: which answers change between runs.

Every other table here scores one run against gold.  This one does not look at gold's value
at all: it asks whether the model gives the same answer when it is asked the same question
again.  Repeat a run and some answers change, because the model is not deterministic; how
often they change is what this module counts.

The unit is one **field instance**: one field of one record.  Its answers in the runs are
compared with each other, and it falls in one of two bands:

* ``consistent`` -- every run gave the same answer;
* ``inconsistent`` -- at least one run gave a different answer.

"The same answer" is strict: blank and ``null`` count as the same blank, and any other value
must match exactly, so ``"Lung"`` and ``"lung"`` are two different answers.  A difference the
scoring would forgive is still the model not answering the same way twice.

A field instance counts only when there is something to answer: gold holds a value, or at
least one run filled one in -- the same field instances precision and recall count.  A field
that gold leaves blank and no run ever fills would be consistent by agreeing on nothing, and
counting it would fill the ``consistent`` band with empty fields.  A field gold wants and every
run leaves blank is counted, and is consistent: the model reliably gave no answer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, iter_records, load_record
from analysis.data_analysis.error_taxonomy import POOLED_ASSAY
from analysis.metrics import _is_missing

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

#: The bands a field instance falls into, in the order a stacked bar draws them.
STABILITY_BANDS = ("consistent", "inconsistent")

STABILITY_COLUMNS = ["assay", "record", "field", "field_type", "n_runs", "n_answers", "band"]


def _answer(value: Any) -> str | None:
    """*value* in a form two runs can be compared on: ``None`` for any blank, else canonical JSON."""
    return None if _is_missing(value) else json.dumps(value, sort_keys=True)


def collect_field_stability(
    data_root: str | Path,
    model: str,
    condition: str,
    *,
    runs: Sequence[int],
) -> pd.DataFrame:
    """One row per field instance of *condition*, with how many different answers *runs* gave.

    Columns are :data:`STABILITY_COLUMNS`.  ``assay`` is the assay's label, ``record`` the
    file name, ``n_answers`` the number of distinct answers across the runs (1 when they all
    agree), and ``band`` one of :data:`STABILITY_BANDS`.

    A record counts only when every run in *runs* predicted it: consistency is a comparison
    across runs, so a record missing from one of them has nothing to be compared on.  A
    field instance counts only when gold or at least one run holds a value (see the module
    docstring).  Gold decides which field instances count and nothing else.

    Raises:
        ValueError: If *runs* holds fewer than two runs, or the same run twice.
    """
    import pandas as pd

    if len(runs) < 2:
        raise ValueError(f"Consistency needs at least two runs to compare, got {list(runs)}.")
    if len(set(runs)) != len(runs):
        raise ValueError(f"Each run may be named once, got {list(runs)}.")

    rows: list[dict[str, object]] = []
    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()
        directories = [assay.output_dir(model, condition, run=run) for run in runs]
        for gold_file, gold in iter_records(assay.gold_dir):
            paths = [directory / gold_file.name for directory in directories]
            if not all(path.exists() for path in paths):
                continue
            predictions = [load_record(path) for path in paths]
            for field, gold_value in gold.items():
                answers = [_answer(prediction.get(field)) for prediction in predictions]
                if _is_missing(gold_value) and all(answer is None for answer in answers):
                    continue
                n_answers = len(set(answers))
                rows.append(
                    {
                        "assay": assay.label,
                        "record": gold_file.name,
                        "field": field,
                        "field_type": "ontology" if field in ontology_fields else "non_ontology",
                        "n_runs": len(runs),
                        "n_answers": n_answers,
                        "band": "consistent" if n_answers == 1 else "inconsistent",
                    }
                )
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def summarize_field_stability(stability: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """The share of field instances in each band, per assay and pooled over the corpus.

    One row per assay in the order the assays first appear in *stability*, then a
    ``POOLED_ASSAY`` row over every field instance -- pooled over instances, not averaged
    over the assays, since the assays differ in size by more than tenfold.  Columns are
    ``assay``, ``n_instances`` and one share per band.  *field_type* restricts the table to
    ``"ontology"`` or ``"non_ontology"`` fields.
    """
    import pandas as pd

    if field_type is not None:
        stability = stability[stability["field_type"] == field_type]

    def shares(frame: pd.DataFrame, assay: str) -> dict[str, object]:
        counts = frame["band"].value_counts()
        total = len(frame)
        return {
            "assay": assay,
            "n_instances": total,
            **{band: (counts.get(band, 0) / total if total else 0.0) for band in STABILITY_BANDS},
        }

    rows = [shares(frame, assay) for assay, frame in stability.groupby("assay", sort=False)]
    rows.append(shares(stability, POOLED_ASSAY))
    return pd.DataFrame(rows, columns=["assay", "n_instances", *STABILITY_BANDS])


def rank_inconsistent_fields(stability: pd.DataFrame, *, top: int | None = 10) -> pd.DataFrame:
    """The fields whose answers change most often, per assay: where the non-determinism is.

    One row per ``(assay, field)`` that was inconsistent at least once, with ``n_records``
    (field instances counted), ``n_inconsistent`` and ``inconsistent_share``, sorted by
    ``n_inconsistent`` so the fields carrying most of the changes come first.  *top* keeps
    that many rows; ``None`` keeps all.
    """
    grouped = stability.groupby(["assay", "field", "field_type"], sort=False)
    table = grouped.agg(
        n_records=("band", "size"),
        n_inconsistent=("band", lambda bands: int((bands == "inconsistent").sum())),
    ).reset_index()
    table = table[table["n_inconsistent"] > 0]
    table["inconsistent_share"] = table["n_inconsistent"] / table["n_records"]
    table = table.sort_values(["n_inconsistent", "inconsistent_share"], ascending=False, kind="stable")
    table = table.reset_index(drop=True)
    return table if top is None else table.head(top)
