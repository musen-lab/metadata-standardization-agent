"""The do-nothing reference point: legacy input scored against the gold standard.

Nothing here involves a model.  Each legacy record in ``data/<assay>/input`` is
compared directly to its gold counterpart, characterizing how far the legacy data
starts from the gold standard -- the number every corrected result has to be read
against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.corpus import iter_assays, iter_pairs
from analysis.data_analysis.accuracy_tables import _accuracy_row, _new_tally
from analysis.metrics import _is_field_correct, _is_missing

if TYPE_CHECKING:
    import pandas as pd


def create_uncorrected_accuracy_summary(
    data_root: str,
    *,
    populated_only: bool = False,
) -> pd.DataFrame:
    """Accuracy of the raw legacy input records against the gold standard.

    When *populated_only* is ``True``, only gold fields that carry a value are
    counted, excluding both-empty agreements (the harder, more informative subset).
    Returns a single-row DataFrame with the same three accuracy columns as
    :func:`~analysis.data_analysis.accuracy_tables.create_overall_accuracy_summary`;
    values are not rounded, so the caller controls display precision.
    """
    import pandas as pd

    tally = _new_tally()

    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()

        for _gold_file, gold, legacy in iter_pairs(assay.gold_dir, assay.input_dir):
            if legacy is None:
                continue
            for field, gold_val in gold.items():
                if populated_only and _is_missing(gold_val):
                    continue
                prefix = "ontology" if field in ontology_fields else "non_ontology"
                tally[f"{prefix}_total"] += 1
                tally[f"{prefix}_correct"] += int(
                    _is_field_correct(legacy, gold, field, match_case=True, match_whole_word=True)
                )

    return pd.DataFrame([_accuracy_row(tally)])
