"""Collecting the paired outcomes every test and interval is built from.

The two conditions compared default to ``baseline`` -- the prompt-only condition, under
``output/<model>/baseline/`` -- and ARMS, under ``output/<model>/arms-agent/``, but
:func:`collect_paired_data` will pair any two conditions asked of it.

One pass over the saved predictions produces every view the rest of the package needs,
because they have to describe the same comparison: a permutation test on precision and
a bootstrap interval on the same precision would be incomparable if each walked the
corpus on its own terms.  A record is included only when *both* conditions predicted it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from analysis.corpus import get_assay, iter_records, load_record
from analysis.metrics import CONFUSION_CATEGORIES, compute_field_confusion

if TYPE_CHECKING:
    from pathlib import Path

#: The field groupings the paper reports.  Aliased to the categories the confusion
#: counts are already keyed by rather than restated, so the two cannot drift apart.
CATEGORIES = CONFUSION_CATEGORIES

CATEGORY_LABELS = {
    "ontology": "Ontology-constrained",
    "non_ontology": "Non-ontology-constrained",
    "all": "All fields",
}


@dataclass
class PairedData:
    """Paired baseline/system outcomes for one assay (or pooled across assays).

    ``record_confusion[category]`` is a list of ``(baseline_tp, baseline_fp, baseline_fn,
    system_tp, system_fp, system_fn)`` integer tuples, one per record that has at least
    one field in that category.  Used for the cluster bootstrap on precision, recall and
    F1, for the paired interval on the difference, and for the permutation test.
    """

    record_confusion: dict[str, list[tuple[int, int, int, int, int, int]]] = field(
        default_factory=lambda: {c: [] for c in CATEGORIES}
    )

    def extend(self, other: PairedData) -> None:
        """Accumulate another assay's data into this one (used for the pooled view)."""
        for c in CATEGORIES:
            self.record_confusion[c].extend(other.record_confusion[c])


def _add_record(
    data: PairedData,
    gold: dict[str, Any],
    baseline_pred: dict[str, Any],
    system_pred: dict[str, Any],
    schema_path: Path,
) -> None:
    """Fold one record's baseline-vs-system comparison into *data*.

    A category the record has no fields in is left untouched rather than recorded as
    zero, so an assay whose template has no ontology-constrained fields does not drag
    that category down with records that could never contribute to it.
    """
    baseline_conf = compute_field_confusion(baseline_pred, gold, schema_path)
    system_conf = compute_field_confusion(system_pred, gold, schema_path)
    for cat in CATEGORIES:
        b, a = baseline_conf[cat], system_conf[cat]
        if b["TP"] + b["FP"] + b["FN"] + b["TN"]:  # category present in this record
            data.record_confusion[cat].append(
                (b["TP"], b["FP"], b["FN"], a["TP"], a["FP"], a["FN"]),
            )


def collect_paired_data(
    data_root: str | Path,
    model: str,
    assay_key: str,
    *,
    baseline: str = "baseline",
    system: str = "arms-agent",
    run: int = 1,
) -> PairedData:
    """Collect paired outcomes for a single assay from saved outputs.

    *baseline* and *system* name the two output directories to compare, and
    every difference below is *system minus baseline*.  They
    default to the prompt-only baseline and ARMS, but any two conditions can be paired
    -- one repetition against another, say -- since nothing below this function knows
    which conditions produced the numbers.

    Records predicted by only one of the two conditions are skipped: the comparison is
    paired, so an unmatched record would put them on different denominators.
    """
    assay = get_assay(data_root, assay_key)
    baseline_dir = assay.output_dir(model, baseline, run=run)
    system_dir = assay.output_dir(model, system, run=run)

    data = PairedData()
    if not assay.has_gold:
        return data

    for gold_file, gold in iter_records(assay.gold_dir):
        baseline_file = baseline_dir / gold_file.name
        system_file = system_dir / gold_file.name
        if not (baseline_file.exists() and system_file.exists()):
            continue
        _add_record(data, gold, load_record(baseline_file), load_record(system_file), assay.schema_path)

    return data
