"""One run measured on its own, so every condition can carry an interval.

:mod:`~analysis.significance.paired_data` exists to compare two runs, and therefore
only ever describes the two it pairs.  A confidence interval needs no comparison: it
needs one run's per-record outcomes.  Those are what this module collects, in exactly
the shapes :mod:`~analysis.significance.bootstrap` takes, so a run the paired tables do
not model -- one repetition on its own, say -- can still be reported with an interval.

The point estimates are the same numbers :mod:`analysis.data_analysis` reports --
pooled (field-weighted) accuracy and micro precision/recall/F1 -- so this puts an
interval around an existing number rather than introducing a differently-weighted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from analysis.corpus import get_assay, iter_assays, iter_records, load_record
from analysis.metrics import compute_field_confusion, compute_field_results
from analysis.significance.paired_data import CATEGORIES

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SingleRunData:
    """One run's per-record outcomes, keyed by field category.

    * ``record_counts[category]`` is ``(correct, total)`` per record, for pooled
      accuracy and its bootstrap.
    * ``record_confusion[category]`` is ``(tp, fp, fn)`` per record, for micro
      precision, recall and F1 and their bootstrap.

    A record with no fields in a category is left out of that category rather than
    recorded as a zero, matching :class:`~analysis.significance.paired_data.PairedData`
    -- an assay whose template has no ontology-constrained fields must not drag that
    category down with records that could never have contributed to it.
    """

    record_counts: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    record_confusion: dict[str, list[tuple[int, int, int]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})

    def extend(self, other: SingleRunData) -> None:
        """Accumulate another assay's data into this one, for the pooled view."""
        for category in CATEGORIES:
            self.record_counts[category].extend(other.record_counts[category])
            self.record_confusion[category].extend(other.record_confusion[category])


def _add_record(
    data: SingleRunData,
    gold: dict[str, Any],
    predicted: dict[str, Any],
    schema_path: Path,
) -> None:
    """Fold one record's outcomes into *data*."""
    confusion = compute_field_confusion(predicted, gold, schema_path)
    for category in CATEGORIES:
        cell = confusion[category]
        if cell["TP"] + cell["FP"] + cell["FN"] + cell["TN"]:  # category present in this record
            data.record_confusion[category].append((cell["TP"], cell["FP"], cell["FN"]))

    per_record: dict[str, list[int]] = {category: [0, 0] for category in CATEGORIES}
    for _field_name, field_type, correct in compute_field_results(predicted, gold, schema_path):
        for category in (field_type, "all"):
            per_record[category][0] += int(correct)
            per_record[category][1] += 1

    for category in CATEGORIES:
        correct, total = per_record[category]
        if total:
            data.record_counts[category].append((correct, total))


def collect_single_run_data(
    data_root: str | Path,
    model: str,
    run_type: str,
    assay_key: str | None = None,
) -> SingleRunData:
    """Collect one run's per-record outcomes, for *assay_key* or pooled across all assays.

    Driven by the gold records, so a gold record the run never predicted is skipped
    rather than scored as a failure -- the same rule the paired collector follows.
    """
    assays = [get_assay(data_root, assay_key)] if assay_key else list(iter_assays(data_root))

    data = SingleRunData()
    for assay in assays:
        if not assay.has_gold:
            continue
        output_dir = assay.output_dir(model, run_type)
        for gold_file, gold in iter_records(assay.gold_dir):
            prediction = output_dir / gold_file.name
            if not prediction.exists():
                continue
            _add_record(data, gold, load_record(prediction), assay.schema_path)

    return data
