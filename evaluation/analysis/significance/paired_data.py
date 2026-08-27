"""Collecting the paired outcomes every test and interval is built from.

The two runs compared default to ``baseline`` -- the prompt-only condition, under
``output/<model>/baseline/`` -- and ARMS, under ``output/<model>/agent-tool/``, but
:func:`collect_paired_data` will pair any two conditions asked of it.

One pass over the saved predictions produces every view the rest of the package needs,
because they have to describe the same comparison: a Wilcoxon test on per-record
accuracy and a bootstrap CI on the same accuracy would be incomparable if each walked
the corpus on its own terms.  A record is included only when *both* runs predicted it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from analysis.corpus import get_assay, iter_records, load_record
from analysis.metrics import CONFUSION_CATEGORIES, compute_field_confusion, compute_field_results

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

    * ``record_acc[category]`` is a list of ``(baseline_accuracy, system_accuracy)``
      tuples, one per record that has at least one field in that category.  Used
      for the per-record (record-weighted) view and the Wilcoxon test, matching
      how the paper's per-assay rows are computed.
    * ``record_counts[category]`` is a list of ``(baseline_correct, system_correct,
      total)`` integer tuples, one per record.  Used for the pooled
      (field-weighted) accuracy and its cluster bootstrap, matching how the
      paper's overall bottom row is computed.
    * ``field_outcomes[category]`` is a list of ``(baseline_correct, system_correct)``
      boolean tuples, one per field of that category across all records.  Used
      for the field-level McNemar test.
    * ``record_discordant[category]`` is a list of ``(baseline_only_correct,
      system_only_correct)`` integer tuples, one per record: the per-record counts
      of McNemar-discordant fields.  Used for the record-clustered permutation
      test, which keeps the record (not the field) as the independent unit.
    * ``record_confusion[category]`` is a list of ``(baseline_tp, baseline_fp,
      baseline_fn, system_tp, system_fp, system_fn)`` integer tuples, one per record.  Used for the
      cluster bootstrap on precision, recall and F1, and for the paired CI on
      the difference between the two approaches.
    """

    record_acc: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    record_counts: dict[str, list[tuple[int, int, int]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    field_outcomes: dict[str, list[tuple[bool, bool]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    record_discordant: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: {c: [] for c in CATEGORIES})
    record_confusion: dict[str, list[tuple[int, int, int, int, int, int]]] = field(
        default_factory=lambda: {c: [] for c in CATEGORIES}
    )

    def extend(self, other: PairedData) -> None:
        """Accumulate another assay's data into this one (used for the pooled view)."""
        for c in CATEGORIES:
            self.record_acc[c].extend(other.record_acc[c])
            self.record_counts[c].extend(other.record_counts[c])
            self.field_outcomes[c].extend(other.field_outcomes[c])
            self.record_discordant[c].extend(other.record_discordant[c])
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
    that category's accuracy down with records that could never contribute to it.
    """
    baseline_results = compute_field_results(baseline_pred, gold, schema_path)
    system_correct = {f: ok for f, _t, ok in compute_field_results(system_pred, gold, schema_path)}

    baseline_conf = compute_field_confusion(baseline_pred, gold, schema_path)
    system_conf = compute_field_confusion(system_pred, gold, schema_path)
    for cat in CATEGORIES:
        b, a = baseline_conf[cat], system_conf[cat]
        if b["TP"] + b["FP"] + b["FN"] + b["TN"]:  # category present in this record
            data.record_confusion[cat].append(
                (b["TP"], b["FP"], b["FN"], a["TP"], a["FP"], a["FN"]),
            )

    # Per-record correct/total counts by category, so a record with no fields
    # in a category is excluded from that category rather than scored 0.
    per_record: dict[str, list[list[int]]] = {c: [[0, 0], [0, 0]] for c in CATEGORIES}
    # Per-record McNemar-discordant counts: [baseline_only_correct, arms_only_correct].
    per_record_disc: dict[str, list[int]] = {c: [0, 0] for c in CATEGORIES}
    for fname, ftype, baseline_ok in baseline_results:
        system_ok = system_correct.get(fname, False)
        data.field_outcomes[ftype].append((baseline_ok, system_ok))
        data.field_outcomes["all"].append((baseline_ok, system_ok))
        for cat in (ftype, "all"):
            per_record[cat][0][0] += int(baseline_ok)
            per_record[cat][0][1] += 1
            per_record[cat][1][0] += int(system_ok)
            per_record[cat][1][1] += 1
            if baseline_ok and not system_ok:
                per_record_disc[cat][0] += 1
            elif system_ok and not baseline_ok:
                per_record_disc[cat][1] += 1

    for cat in CATEGORIES:
        (base_corr, base_tot), (sys_corr, sys_tot) = per_record[cat]
        if base_tot > 0:  # category present in this record
            data.record_acc[cat].append((base_corr / base_tot, sys_corr / sys_tot))
            data.record_counts[cat].append((base_corr, sys_corr, base_tot))
            base_only, sys_only = per_record_disc[cat]
            data.record_discordant[cat].append((base_only, sys_only))


def collect_paired_data(
    data_root: str | Path,
    model: str,
    assay_key: str,
    *,
    baseline_run: str = "baseline",
    system_run: str = "agent-tool",
) -> PairedData:
    """Collect paired outcomes for a single assay from saved outputs.

    *baseline_run* and *system_run* name the two output directories to compare, and
    every difference below is *system minus baseline*.  They
    default to the prompt-only baseline and ARMS, but any two conditions can be paired
    -- one repetition against another, say -- since nothing below this function knows
    which conditions produced the numbers.

    Records predicted by only one of the two runs are skipped: the comparison is
    paired, so an unmatched record would put them on different denominators.
    """
    assay = get_assay(data_root, assay_key)
    baseline_dir = assay.output_dir(model, baseline_run)
    system_dir = assay.output_dir(model, system_run)

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
