"""Do the headline numbers survive the corpus's repeated values?

The corpus is repetitive: the same ``(field, value)`` correction recurs across many
records, so an instance-weighted number can be carried by a handful of common values.
The three tables here take that apart.  :func:`create_deduplicated_accuracy_summary`
counts each distinct correction once; :func:`create_deduplicated_precision_recall_summary`
does the same for precision and recall; :func:`create_frequency_split_accuracy_summary`
keeps the instance weighting but separates values seen once from values seen again.

The accuracy tables key a correction by ``(assay, field, gold-value)``.  The
precision/recall table keys each ratio by its own denominator, which is what
:func:`create_deduplicated_precision_recall_summary` documents.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from analysis.corpus import ValueKey, iter_assays, iter_pairs, iter_records, matches_field_type, value_key
from analysis.metrics import CONFUSION_CATEGORIES, _is_field_correct, _is_missing

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import pandas as pd

    from analysis.corpus import Assay

logger = logging.getLogger(__name__)


def _macro_over_clusters(clusters: dict[ValueKey, list[int]]) -> tuple[float, int]:
    """Summed per-cluster match fraction, and how many clusters there were.

    A cluster correct in 7 of its 10 instances contributes 0.7, the same partial
    credit :func:`create_deduplicated_accuracy_summary` gives it, so the two tables
    cannot disagree about what counting a value once means.
    """
    return sum(sum(outcomes) / len(outcomes) for outcomes in clusters.values()), len(clusters)


def create_deduplicated_accuracy_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    populated_only: bool = False,
) -> pd.DataFrame:
    """Accuracy with each unique correction counted once, controlling for repetition.

    Groups field instances by their unique ``(assay, field, gold-value)`` key, averages
    correctness within each group, then macro-averages over groups (split by ontology
    vs non-ontology fields).  It complements the instance-weighted summaries by showing
    performance across *distinct* corrections rather than repeated ones.

    Returns a single-row DataFrame with the three accuracy columns plus the number
    of unique pairs contributing to each (``n_ontology_pairs``,
    ``n_non_ontology_pairs``, ``n_unique_pairs``).  When *populated_only* is
    ``True``, only gold fields that carry a value are counted.
    """
    import pandas as pd

    groups: dict[str, dict[ValueKey, list[int]]] = {
        "ontology": defaultdict(list),
        "non_ontology": defaultdict(list),
    }

    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()

        for _gold_file, gold, predicted in iter_pairs(assay.gold_dir, assay.output_dir(model, run_type)):
            if predicted is None:
                continue
            for field, gold_val in gold.items():
                if populated_only and _is_missing(gold_val):
                    continue
                correct = _is_field_correct(predicted, gold, field, match_case=True, match_whole_word=True)
                category = "ontology" if field in ontology_fields else "non_ontology"
                groups[category][value_key(assay.key, field, gold_val)].append(int(correct))

    def _macro(pairs: dict[ValueKey, list[int]]) -> tuple[float, int]:
        total, n_pairs = _macro_over_clusters(pairs)
        return (total / n_pairs if n_pairs else 0.0), n_pairs

    ont_acc, ont_n = _macro(groups["ontology"])
    non_acc, non_n = _macro(groups["non_ontology"])
    all_acc, all_n = _macro({**groups["ontology"], **groups["non_ontology"]})

    return pd.DataFrame(
        [
            {
                "ontology_constrained_accuracy": ont_acc,
                "non_ontology_constrained_accuracy": non_acc,
                "all_field_accuracy": all_acc,
                "n_ontology_pairs": ont_n,
                "n_non_ontology_pairs": non_n,
                "n_unique_pairs": all_n,
            }
        ]
    )


def _accumulate_value_clusters(
    assays: Iterable[Assay],
    model: str,
    run_type: str,
) -> tuple[dict[str, dict[ValueKey, list[int]]], dict[str, dict[ValueKey, list[int]]], int]:
    """Cluster every field instance of *assays* by the value it turns on.

    Recall clusters key on the gold value, precision clusters on the predicted value,
    following the denominator each ratio actually has.  Both keys carry the assay, so a
    string that means one thing in one assay and something else in another is never
    merged.  Returns ``(recall_clusters, precision_clusters, n_skipped)``.
    """
    recall_clusters: dict[str, dict[ValueKey, list[int]]] = {
        "ontology": defaultdict(list),
        "non_ontology": defaultdict(list),
    }
    precision_clusters: dict[str, dict[ValueKey, list[int]]] = {
        "ontology": defaultdict(list),
        "non_ontology": defaultdict(list),
    }
    n_skipped = 0

    for assay in assays:
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()

        for _gold_file, gold, predicted in iter_pairs(assay.gold_dir, assay.output_dir(model, run_type)):
            if predicted is None:
                n_skipped += 1
                continue
            for field, gold_val in gold.items():
                category = "ontology" if field in ontology_fields else "non_ontology"
                correct = int(_is_field_correct(predicted, gold, field, match_case=True, match_whole_word=True))
                if not _is_missing(gold_val):
                    recall_clusters[category][value_key(assay.key, field, gold_val)].append(correct)
                predicted_val = predicted.get(field)
                if not _is_missing(predicted_val):
                    precision_clusters[category][value_key(assay.key, field, predicted_val)].append(correct)

    return recall_clusters, precision_clusters, n_skipped


def _deduplicated_rows(
    recall_clusters: dict[str, dict[ValueKey, list[int]]],
    precision_clusters: dict[str, dict[ValueKey, list[int]]],
    decimal_places: int,
) -> list[dict[str, Any]]:
    """One row per field category from a set of clusters."""
    rows: list[dict[str, Any]] = []
    for category in CONFUSION_CATEGORIES:
        if category == "all":
            # A field belongs to exactly one category, so the two never share a key.
            gold_group = {**recall_clusters["ontology"], **recall_clusters["non_ontology"]}
            asserted_group = {**precision_clusters["ontology"], **precision_clusters["non_ontology"]}
        else:
            gold_group = recall_clusters[category]
            asserted_group = precision_clusters[category]

        reproduced, n_gold_values = _macro_over_clusters(gold_group)
        correct_asserted, n_asserted_values = _macro_over_clusters(asserted_group)
        recall = reproduced / n_gold_values if n_gold_values else 0.0
        precision = correct_asserted / n_asserted_values if n_asserted_values else 0.0
        # Computed here rather than through precision_recall_f1, which takes one
        # integer TP: these two ratios count over different cluster sets.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append(
            {
                "category": category,
                "n_gold_values": n_gold_values,
                "gold_values_reproduced": round(reproduced, decimal_places),
                "n_asserted_values": n_asserted_values,
                "asserted_values_correct": round(correct_asserted, decimal_places),
                "precision": round(precision, decimal_places),
                "recall": round(recall, decimal_places),
                "f1": round(f1, decimal_places),
            }
        )
    return rows


def create_deduplicated_precision_recall_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 3,
) -> pd.DataFrame:
    """Precision and recall with each distinct value counted once.

    The instance-weighted tables in :mod:`~analysis.data_analysis.precision_recall_tables`
    answer "of all the field *instances*, how many were right".  This one answers "of
    all the *distinct values*, how many were right".  Where one ``(field, value)`` pair
    recurs across ten records, one systematic error costs ten there and one here, so a
    gap between the two tables is the sign that the headline is being carried -- in
    either direction -- by how often values repeat rather than by how many the run knows.

    Clustering follows the denominator each ratio actually has:

    * **Recall** counts the values gold asserts, so its clusters are keyed by
      ``(assay, field, gold-value)``.
    * **Precision** counts the values the run asserts, so its clusters are keyed by
      ``(assay, field, predicted-value)``.  Three different wrong answers for one field
      are three distinct wrong assertions; the same wrong answer three times is one.

    Blank gold contributes no recall cluster and a blank prediction no precision
    cluster, matching the instance-weighted tables, where agreeing that a field is
    empty earns no credit.  Only fields present in gold are visited, also as there.

    Returns one row per entry in :data:`analysis.metrics.CONFUSION_CATEGORIES`.
    ``gold_values_reproduced`` and ``asserted_values_correct`` are summed per-cluster
    fractions -- the deduplicated analogue of ``TP`` -- over ``n_gold_values`` and
    ``n_asserted_values`` respectively.  When every value in the corpus is unique,
    every cluster holds one instance and these ratios equal the instance-weighted ones.
    """
    import pandas as pd

    recall_clusters, precision_clusters, n_skipped = _accumulate_value_clusters(iter_assays(data_root), model, run_type)
    if n_skipped:
        logger.warning(
            "Deduplicated precision/recall skipped %d gold record(s) with no %s/%s prediction",
            n_skipped,
            model,
            run_type,
        )
    return pd.DataFrame(_deduplicated_rows(recall_clusters, precision_clusters, decimal_places))


def create_per_assay_deduplicated_precision_recall_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    decimal_places: int = 3,
) -> pd.DataFrame:
    """:func:`create_deduplicated_precision_recall_summary`, broken out per assay.

    Clusters never span assays -- their key carries the assay -- so scoring each assay
    on its own changes nothing about how a value is counted; it only reports the rows
    separately.  Returns one row per (assay, field category) with an ``assay`` column
    holding the label, following ``ASSAY_ORDER``, and skips an assay with no gold or no
    predictions rather than reporting it as zero.
    """
    import pandas as pd

    frames = []
    for assay in iter_assays(data_root):
        if not assay.has_gold or not any(assay.output_dir(model, run_type).glob("*.json")):
            continue
        recall_clusters, precision_clusters, _skipped = _accumulate_value_clusters([assay], model, run_type)
        rows = _deduplicated_rows(recall_clusters, precision_clusters, decimal_places)
        frames.append(pd.DataFrame(rows).assign(assay=assay.label))

    if not frames:
        logger.warning("No assay had both gold records and %s/%s predictions", model, run_type)
        return pd.DataFrame(columns=["assay", "category"])
    return pd.concat(frames, ignore_index=True)


def _count_value_frequencies(data_root: str, field_type: str | None) -> dict[ValueKey, int]:
    """How often each ``(assay, field, gold-value)`` occurs in the gold standard."""
    frequency: dict[ValueKey, int] = defaultdict(int)

    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()
        for _gold_file, gold in iter_records(assay.gold_dir):
            for field, gold_val in gold.items():
                if matches_field_type(field, ontology_fields, field_type):
                    frequency[value_key(assay.key, field, gold_val)] += 1

    return frequency


def create_frequency_split_accuracy_summary(
    data_root: str,
    model: str,
    run_type: str,
    *,
    field_type: str | None = None,
) -> pd.DataFrame:
    """Instance-weighted accuracy split by how often each value recurs.

    Each field instance is bucketed by the frequency of its ``(assay, field,
    gold-value)`` key across the corpus: ``singleton`` (appears once) versus
    ``recurring`` (appears in two or more records).  Within each bucket the function
    reports the instance-weighted accuracy, separating the gains on common, repeated
    values from those on rare, one-off values.  ``field_type`` filters to
    ``"ontology"`` or ``"non_ontology"`` (default: all fields).  Returns a two-row
    DataFrame with columns ``bucket``, ``accuracy``, and ``n_instances``.
    """
    import pandas as pd

    frequency = _count_value_frequencies(data_root, field_type)

    buckets = {"singleton": [0, 0], "recurring": [0, 0]}  # bucket -> [correct, total]
    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()

        for _gold_file, gold, predicted in iter_pairs(assay.gold_dir, assay.output_dir(model, run_type)):
            if predicted is None:
                continue
            for field, gold_val in gold.items():
                if not matches_field_type(field, ontology_fields, field_type):
                    continue
                correct = _is_field_correct(predicted, gold, field, match_case=True, match_whole_word=True)
                bucket = "singleton" if frequency[value_key(assay.key, field, gold_val)] == 1 else "recurring"
                buckets[bucket][1] += 1
                buckets[bucket][0] += int(correct)

    return pd.DataFrame(
        [
            {
                "bucket": name,
                "accuracy": correct / total if total else 0.0,
                "n_instances": total,
            }
            for name, (correct, total) in buckets.items()
        ]
    )
