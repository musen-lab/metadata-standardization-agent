"""Paired tests that count each distinct value once, not each field instance.

The tests in :mod:`analysis.significance.hypothesis_tests` resample whole records, which
corrects for fields being correlated inside a record but not for the same
``(field, value)`` recurring across records -- the effect
:func:`~analysis.significance.clustering.effective_sample_size` measures.  These two
tests remove it by construction, taking the distinct value, or the field, as the unit:

* **Recall** pairs on one distinct ``(assay, field, gold value)``.  Gold fixes that list,
  so both runs are scored on identical items and each item yields a matched pair.
* **Precision** cannot pair on the value, because it is scored over whatever each run
  *asserted* and the two runs assert different values -- often with no overlap at all.
  Pairing one level up, on ``(assay, field)``, restores it: each field's distinct
  assertions are averaged within the field, then the field-level scores are paired.
  That makes it a macro-average over fields rather than over distinct assertions, which
  is the price of a paired comparison; the estimand is named in the returned rows.

A difference that survives here is not an artifact of repetition.  One that shows up in
the record-level tests but not here is a real saving of work, carried by values the
corpus repeats, rather than evidence that a run knows more distinct answers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from analysis.corpus import iter_assays, iter_records, load_record, value_key
from analysis.metrics import compute_field_results
from analysis.significance.paired_data import CATEGORIES, CATEGORY_LABELS

if TYPE_CHECKING:
    from pathlib import Path

#: Resamples for both the permutation test and the bootstrap interval, matching
#: :func:`~analysis.significance.hypothesis_tests.paired_permutation_prf`.
N_RESAMPLES = 10_000

#: Fixed so a table is reproducible: the same corpus gives the same p-value twice.
SEED = 0

FieldKey = tuple[str, str]


class DeduplicatedOutcomes:
    """Both runs' per-item outcomes, collected in one pass over the records they share.

    * :attr:`recall_clusters` maps one distinct ``(assay, field, gold value)`` to each
      run's outcomes over the records that value appears in -- the paired unit for
      recall.
    * :attr:`asserted_by_field` maps ``(assay, field)`` to each run's own distinct
      asserted values and their outcomes -- the paired unit for precision is the field,
      since the two runs do not assert the same values.
    * :attr:`field_types` classifies each ``(assay, field)`` as ontology or not, per that
      assay's own schema.
    """

    def __init__(self) -> None:
        self.recall_clusters: dict[tuple[str, str, str], dict[str, list[int]]] = {}
        self.asserted_by_field: dict[FieldKey, dict[str, dict[str, list[int]]]] = {}
        self.field_types: dict[FieldKey, str] = {}

    def in_category(self, field_key: FieldKey, category: str) -> bool:
        """Whether *field_key* belongs to *category*, where ``"all"`` takes everything."""
        return category == "all" or self.field_types[field_key] == category


def collect_deduplicated_outcomes(
    data_root: str | Path,
    model: str,
    *,
    baseline_run: str,
    system_run: str,
) -> DeduplicatedOutcomes:
    """Collect both runs' outcomes, keyed the two ways the paired tests need.

    Only records both runs produced are visited: the comparison is paired, so an
    unmatched record would put the two runs on different denominators.
    """
    outcomes = DeduplicatedOutcomes()
    runs = (baseline_run, system_run)

    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()
        directories = {run: assay.output_dir(model, run) for run in runs}

        for gold_file, gold in iter_records(assay.gold_dir):
            paths = {run: directory / gold_file.name for run, directory in directories.items()}
            if not all(path.exists() for path in paths.values()):
                continue

            records = {run: load_record(path) for run, path in paths.items()}
            correct = {
                run: {field: ok for field, _type, ok in compute_field_results(record, gold, assay.schema_path)}
                for run, record in records.items()
            }

            for field, gold_value in gold.items():
                if field not in correct[baseline_run]:
                    continue
                field_key = (assay.key, field)
                outcomes.field_types[field_key] = "ontology" if field in ontology_fields else "non_ontology"
                per_run = {run: int(correct[run].get(field, False)) for run in runs}

                if gold_value not in (None, ""):  # blank gold asks for nothing to recall
                    cluster = outcomes.recall_clusters.setdefault(value_key(assay.key, field, gold_value), {})
                    for run, outcome in per_run.items():
                        cluster.setdefault(run, []).append(outcome)

                for run in runs:  # precision counts what the run itself asserted
                    asserted = records[run].get(field)
                    if asserted in (None, ""):
                        continue
                    values = outcomes.asserted_by_field.setdefault(field_key, {}).setdefault(run, {})
                    values.setdefault(json.dumps(asserted, sort_keys=True), []).append(per_run[run])

    return outcomes


def paired_cluster_test(
    baseline_scores: np.ndarray,
    system_scores: np.ndarray,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    """Permutation p-value and bootstrap CI for system-minus-baseline over paired items.

    Under the null the two runs are interchangeable, so a replicate swaps the two scores
    of a random subset of items.  The interval resamples whole items.  Both treat the
    item -- a value cluster or a field -- as the unit, so a value repeated across records
    is never counted as independent evidence.

    Returns the two means, the observed difference, its 95% interval and the two-sided
    p-value.
    """
    observed = float(system_scores.mean() - baseline_scores.mean())
    rng = np.random.default_rng(seed)

    swap = rng.random((n_resamples, len(baseline_scores))) < 0.5
    shuffled = np.where(swap, baseline_scores, system_scores).mean(axis=1) - np.where(
        swap, system_scores, baseline_scores
    ).mean(axis=1)
    # Float tolerance, as in hypothesis_tests: an exact tie can miss by an ulp.
    extreme = int((np.abs(shuffled) >= abs(observed) - 1e-12).sum())

    draws = rng.integers(0, len(baseline_scores), (n_resamples, len(baseline_scores)))
    replicates = system_scores[draws].mean(axis=1) - baseline_scores[draws].mean(axis=1)
    lo, hi = np.percentile(replicates, [2.5, 97.5])

    return {
        "baseline": float(baseline_scores.mean()),
        "system": float(system_scores.mean()),
        "delta": observed,
        "lo": float(lo),
        "hi": float(hi),
        "pvalue": (extreme + 1) / (n_resamples + 1),
    }


def _field_precision(values: dict[str, list[int]]) -> float:
    """One field's precision over its own distinct assertions, each counted once."""
    return sum(sum(outcomes) / len(outcomes) for outcomes in values.values()) / len(values)


def deduplicated_paired_tests(
    data_root: str | Path,
    model: str,
    *,
    baseline_run: str,
    system_run: str,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Run both paired tests for every field category, and return one row per test.

    Each row carries the field category, the metric, what the pairing unit was, how many
    items stood behind it, both runs' means, and the difference with its interval and
    p-value.  Formatting and thresholding are left to the caller: this returns numbers.
    """
    outcomes = collect_deduplicated_outcomes(data_root, model, baseline_run=baseline_run, system_run=system_run)
    both_asserted = sorted(key for key, runs in outcomes.asserted_by_field.items() if len(runs) == 2)
    runs = (baseline_run, system_run)

    rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        # Recall: the item is one distinct (assay, field, gold value).
        keys = [key for key in outcomes.recall_clusters if outcomes.in_category((key[0], key[1]), category)]
        if keys:
            scores = {run: np.array([np.mean(outcomes.recall_clusters[key][run]) for key in keys]) for run in runs}
            rows.append(
                {
                    "field_type": CATEGORY_LABELS[category],
                    "metric": "recall",
                    "paired on": "distinct value",
                    "n_items": len(keys),
                    **paired_cluster_test(scores[baseline_run], scores[system_run], n_resamples=n_resamples, seed=seed),
                }
            )

        # Precision: the item is one (assay, field), since the runs assert different values.
        fields = [key for key in both_asserted if outcomes.in_category(key, category)]
        if fields:
            scores = {
                run: np.array([_field_precision(outcomes.asserted_by_field[key][run]) for key in fields])
                for run in runs
            }
            rows.append(
                {
                    "field_type": CATEGORY_LABELS[category],
                    "metric": "precision",
                    "paired on": "field",
                    "n_items": len(fields),
                    **paired_cluster_test(scores[baseline_run], scores[system_run], n_resamples=n_resamples, seed=seed),
                }
            )

    return rows
