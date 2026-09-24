"""How far the prompt-only vs. agent-tool difference can be trusted.

The comparison is between two ways of standardizing the same record: the **prompt-only**
approach, which spells the template out in the prompt and calls the LLM once, and the
**agent-tool** approach (ARMS), which fetches the CEDAR template and queries BioPortal
at inference time.  The prompt-only condition is ``baseline``, which supplies the
template's field and vocabulary names and nothing more, and it is what these tables
compare against ARMS.

Each condition writes to its own directory under ``data/<assay>/output/<model>/``: the
prompt-only condition under ``baseline/``, ARMS under ``arms-agent/``, each holding one
``run-<n>/`` per repeat, and every function here reads ``run-1`` unless given ``run``.  The code
follows those directories, which is why ``baseline`` and ``arms`` are what the
parameters, columns and tuple fields below are called.

:mod:`analysis.data_analysis` reports what the numbers are; this package reports how
much of the gap between the two approaches could be an artifact of which records
happened to be sampled.  It reads the prediction files already on disk (no LLM/API
calls) and computes:

* **Bootstrap 95% confidence intervals** on micro precision, recall and F1, for each
  condition and for the paired difference, resampling whole records.
* **Record-clustered permutation test** on the difference in precision and recall,
  swapping whole records between the two conditions, with the per-assay p-values
  corrected over the table (Holm or Benjamini-Hochberg).
* **Deduplicated paired tests**, taking the distinct value (recall) or the field
  (precision) as the unit, so values the corpus repeats are counted once.
* **Single-condition intervals** on accuracy and precision/recall/F1, for a condition
  measured on its own.

Each is produced for the three field categories used in the paper (``ontology``,
``non_ontology``, ``all``).

Everything is paired -- a record counts only when both conditions produced it -- and the
record, not the field, is the unit of resampling, because the corpus repeats the same
correction across many records.  :mod:`~analysis.significance.clustering` measures how
much that repetition costs in independent evidence.

The modules follow that pipeline:

* :mod:`~analysis.significance.paired_data` -- one pass over the predictions producing
  every paired view the estimators need.
* :mod:`~analysis.significance.bootstrap` -- confidence intervals, resampling records.
* :mod:`~analysis.significance.hypothesis_tests` -- the record-clustered permutation
  test and the multiple-testing correction.
* :mod:`~analysis.significance.clustering` -- how much independent evidence the
  repeated corpus actually holds.
* :mod:`~analysis.significance.tables` -- the reported tables.

This package and :mod:`analysis.data_analysis` both walk the corpus through
:mod:`analysis.corpus`, so an interval and the point estimate it qualifies are computed
over the same files.

This module re-exports the whole surface, so ``from analysis.significance import ...``
reaches every name regardless of which module defines it.
"""

from __future__ import annotations

from analysis.significance.bootstrap import (
    _prf_from_sums,
    bootstrap_ci,
    bootstrap_pooled_accuracy,
    bootstrap_prf,
    cluster_bootstrap_prf,
    cluster_bootstrap_prf_delta,
)
from analysis.significance.clustering import effective_sample_size
from analysis.significance.deduplicated import (
    DeduplicatedOutcomes,
    collect_deduplicated_outcomes,
    deduplicated_paired_tests,
    paired_cluster_test,
)
from analysis.significance.hypothesis_tests import (
    adjust_pvalues,
    paired_permutation_prf,
)
from analysis.significance.paired_data import CATEGORIES, CATEGORY_LABELS, PairedData, collect_paired_data
from analysis.significance.single_condition import SingleConditionData, collect_single_condition_data
from analysis.significance.tables import (
    build_per_assay_precision_recall_table,
    build_precision_recall_table,
    build_single_condition_table,
)

__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "DeduplicatedOutcomes",
    "PairedData",
    "SingleConditionData",
    "_prf_from_sums",
    "adjust_pvalues",
    "bootstrap_ci",
    "bootstrap_pooled_accuracy",
    "bootstrap_prf",
    "build_per_assay_precision_recall_table",
    "build_precision_recall_table",
    "build_single_condition_table",
    "cluster_bootstrap_prf",
    "cluster_bootstrap_prf_delta",
    "collect_deduplicated_outcomes",
    "collect_paired_data",
    "collect_single_condition_data",
    "deduplicated_paired_tests",
    "effective_sample_size",
    "paired_cluster_test",
    "paired_permutation_prf",
]
