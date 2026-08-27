"""How far the prompt-only vs. agent-tool difference can be trusted.

The comparison is between two ways of standardizing the same record: the **prompt-only**
approach, which spells the template out in the prompt and calls the LLM once, and the
**agent-tool** approach (ARMS), which fetches the CEDAR template and queries BioPortal
at inference time.  The prompt-only condition is ``baseline``, which supplies the
template's field and vocabulary names and nothing more, and it is what these tables
compare against ARMS.

Each condition writes to its own directory under ``data/<assay>/output/<model>/``: the
prompt-only condition under its own name, ARMS under ``agent-tool/``.  The code
follows those directories, which is why ``baseline`` and ``arms`` are what the
parameters, columns and tuple fields below are called.

:mod:`analysis.data_analysis` reports what the numbers are; this package reports how
much of the gap between the two approaches could be an artifact of which records
happened to be sampled.  It reads the prediction files already on disk (no LLM/API
calls) and computes:

* **Bootstrap 95% confidence intervals** on per-record accuracy, for the prompt-only
  run and ARMS separately.
* **Paired Wilcoxon signed-rank test** on per-record accuracy (same record under
  both runs).
* **Paired McNemar test** on per-field correctness (same field of the same record
  under both runs), reporting ``b`` (only the prompt-only run correct), ``c`` (only
  ARMS correct), and the p-value.
* **Record-clustered permutation test** on the same discordant outcomes, but with
  the record as the independent unit (whole-record label swaps).  Unlike the flat
  McNemar test, this does not treat duplicated or within-record-correlated fields
  as independent, so it does not overstate significance.

All four are produced for each of the three field categories used in the paper
(``ontology``, ``non_ontology``, ``all``) and both per assay and pooled overall.

Everything is paired -- a record counts only when both runs produced it -- and the
record, not the field, is the unit of resampling, because the corpus repeats the same
correction across many records.  :mod:`~analysis.significance.clustering` measures how
much that repetition costs in independent evidence.

The modules follow that pipeline:

* :mod:`~analysis.significance.paired_data` -- one pass over the predictions producing
  every paired view the estimators need.
* :mod:`~analysis.significance.bootstrap` -- confidence intervals, resampling records.
* :mod:`~analysis.significance.hypothesis_tests` -- Wilcoxon, McNemar and the
  record-clustered permutation test.
* :mod:`~analysis.significance.clustering` -- how much independent evidence the
  repeated corpus actually holds.
* :mod:`~analysis.significance.tables` -- the reported tables.
* :mod:`~analysis.significance.cli` -- the command-line entry point.

This package and :mod:`analysis.data_analysis` both walk the corpus through
:mod:`analysis.corpus`, so an interval and the point estimate it qualifies are computed
over the same files.

Run from the project root::

    uv run python -m evaluation.analysis.significance --data-root data --model gpt5mini
    uv run python -m evaluation.analysis.significance --data-root data --model gpt5mini --csv-dir out/

Or, from the ``evaluation/`` directory (same convention as the notebook)::

    uv run python -m analysis.significance --data-root ../data --model gpt5mini

This module re-exports the whole surface, so ``from analysis.significance import ...``
reaches every name regardless of which module defines it.
"""

from __future__ import annotations

from analysis.significance.bootstrap import (
    _prf_from_sums,
    bootstrap_ci,
    bootstrap_pooled_accuracy,
    bootstrap_prf,
    cluster_bootstrap_pooled,
    cluster_bootstrap_prf,
    cluster_bootstrap_prf_delta,
)
from analysis.significance.cli import main
from analysis.significance.clustering import effective_sample_size
from analysis.significance.deduplicated import (
    DeduplicatedOutcomes,
    collect_deduplicated_outcomes,
    deduplicated_paired_tests,
    paired_cluster_test,
)
from analysis.significance.hypothesis_tests import (
    adjust_pvalues,
    paired_mcnemar,
    paired_permutation,
    paired_permutation_prf,
    paired_wilcoxon,
)
from analysis.significance.paired_data import CATEGORIES, CATEGORY_LABELS, PairedData, collect_paired_data
from analysis.significance.single_run import SingleRunData, collect_single_run_data
from analysis.significance.tables import (
    build_overall_table,
    build_per_assay_precision_recall_table,
    build_per_assay_table,
    build_precision_recall_table,
    build_single_run_table,
)

__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "DeduplicatedOutcomes",
    "PairedData",
    "SingleRunData",
    "_prf_from_sums",
    "adjust_pvalues",
    "bootstrap_ci",
    "bootstrap_pooled_accuracy",
    "bootstrap_prf",
    "build_overall_table",
    "build_per_assay_precision_recall_table",
    "build_per_assay_table",
    "build_precision_recall_table",
    "build_single_run_table",
    "cluster_bootstrap_pooled",
    "cluster_bootstrap_prf",
    "cluster_bootstrap_prf_delta",
    "collect_deduplicated_outcomes",
    "collect_paired_data",
    "collect_single_run_data",
    "deduplicated_paired_tests",
    "effective_sample_size",
    "main",
    "paired_cluster_test",
    "paired_mcnemar",
    "paired_permutation",
    "paired_permutation_prf",
    "paired_wilcoxon",
]
