"""How much independent evidence the corpus actually contains.

The corpus repeats the same ``(assay, field, gold-value)`` correction across many
records, so its 839 records do not carry 839 records' worth of independent evidence.
This module puts a number on that: the intra-cluster correlation of the correctness
outcome, the resulting design effect, and the effective sample size.  It is the
justification for why the rest of the package resamples records instead of fields --
and for reading the field-level McNemar p-values with suspicion.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from analysis.corpus import iter_assays, iter_pairs, matches_field_type, value_key
from analysis.metrics import _is_field_correct

if TYPE_CHECKING:
    from pathlib import Path

    from analysis.corpus import ValueKey


def effective_sample_size(
    data_root: str | Path,
    model: str,
    run_type: str,
    *,
    field_type: str | None = None,
) -> dict[str, float]:
    """Effective number of independent field observations, given the repetition.

    Groups every field instance by its ``(assay, field, gold-value)`` key, estimates
    the intra-cluster correlation (ICC) of the binary correctness outcome via a
    one-way random-effects ANOVA, and reports the design effect
    ``DEFF = 1 + (m_bar - 1) * ICC`` and the effective sample size
    ``n_effective = n / DEFF``.  ``field_type`` filters to ``"ontology"`` or
    ``"non_ontology"`` (default: all fields).  Returns a dict with ``n``,
    ``n_clusters``, ``icc``, ``design_effect``, and ``n_effective``.

    Fewer than two clusters leaves the ICC undefined, in which case ``n_effective``
    falls back to ``n``.
    """
    groups: dict[ValueKey, list[int]] = defaultdict(lambda: [0, 0])  # key -> [n, n_correct]

    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()

        for _gold_file, gold, predicted in iter_pairs(assay.gold_dir, assay.output_dir(model, run_type)):
            if predicted is None:
                continue
            for field_name, gold_val in gold.items():
                if not matches_field_type(field_name, ontology_fields, field_type):
                    continue
                correct = _is_field_correct(predicted, gold, field_name, match_case=True, match_whole_word=True)
                cell = groups[value_key(assay.key, field_name, gold_val)]
                cell[0] += 1
                cell[1] += int(correct)

    n = sum(c[0] for c in groups.values())
    k = len(groups)
    if k < 2 or n <= k:
        return {"n": n, "n_clusters": k, "icc": float("nan"), "design_effect": float("nan"), "n_effective": float(n)}

    grand = sum(c[1] for c in groups.values()) / n
    ss_between = sum(c[0] * ((c[1] / c[0]) - grand) ** 2 for c in groups.values())
    ss_within = sum(c[1] * (c[0] - c[1]) / c[0] for c in groups.values())  # binary within-SS = m*p*(1-p)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    m0 = (n - sum(c[0] ** 2 for c in groups.values()) / n) / (k - 1)
    denom = ms_between + (m0 - 1) * ms_within
    icc = max(0.0, (ms_between - ms_within) / denom) if denom > 0 else 0.0
    m_bar = n / k
    design_effect = 1 + (m_bar - 1) * icc
    return {
        "n": n,
        "n_clusters": k,
        "icc": icc,
        "design_effect": design_effect,
        "n_effective": n / design_effect,
    }
