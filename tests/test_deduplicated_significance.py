"""Tests for the per-assay deduplicated summary and the deduplicated paired tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from analysis.data_analysis import create_per_assay_deduplicated_precision_recall_summary
from analysis.significance import (
    collect_deduplicated_outcomes,
    deduplicated_paired_tests,
    paired_cluster_test,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = {
    "children": [
        {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
        {"name": "title", "permissible_values": []},
    ]
}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _build_two_assays(root: Path) -> None:
    """Two assays whose gold shares the string "lung" under the same field name.

    atacseq: 'lung' in three records, always right; titles unique, always wrong.
    rnaseq:  'lung' in two records, always wrong.

    So keying clusters on the value alone would merge five instances into one; keying on
    (assay, field, value) keeps them apart, which is what the per-assay rows assume.
    """
    for assay in ("atacseq", "rnaseq"):
        _write(root / "schemas" / f"{assay}.json", SCHEMA)

    for index in range(3):
        _write(root / "atacseq" / "gold" / f"r{index}.json", {"tissue": "lung", "title": f"study {index}"})
        _write(
            root / "atacseq" / "output" / "m" / "sys" / f"r{index}.json",
            {"tissue": "lung", "title": "WRONG"},
        )
    for index in range(2):
        _write(root / "rnaseq" / "gold" / f"r{index}.json", {"tissue": "lung", "title": f"t{index}"})
        _write(
            root / "rnaseq" / "output" / "m" / "sys" / f"r{index}.json",
            {"tissue": "liver", "title": f"t{index}"},
        )


class TestPerAssayDeduplicatedSummary:
    def test_one_row_per_assay_and_category(self, tmp_path: Path) -> None:
        _build_two_assays(tmp_path)
        table = create_per_assay_deduplicated_precision_recall_summary(str(tmp_path), "m", "sys")
        assert set(table["assay"]) == {"ATACseq", "RNAseq"}
        assert len(table) == 2 * 3  # two assays x {ontology, non_ontology, all}

    def test_clusters_do_not_merge_across_assays(self, tmp_path: Path) -> None:
        # "lung" is right in ATACseq and wrong in RNAseq.  Merged, the two would average
        # to a partial credit in one row; kept apart they are 1.0 and 0.0.
        _build_two_assays(tmp_path)
        table = create_per_assay_deduplicated_precision_recall_summary(str(tmp_path), "m", "sys").set_index(
            ["assay", "category"]
        )
        assert table.loc[("ATACseq", "ontology"), "recall"] == 1.0
        assert table.loc[("RNAseq", "ontology"), "recall"] == 0.0
        assert table.loc[("ATACseq", "ontology"), "n_gold_values"] == 1  # three records, one value

    def test_repeated_wrong_answer_is_one_assertion(self, tmp_path: Path) -> None:
        # Three records, three distinct gold titles, one repeated wrong answer.
        _build_two_assays(tmp_path)
        row = create_per_assay_deduplicated_precision_recall_summary(str(tmp_path), "m", "sys").set_index(
            ["assay", "category"]
        )
        assert row.loc[("ATACseq", "non_ontology"), "n_gold_values"] == 3
        assert row.loc[("ATACseq", "non_ontology"), "n_asserted_values"] == 1

    def test_assay_without_predictions_is_skipped(self, tmp_path: Path) -> None:
        _build_two_assays(tmp_path)
        for path in (tmp_path / "rnaseq" / "output" / "m" / "sys").glob("*.json"):
            path.unlink()
        table = create_per_assay_deduplicated_precision_recall_summary(str(tmp_path), "m", "sys")
        assert set(table["assay"]) == {"ATACseq"}

    def test_no_predictions_at_all_returns_empty(self, tmp_path: Path) -> None:
        _build_two_assays(tmp_path)
        table = create_per_assay_deduplicated_precision_recall_summary(str(tmp_path), "m", "absent")
        assert table.empty


class TestPairedClusterTest:
    def test_identical_runs_cannot_reject(self) -> None:
        scores = np.array([0.2, 0.5, 0.9, 1.0])
        result = paired_cluster_test(scores, scores, n_resamples=200)
        assert result["delta"] == 0.0
        assert result["pvalue"] == 1.0  # every swap reproduces the observed difference

    def test_a_clear_difference_is_detected(self) -> None:
        baseline = np.zeros(40)
        system = np.ones(40)
        result = paired_cluster_test(baseline, system, n_resamples=1000)
        assert result["delta"] == 1.0
        assert result["pvalue"] < 0.05
        assert result["lo"] == result["hi"] == 1.0  # no variation to resample

    def test_interval_brackets_the_difference(self) -> None:
        rng = np.random.default_rng(1)
        baseline = rng.uniform(0.0, 0.6, 60)
        system = baseline + 0.2
        result = paired_cluster_test(baseline, system, n_resamples=500)
        assert result["lo"] <= result["delta"] <= result["hi"]

    def test_seed_makes_it_reproducible(self) -> None:
        rng = np.random.default_rng(2)
        baseline, system = rng.uniform(size=30), rng.uniform(size=30)
        first = paired_cluster_test(baseline, system, n_resamples=300, seed=7)
        second = paired_cluster_test(baseline, system, n_resamples=300, seed=7)
        assert first == second


class TestDeduplicatedPairedTests:
    def _build_pair(self, root: Path) -> None:
        """One assay where the system beats the baseline on the ontology field only."""
        _write(root / "schemas" / "atacseq.json", SCHEMA)
        for index in range(4):
            _write(root / "atacseq" / "gold" / f"r{index}.json", {"tissue": f"organ{index}", "title": f"t{index}"})
            _write(
                root / "atacseq" / "output" / "m" / "base" / f"r{index}.json",
                {"tissue": "WRONG", "title": f"t{index}"},
            )
            _write(
                root / "atacseq" / "output" / "m" / "sys" / f"r{index}.json",
                {"tissue": f"organ{index}", "title": f"t{index}"},
            )

    def test_pairing_units_are_reported(self, tmp_path: Path) -> None:
        self._build_pair(tmp_path)
        rows = deduplicated_paired_tests(str(tmp_path), "m", baseline_run="base", system_run="sys", n_resamples=200)
        units = {(row["metric"], row["paired on"]) for row in rows}
        assert units == {("recall", "distinct value"), ("precision", "field")}

    def test_recall_items_are_distinct_gold_values(self, tmp_path: Path) -> None:
        self._build_pair(tmp_path)
        rows = deduplicated_paired_tests(str(tmp_path), "m", baseline_run="base", system_run="sys", n_resamples=200)
        ontology_recall = next(r for r in rows if r["metric"] == "recall" and r["field_type"].startswith("Ontology"))
        assert ontology_recall["n_items"] == 4  # four distinct organs
        assert ontology_recall["baseline"] == 0.0
        assert ontology_recall["system"] == 1.0

    def test_precision_items_are_fields(self, tmp_path: Path) -> None:
        self._build_pair(tmp_path)
        rows = deduplicated_paired_tests(str(tmp_path), "m", baseline_run="base", system_run="sys", n_resamples=200)
        all_precision = next(r for r in rows if r["metric"] == "precision" and r["field_type"] == "All fields")
        assert all_precision["n_items"] == 2  # tissue and title, whatever values they hold

    def test_records_only_one_run_produced_are_skipped(self, tmp_path: Path) -> None:
        self._build_pair(tmp_path)
        (tmp_path / "atacseq" / "output" / "m" / "sys" / "r0.json").unlink()
        outcomes = collect_deduplicated_outcomes(str(tmp_path), "m", baseline_run="base", system_run="sys")
        # organ0 and t0 came only from the dropped record, so their clusters are gone.
        assert not any("organ0" in key[2] or key[2] == '"t0"' for key in outcomes.recall_clusters)
        # Three remaining records, each contributing one tissue and one title cluster.
        assert len(outcomes.recall_clusters) == 6
        assert all(set(cluster) == {"base", "sys"} for cluster in outcomes.recall_clusters.values())
