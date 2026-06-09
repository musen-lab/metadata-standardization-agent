"""Tests for the significance / uncertainty analysis module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from significance import (
    bootstrap_ci,
    build_overall_table,
    cluster_bootstrap_pooled,
    collect_paired_data,
    paired_mcnemar,
    paired_wilcoxon,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestBootstrapCI:
    def test_constant_values(self) -> None:
        mean, lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert mean == 0.5
        assert lo == 0.5
        assert hi == 0.5

    def test_ci_brackets_mean(self) -> None:
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        mean, lo, hi = bootstrap_ci(values, seed=1)
        assert abs(mean - 0.6) < 1e-9
        assert lo <= mean <= hi
        assert lo < hi

    def test_empty_returns_nan(self) -> None:
        mean, lo, hi = bootstrap_ci([])
        assert mean != mean  # nan
        assert lo != lo
        assert hi != hi

    def test_reproducible_with_seed(self) -> None:
        values = [0.1, 0.3, 0.9, 0.5, 0.7]
        assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)


class TestPairedWilcoxon:
    def test_all_improve_is_significant(self) -> None:
        pairs = [(0.5, 0.9)] * 10
        _stat, p, n = paired_wilcoxon(pairs)
        assert n == 10
        assert p < 0.05

    def test_no_difference_returns_one(self) -> None:
        pairs = [(0.6, 0.6)] * 8
        _stat, p, n = paired_wilcoxon(pairs)
        assert n == 0
        assert p == 1.0

    def test_empty_returns_one(self) -> None:
        _stat, p, n = paired_wilcoxon([])
        assert p == 1.0
        assert n == 0


class TestPairedMcnemar:
    def test_known_counts(self) -> None:
        # 9 ARMS-only correct, 1 baseline-only correct, plus ties.
        outcomes = [(False, True)] * 9 + [(True, False)] * 1 + [(True, True)] * 5 + [(False, False)] * 5
        result = paired_mcnemar(outcomes)
        assert result["c"] == 9
        assert result["b"] == 1
        assert result["n_discordant"] == 10

    def test_no_discordant_pairs(self) -> None:
        outcomes = [(True, True)] * 4 + [(False, False)] * 4
        result = paired_mcnemar(outcomes)
        assert result["n_discordant"] == 0
        assert result["pvalue"] == 1.0

    def test_strongly_lopsided_is_significant(self) -> None:
        outcomes = [(False, True)] * 50 + [(True, False)] * 2
        result = paired_mcnemar(outcomes)
        assert result["pvalue"] < 0.001


class TestClusterBootstrapPooled:
    def test_pooled_is_field_weighted(self) -> None:
        # Record A: 1/1 correct; Record B: 1/3 correct. Pooled = 2/4 = 0.5,
        # which differs from the record-mean (1.0 + 0.333)/2 = 0.667.
        counts = [(1, 1, 1), (1, 1, 3)]
        point, lo, hi = cluster_bootstrap_pooled(counts, "baseline")
        assert abs(point - 0.5) < 1e-9
        assert lo <= point <= hi

    def test_empty_returns_nan(self) -> None:
        point, lo, hi = cluster_bootstrap_pooled([], "arms")
        assert point != point  # nan


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _build_mini_data_root(root: Path) -> None:
    """Create a tiny atacseq dataset: 2 records, baseline weaker than ARMS."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
        ]
    }
    _write_record(root / "schemas" / "atacseq.json", schema)
    for name in ("r1.json", "r2.json"):
        gold = {"tissue": "lung", "title": "study"}
        _write_record(root / "atacseq" / "gold" / name, gold)
        # baseline gets the ontology field wrong; ARMS gets everything right.
        _write_record(
            root / "atacseq" / "output" / "gpt5mini" / "baseline" / name, {"tissue": "WRONG", "title": "study"}
        )
        _write_record(
            root / "atacseq" / "output" / "gpt5mini" / "experiment" / name, {"tissue": "lung", "title": "study"}
        )


class TestCollectPairedData:
    def test_collects_records_and_fields(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        data = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        # 2 records, each with 1 ontology + 1 non-ontology field.
        assert len(data.record_acc["all"]) == 2
        assert len(data.field_outcomes["ontology"]) == 2
        assert len(data.field_outcomes["non_ontology"]) == 2
        # Ontology field: baseline wrong, ARMS right.
        assert data.field_outcomes["ontology"][0] == (False, True)

    def test_missing_assay_returns_empty(self, tmp_path: Path) -> None:
        data = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        assert data.record_acc["all"] == []


class TestBuildOverallTable:
    def test_overall_table_shape_and_values(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_overall_table(tmp_path, "gpt5mini")
        assert list(table["category"]) == [
            "Ontology-constrained",
            "Non-ontology-constrained",
            "All fields",
        ]
        ont = table[table["category"] == "Ontology-constrained"].iloc[0]
        # baseline 0/2 correct, ARMS 2/2; McNemar c=2, b=0.
        assert ont["mcnemar_c"] == 2
        assert ont["mcnemar_b"] == 0
