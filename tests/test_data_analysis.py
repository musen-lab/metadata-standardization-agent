"""Tests for the uncorrected (do-nothing) accuracy summary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from data_analysis import (
    create_deduplicated_accuracy_summary,
    create_frequency_split_accuracy_summary,
    create_uncorrected_accuracy_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _build_root(root: Path) -> None:
    """One atacseq record: legacy matches gold on 'tissue' but not 'title'; 'note' both empty."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
            {"name": "note", "permissible_values": []},
        ]
    }
    _write(root / "schemas" / "atacseq.json", schema)
    _write(root / "atacseq" / "gold" / "r1.json", {"tissue": "lung", "title": "study", "note": None})
    _write(root / "atacseq" / "input" / "r1.json", {"tissue": "lung", "title": "WRONG", "note": None})


class TestUncorrectedAccuracy:
    def test_all_fields_counts_both_empty(self, tmp_path: Path) -> None:
        # tissue correct, title wrong, note both-empty (correct) -> 2/3.
        _build_root(tmp_path)
        df = create_uncorrected_accuracy_summary(str(tmp_path))
        assert abs(df["all_field_accuracy"].iloc[0] - 2 / 3) < 1e-9
        assert df["ontology_constrained_accuracy"].iloc[0] == 1.0  # tissue matches

    def test_populated_only_excludes_empty(self, tmp_path: Path) -> None:
        # Only tissue and title are populated in gold; note is excluded -> 1/2.
        _build_root(tmp_path)
        df = create_uncorrected_accuracy_summary(str(tmp_path), populated_only=True)
        assert df["all_field_accuracy"].iloc[0] == 0.5


def _build_repetitive_root(root: Path) -> None:
    """3 atacseq records: a repeated ontology pair (always correct) and unique
    non-ontology values (always wrong), so deduplicated != instance-weighted."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
        ]
    }
    _write(root / "schemas" / "atacseq.json", schema)
    for i in range(3):
        name = f"r{i}.json"
        _write(root / "atacseq" / "gold" / name, {"tissue": "lung", "title": f"study{i}"})
        # ARMS: tissue right every time (1 unique pair); title wrong every time (3 unique pairs).
        _write(root / "atacseq" / "output" / "gpt5mini" / "experiment" / name, {"tissue": "lung", "title": "WRONG"})


class TestDeduplicatedAccuracy:
    def test_counts_each_unique_pair_once(self, tmp_path: Path) -> None:
        _build_repetitive_root(tmp_path)
        df = create_deduplicated_accuracy_summary(str(tmp_path), "gpt5mini", "experiment")
        row = df.iloc[0]
        # 1 unique ontology pair (correct) -> 1.0; 3 unique non-ontology pairs (wrong) -> 0.0.
        assert row["ontology_constrained_accuracy"] == 1.0
        assert row["non_ontology_constrained_accuracy"] == 0.0
        assert row["n_ontology_pairs"] == 1
        assert row["n_non_ontology_pairs"] == 3
        assert row["n_unique_pairs"] == 4
        # all-fields macro over 4 unique pairs = (1 + 0 + 0 + 0) / 4 = 0.25,
        # which differs from the instance-weighted 3/6 = 0.5.
        assert row["all_field_accuracy"] == 0.25


class TestFrequencySplitAccuracy:
    def test_splits_by_value_frequency(self, tmp_path: Path) -> None:
        # tissue="lung" recurs in all 3 records (ARMS correct); each title is unique
        # (ARMS wrong). So recurring -> 3/3 = 1.0, singleton -> 0/3 = 0.0.
        _build_repetitive_root(tmp_path)
        df = create_frequency_split_accuracy_summary(str(tmp_path), "gpt5mini", "experiment")
        rec = df[df["bucket"] == "recurring"].iloc[0]
        sing = df[df["bucket"] == "singleton"].iloc[0]
        assert rec["accuracy"] == 1.0
        assert rec["n_instances"] == 3
        assert sing["accuracy"] == 0.0
        assert sing["n_instances"] == 3

    def test_field_type_filter(self, tmp_path: Path) -> None:
        _build_repetitive_root(tmp_path)
        # Non-ontology fields are only the unique titles -> all singletons, all wrong.
        non = create_frequency_split_accuracy_summary(
            str(tmp_path), "gpt5mini", "experiment", field_type="non_ontology"
        )
        sing = non[non["bucket"] == "singleton"].iloc[0]
        rec = non[non["bucket"] == "recurring"].iloc[0]
        assert sing["n_instances"] == 3
        assert sing["accuracy"] == 0.0
        assert rec["n_instances"] == 0  # no recurring non-ontology values here
