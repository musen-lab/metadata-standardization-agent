"""Tests for the uncorrected (do-nothing) accuracy summary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from data_analysis import create_uncorrected_accuracy_summary

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
        df = create_uncorrected_accuracy_summary(str(tmp_path), decimal_places=4)
        assert abs(df["all_field_accuracy"].iloc[0] - 2 / 3) < 1e-3
        assert df["ontology_constrained_accuracy"].iloc[0] == 1.0  # tissue matches

    def test_populated_only_excludes_empty(self, tmp_path: Path) -> None:
        # Only tissue and title are populated in gold; note is excluded -> 1/2.
        _build_root(tmp_path)
        df = create_uncorrected_accuracy_summary(str(tmp_path), populated_only=True, decimal_places=4)
        assert df["all_field_accuracy"].iloc[0] == 0.5
