"""Tests for the accuracy summaries and the precision/recall summaries."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import (
    analyze_prediction_errors,
    create_deduplicated_accuracy_summary,
    create_deduplicated_precision_recall_summary,
    create_frequency_split_accuracy_summary,
    create_overall_precision_recall_summary,
    create_per_assay_precision_recall_summary,
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
        _write(root / "atacseq" / "output" / "gpt5mini" / "agent-tool" / name, {"tissue": "lung", "title": "WRONG"})


class TestDeduplicatedAccuracy:
    def test_counts_each_unique_pair_once(self, tmp_path: Path) -> None:
        _build_repetitive_root(tmp_path)
        df = create_deduplicated_accuracy_summary(str(tmp_path), "gpt5mini", "agent-tool")
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


class TestDeduplicatedPrecisionRecall:
    def test_counts_each_distinct_value_once(self, tmp_path: Path) -> None:
        # gold: tissue="lung" in all 3 records, title unique per record.
        # run:  tissue="lung" (right every time), title="WRONG" (same wrong answer 3x).
        _build_repetitive_root(tmp_path)
        df = create_deduplicated_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        row = df[df["category"] == "all"].iloc[0]

        # Recall clusters key on the gold value: 1 for "lung" (reproduced) + 3 titles (not).
        assert row["n_gold_values"] == 4
        assert row["gold_values_reproduced"] == 1.0
        assert row["recall"] == 0.25
        # Precision clusters key on the asserted value: "lung" (right) and "WRONG" (wrong).
        # The same wrong answer three times is one distinct wrong assertion.
        assert row["n_asserted_values"] == 2
        assert row["asserted_values_correct"] == 1.0
        assert row["precision"] == 0.5

    def test_differs_from_instance_weighted_when_values_repeat(self, tmp_path: Path) -> None:
        """The whole point: repetition moves the instance-weighted number and not this one."""
        _build_repetitive_root(tmp_path)
        dedup = create_deduplicated_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        weighted = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")

        # Instance-weighted recall is 3 of 6 gold values produced; deduplicated is 1 of 4.
        assert weighted[weighted["category"] == "all"].iloc[0]["recall"] == 0.5
        assert dedup[dedup["category"] == "all"].iloc[0]["recall"] == 0.25

    def test_matches_instance_weighted_when_every_value_is_unique(self, tmp_path: Path) -> None:
        """With no repetition every cluster holds one instance, so macro == micro."""
        schema = {
            "children": [
                {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
                {"name": "title", "permissible_values": []},
            ]
        }
        _write(tmp_path / "schemas" / "atacseq.json", schema)
        for i in range(4):
            name = f"r{i}.json"
            _write(tmp_path / "atacseq" / "gold" / name, {"tissue": f"tissue{i}", "title": f"study{i}"})
            # Right on tissue, wrong on title, with a different wrong value each time.
            _write(
                tmp_path / "atacseq" / "output" / "gpt5mini" / "agent-tool" / name,
                {"tissue": f"tissue{i}", "title": f"wrong{i}"},
            )

        dedup = create_deduplicated_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        weighted = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        for category in ("ontology", "non_ontology", "all"):
            deduped_row = dedup[dedup["category"] == category].iloc[0]
            weighted_row = weighted[weighted["category"] == category].iloc[0]
            assert deduped_row["precision"] == weighted_row["precision"], category
            assert deduped_row["recall"] == weighted_row["recall"], category
            assert deduped_row["f1"] == weighted_row["f1"], category

    def test_cluster_counts_split_across_categories(self, tmp_path: Path) -> None:
        _build_repetitive_root(tmp_path)
        df = create_deduplicated_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool").set_index("category")
        for column in ("n_gold_values", "n_asserted_values"):
            assert df.loc["all", column] == df.loc["ontology", column] + df.loc["non_ontology", column]
        assert df.loc["ontology", "recall"] == 1.0  # tissue="lung", reproduced
        assert df.loc["non_ontology", "recall"] == 0.0  # every title missed

    def test_partial_credit_within_a_cluster(self, tmp_path: Path) -> None:
        """A value right in some records and wrong in others counts as the fraction."""
        schema = {"children": [{"name": "title", "permissible_values": []}]}
        _write(tmp_path / "schemas" / "atacseq.json", schema)
        out = tmp_path / "atacseq" / "output" / "gpt5mini" / "agent-tool"
        for i in range(4):
            _write(tmp_path / "atacseq" / "gold" / f"r{i}.json", {"title": "study"})
            _write(out / f"r{i}.json", {"title": "study" if i < 3 else "WRONG"})

        row = create_deduplicated_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool").iloc[-1]
        # One gold cluster "study", right in 3 of 4 records -> 0.75 of one distinct value.
        assert row["n_gold_values"] == 1
        assert row["gold_values_reproduced"] == 0.75
        assert row["recall"] == 0.75

    def test_missing_prediction_is_skipped_and_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _build_repetitive_root(tmp_path)
        (tmp_path / "atacseq" / "output" / "gpt5mini" / "agent-tool" / "r0.json").unlink()
        with caplog.at_level(logging.WARNING):
            df = create_deduplicated_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        assert "skipped 1 gold record" in caplog.text
        # r0's unique title is gone from gold's side too, so 3 distinct gold values remain.
        assert df[df["category"] == "all"].iloc[0]["n_gold_values"] == 3


class TestFrequencySplitAccuracy:
    def test_splits_by_value_frequency(self, tmp_path: Path) -> None:
        # tissue="lung" recurs in all 3 records (ARMS correct); each title is unique
        # (ARMS wrong). So recurring -> 3/3 = 1.0, singleton -> 0/3 = 0.0.
        _build_repetitive_root(tmp_path)
        df = create_frequency_split_accuracy_summary(str(tmp_path), "gpt5mini", "agent-tool")
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
            str(tmp_path), "gpt5mini", "agent-tool", field_type="non_ontology"
        )
        sing = non[non["bucket"] == "singleton"].iloc[0]
        rec = non[non["bucket"] == "recurring"].iloc[0]
        assert sing["n_instances"] == 3
        assert sing["accuracy"] == 0.0
        assert rec["n_instances"] == 0  # no recurring non-ontology values here


def _build_confusion_root(root: Path) -> None:
    """Two atacseq records covering all five confusion classes.

    r1: tissue TP, cell_type deletion, title insertion, note TN
    r2: tissue substitution, cell_type TN, title TP, note deletion
    """
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "cell_type", "permissible_values": [{"type": "branch"}]},
            {"name": "title", "permissible_values": []},
            {"name": "note", "permissible_values": []},
        ]
    }
    _write(root / "schemas" / "atacseq.json", schema)
    out = root / "atacseq" / "output" / "gpt5mini" / "agent-tool"

    gold_r1 = {"tissue": "lung", "cell_type": "T cell", "title": None, "note": None}
    _write(root / "atacseq" / "gold" / "r1.json", gold_r1)
    _write(out / "r1.json", {"tissue": "lung", "cell_type": None, "title": "invented", "note": None})

    _write(root / "atacseq" / "gold" / "r2.json", {"tissue": "lung", "cell_type": None, "title": "study", "note": "n"})
    _write(out / "r2.json", {"tissue": "liver", "cell_type": None, "title": "study", "note": None})


class TestPrecisionRecallSummaries:
    def test_overall_counts_and_scores(self, tmp_path: Path) -> None:
        _build_confusion_root(tmp_path)
        df = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        row = df[df["category"] == "all"].iloc[0]
        # TP: r1.tissue, r2.title.  substitutions: r2.tissue.  deletions: r1.cell_type, r2.note.
        # insertions: r1.title.  TN: r1.note, r2.cell_type.
        assert (row["TP"], row["TN"], row["substitutions"], row["insertions"], row["deletions"]) == (2, 2, 1, 1, 2)
        assert row["FP"] == row["insertions"] + row["substitutions"] == 2
        assert row["FN"] == row["deletions"] + row["substitutions"] == 3
        assert row["precision"] == 0.5  # 2 of 4 asserted values right
        assert row["recall"] == 0.4  # 2 of 5 gold values produced
        assert row["n_records"] == 2

    def test_overall_categories_sum_to_all(self, tmp_path: Path) -> None:
        _build_confusion_root(tmp_path)
        df = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool").set_index("category")
        for key in ("TP", "FP", "FN", "TN", "insertions", "deletions", "substitutions"):
            assert df.loc["all", key] == df.loc["ontology", key] + df.loc["non_ontology", key]

    def test_per_assay_matches_overall_for_a_single_assay(self, tmp_path: Path) -> None:
        _build_confusion_root(tmp_path)
        per_assay = create_per_assay_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        overall = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        assert list(per_assay["assay"]) == ["ATACseq"]
        assert per_assay["f1"].iloc[0] == overall[overall["category"] == "all"]["f1"].iloc[0]

    def test_per_assay_rejects_unknown_category(self, tmp_path: Path) -> None:
        _build_confusion_root(tmp_path)
        with pytest.raises(ValueError, match="category must be one of"):
            create_per_assay_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool", category="literal")

    def test_missing_prediction_is_skipped_and_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _build_confusion_root(tmp_path)
        unpredicted = {"tissue": "lung", "cell_type": None, "title": None, "note": None}
        _write(tmp_path / "atacseq" / "gold" / "r3.json", unpredicted)
        with caplog.at_level(logging.WARNING):
            df = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        assert df["n_records"].iloc[0] == 2
        assert "skipped 1 gold record" in caplog.text

    def test_reconciles_with_analyze_prediction_errors(self, tmp_path: Path) -> None:
        """insertions + deletions + substitutions must equal the error-row count."""
        _build_confusion_root(tmp_path)
        df = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "agent-tool")
        row = df[df["category"] == "all"].iloc[0]
        errors = analyze_prediction_errors(str(tmp_path), "gpt5mini", "agent-tool")
        assert row["insertions"] + row["deletions"] + row["substitutions"] == len(errors)
