"""Tests for the error-cause quantification module."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from error_causes import classify_cause, extract_empty_search_terms

if TYPE_CHECKING:
    from pathlib import Path


def _write_trace_csv(path: Path, tool_calls: list[tuple[str, int]]) -> None:
    """Write a minimal experiment trace CSV.

    *tool_calls* is a list of ``(search_string, total_count)`` pairs; each becomes
    one ``term_search`` tool row with realistic ``inputs``/``outputs`` shapes.
    """
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "run_type", "inputs", "outputs"])
        for term, total in tool_calls:
            inputs = f"{{'input': \"{{'search_string': '{term}', 'ontology_acronym': 'HRAVS'}}\"}}"
            collection = "[]" if total == 0 else "[{}]"
            content = f'{{"totalCount": {total}, "collection": {collection}}}'
            outputs = f"{{'output': {{'content': '{content}'}}}}"
            writer.writerow(["term_search_from_branch", "tool", inputs, outputs])


def _row(**kwargs: object) -> dict:
    """Build an error row with sensible defaults."""
    base = {
        "field": "f",
        "gold_value": None,
        "predicted_value": None,
        "field_type": "non-ontology-constrained",
        "error_type": "wrong_value",
    }
    base.update(kwargs)
    return base


class TestClassifyCause:
    def test_format_or_variant(self) -> None:
        row = _row(field="title", gold_value="Yes", predicted_value="true", error_type="boolean_representation")
        assert classify_cause(row, {"title": "true"}, None) == "format_or_variant"

    def test_field_mapping_confusion(self) -> None:
        row = _row(
            field="protocols_io_doi",
            gold_value="10.x/assay",
            predicted_value="10.x/section",
            error_type="wrong_value",
        )
        record = {"protocols_io_doi": "10.x/assay", "section_prep_protocols_io_doi": "10.x/section"}
        assert classify_cause(row, record, None) == "field_mapping_confusion"

    def test_external_information_required(self) -> None:
        row = _row(
            field="umi_size",
            gold_value="16",
            predicted_value=None,
            error_type="missed_non_null",
        )
        record = {"description": "an rnaseq run", "title": "study"}
        assert classify_cause(row, record, None) == "external_information_required"

    def test_external_not_applied_to_ontology_fields(self) -> None:
        row = _row(
            field="tissue",
            gold_value="single nucleus",
            predicted_value="nucleus",
            field_type="ontology-constrained",
            error_type="wrong_value",
        )
        # gold not literally in legacy, but ontology canonicalization -> not external.
        assert classify_cause(row, {"tissue": "nucleus"}, None) == "other"

    def test_missing_ontology_result_with_traces(self) -> None:
        row = _row(
            field="acquisition_instrument_model",
            gold_value="Axio Scan.Z1",
            predicted_value="AxioScan.Z1",
            field_type="ontology-constrained",
            error_type="delimiter_or_case",
        )
        record = {"acquisition_instrument_model": "AxioScan.Z1"}
        empty_terms = {"axioscan.z1"}
        assert classify_cause(row, record, empty_terms) == "missing_ontology_result"

    def test_missing_ontology_takes_priority_over_format(self) -> None:
        # Same row as above but without trace info -> falls back to surface label.
        row = _row(
            field="acquisition_instrument_model",
            gold_value="Axio Scan.Z1",
            predicted_value="AxioScan.Z1",
            field_type="ontology-constrained",
            error_type="delimiter_or_case",
        )
        record = {"acquisition_instrument_model": "AxioScan.Z1"}
        assert classify_cause(row, record, None) == "format_or_variant"

    def test_other(self) -> None:
        row = _row(
            field="tissue",
            gold_value="lung",
            predicted_value="liver",
            field_type="ontology-constrained",
            error_type="wrong_value",
        )
        assert classify_cause(row, {"tissue": "spleen"}, set()) == "other"


class TestExtractEmptySearchTerms:
    def test_detects_empty_and_excludes_recovered(self, tmp_path: Path) -> None:
        # AxioScan.Z1 -> totalCount 0 (empty); Zeiss -> totalCount 1 (found).
        _write_trace_csv(
            tmp_path / "x-gpt5mini-experiment-traces.csv",
            [("AxioScan.Z1", 0), ("Zeiss", 1)],
        )
        empty = extract_empty_search_terms(tmp_path)
        assert "axioscan.z1" in empty
        assert "zeiss" not in empty

    def test_term_recovered_elsewhere_not_empty(self, tmp_path: Path) -> None:
        # 'lipids' returns nothing in one call but is found in another.
        _write_trace_csv(
            tmp_path / "y-gpt5mini-experiment-traces.csv",
            [("lipids", 0), ("lipids", 2)],
        )
        empty = extract_empty_search_terms(tmp_path)
        assert "lipids" not in empty  # found in at least one call

    def test_ignores_baseline_files(self, tmp_path: Path) -> None:
        # Files without 'experiment' in the name are skipped.
        _write_trace_csv(tmp_path / "z-gpt5mini-baseline-traces.csv", [("foo", 0)])
        empty = extract_empty_search_terms(tmp_path)
        assert empty == set()
