"""Tests for the error taxonomy.

Two load-bearing properties.  Reconciliation: every false positive and false negative the
precision/recall tables count must be recoverable from the labelled rows, since a taxonomy
that silently dropped errors would make every share it reports wrong.  And partition: each
error carries exactly one sub-category, whose category is the confusion case, so the two
levels cannot drift apart.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import (
    CATEGORIES,
    CATEGORY_BY_SUBCATEGORY,
    CONFUSION_CELLS_BY_CATEGORY,
    SUBCATEGORIES,
    category_shares,
    collect_field_errors,
    deduplicate_errors,
    near_match_reasons,
    reconcile_with_confusion,
    summarize_error_categories,
    summarize_error_subcategories,
    summarize_error_subcategories_by_field_type,
)
from analysis.data_analysis.error_taxonomy import (
    CONFIRMED_VALUE_MAPPING,
    DIFFERENT_VALUE,
    EXTERNAL_GAP,
    INSERTIONS,
    MISSED_VALUE,
    NEAR_MATCH,
    OMISSIONS,
    POOLED_ASSAY,
    SAME_READ_FORMAT,
    SAME_VALUE_OTHER_SHAPE,
    SUBSTITUTIONS,
    UNEXPECTED_COPY,
    UNEXPECTED_FILL,
    WRONG_MAPPING,
    _check_levels_agree,
    _legacy_fields_carrying,
)

if TYPE_CHECKING:
    from pathlib import Path

#: ``ms_scan_mode`` enumerates its permissible values and the others do not, so a fixture
#: can reach the vocabulary test or avoid it by choosing a field.
SCHEMA = {
    "children": [
        {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
        {"name": "title", "permissible_values": []},
        {"name": "count", "permissible_values": []},
        {"name": "ms_scan_mode", "permissible_values": [{"type": "branch", "options": ["MS1", "MS2"]}]},
    ]
}


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _case(root: Path, name: str, *, gold: dict, predicted: dict, legacy: dict, assay: str = "atacseq") -> None:
    """One record, with its legacy input and one run's prediction."""
    _write(root / "schemas" / f"{assay}.json", SCHEMA)
    _write(root / assay / "gold" / f"{name}.json", gold)
    _write(root / assay / "input" / f"{name}.json", legacy)
    _write(root / assay / "output" / "m" / "sys" / f"{name}.json", predicted)


def _one(root: Path) -> dict:
    """The single labelled row for a one-error fixture."""
    errors = collect_field_errors(str(root), "m", "sys")
    assert len(errors) == 1, errors.to_string()
    return errors.iloc[0].to_dict()


class TestSubstitutions:
    def test_near_match_when_only_the_shape_differs(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": "Lung Biopsy"}, predicted={"title": "lung  biopsy "}, legacy={})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (SUBSTITUTIONS, NEAR_MATCH)
        assert SAME_VALUE_OTHER_SHAPE in row["near_match_reasons"]

    def test_a_number_in_another_shape_is_close(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"count": 10}, predicted={"count": "10.0"}, legacy={})
        assert SAME_VALUE_OTHER_SHAPE in _one(tmp_path)["near_match_reasons"]

    @pytest.mark.parametrize(
        ("reference", "prediction"),
        [(5, -5), (-5, 5), ("5", "-5"), ("-0.5", ".5"), (".5", "5"), (9007199254740992, 9007199254740993)],
    )
    def test_distinct_numbers_do_not_qualify_as_near_matches(
        self, tmp_path: Path, reference: int | str, prediction: int | str
    ) -> None:
        _case(tmp_path, "r", gold={"count": reference}, predicted={"count": prediction}, legacy={})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    @pytest.mark.parametrize(("reference", "prediction"), [(".NET", "NET"), ("A-5", "A5"), ("sample/", "sample")])
    def test_meaningful_punctuation_is_preserved(self, tmp_path: Path, reference: str, prediction: str) -> None:
        _case(tmp_path, "r", gold={"title": reference}, predicted={"title": prediction}, legacy={})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_metadata_directory_notation_can_be_equivalent(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"data_path": "/Proteomics/"}, predicted={"data_path": "./Proteomics"}, legacy={})
        assert _one(tmp_path)["subcategory"] == NEAR_MATCH

    @pytest.mark.parametrize("prediction", ["../Proteomics", "./proteomics"])
    def test_distinct_dataset_paths_do_not_qualify_as_near_matches(self, tmp_path: Path, prediction: str) -> None:
        _case(tmp_path, "r", gold={"data_path": "./Proteomics"}, predicted={"data_path": prediction}, legacy={})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_numeric_looking_identifiers_keep_leading_zeros(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"sample_id": "001"}, predicted={"sample_id": "1"}, legacy={})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_text_containment_alone_is_not_close(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": "Orbitrap Fusion"}, predicted={"title": "Orbitrap Fusion Lumos"}, legacy={})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_copying_the_same_legacy_field_alone_is_not_close(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={"tissue": "kidney"})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_permitted_vocabulary_membership_alone_is_not_close(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"ms_scan_mode": "MS"},
            predicted={"ms_scan_mode": "MS1"},
            legacy={"ms_scan_mode": "MS"},
        )
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_prototype_to_custom_is_a_different_value(self, tmp_path: Path) -> None:
        field = "preparation_instrument_model"
        reference = "prototype robot - Stanford/Nolan Lab"
        _case(tmp_path, "r", gold={field: reference}, predicted={field: "Custom"}, legacy={field: reference})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE
        assert near_match_reasons(reference, "Custom", field, {field: reference}, {"custom"}) == []

    def test_doi_host_and_whitespace_differences_are_close(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"preparation_protocol_doi": "https://dx.doi.org/10.17504/protocols.io.3fugjnw "},
            predicted={"preparation_protocol_doi": "https://doi.org/10.17504/protocols.io.3fugjnw"},
            legacy={"protocols_io_doi": "10.17504/protocols.io.3fugjnw"},
        )
        row = _one(tmp_path)
        assert row["subcategory"] == NEAR_MATCH
        assert SAME_VALUE_OTHER_SHAPE in row["near_match_reasons"]

    @pytest.mark.parametrize("prediction", ["70/6/104", "70+6+104"])
    def test_read_length_separators_do_not_change_the_answer(self, tmp_path: Path, prediction: str) -> None:
        _case(
            tmp_path,
            "r",
            gold={"sequencing_read_format": "70,6,104"},
            predicted={"sequencing_read_format": prediction},
            legacy={},
        )
        row = _one(tmp_path)
        assert row["subcategory"] == NEAR_MATCH
        assert SAME_READ_FORMAT in row["near_match_reasons"]

    def test_read_length_order_changes_are_different_values(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"sequencing_read_format": "70,6,104"},
            predicted={"sequencing_read_format": "104/6/70"},
            legacy={},
        )
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE

    def test_confirmed_mapping_is_explicit_and_scoped_to_the_field(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"title": "legacy term"},
            predicted={"title": "standard term"},
            legacy={"title": "legacy term"},
        )
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE
        mappings = {"title": {("legacy term", "standard term"): "Confirmed by curator review"}}
        errors = collect_field_errors(tmp_path, "m", "sys", confirmed_value_mappings=mappings)
        assert errors.iloc[0]["subcategory"] == NEAR_MATCH
        assert CONFIRMED_VALUE_MAPPING in errors.iloc[0]["near_match_reasons"]
        assert (
            near_match_reasons("legacy term", "standard term", "tissue", {}, None, confirmed_value_mappings=mappings)
            == []
        )

    def test_mislocated_when_the_value_came_from_another_field(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"tissue": "lung"},
            predicted={"tissue": "SN123"},
            legacy={"tissue": "lung tissue", "title": "SN123"},
        )
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (SUBSTITUTIONS, WRONG_MAPPING)

    @pytest.mark.parametrize(
        ("field", "source", "reference", "prediction"),
        [
            ("dataset_type", "assay_type", "ATACseq", "SNARE-seq2"),
            ("barcode_offset", "cell_barcode_offset", "Not applicable", "0"),
            ("barcode_size", "cell_barcode_size", "Not applicable", "40"),
        ],
    )
    def test_valid_atacseq_mappings_with_different_values(
        self, tmp_path: Path, field: str, source: str, reference: str, prediction: str
    ) -> None:
        _case(
            tmp_path,
            "r",
            gold={field: reference},
            predicted={field: prediction},
            legacy={source: prediction, "unrelated_field": prediction},
        )
        row = _one(tmp_path)
        assert row["subcategory"] == DIFFERENT_VALUE
        assert row["legacy_sources"][source] == prediction
        assert not row["near_match_reasons"]

    def test_valid_mapping_requires_a_matching_value_in_the_valid_source(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"barcode_offset": "Not applicable"},
            predicted={"barcode_offset": "0"},
            legacy={"cell_barcode_offset": "12", "sequencing_phix_percent": "0"},
        )
        assert _one(tmp_path)["subcategory"] == WRONG_MAPPING

    def test_near_match_wins_over_mislocated(self, tmp_path: Path) -> None:
        # The asserted value equals gold's once shape is relaxed *and* sits in another
        # field.  Calling it a mislocation would report where a right answer came from.
        _case(tmp_path, "r", gold={"tissue": "Lung"}, predicted={"tissue": "lung"}, legacy={"title": "lung"})
        assert _one(tmp_path)["subcategory"] == NEAR_MATCH

    def test_completely_wrong_when_neither_near_gold_nor_in_the_record(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={"title": "unrelated"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (SUBSTITUTIONS, DIFFERENT_VALUE)

    def test_empty_normalized_values_do_not_establish_equivalence(self, tmp_path: Path) -> None:
        # Erasing both values through punctuation trimming is not evidence of equivalence.
        _case(tmp_path, "r", gold={"title": "/"}, predicted={"title": "."}, legacy={"title": "/"})
        assert _one(tmp_path)["subcategory"] == DIFFERENT_VALUE


class TestDeletionsAndInsertions:
    def test_underestimated_when_the_record_held_gold_s_value(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "lung"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (OMISSIONS, MISSED_VALUE)

    def test_dont_know_when_the_record_held_nothing(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "kidney"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (OMISSIONS, EXTERNAL_GAP)

    @pytest.mark.parametrize(
        ("assay", "field", "reference", "legacy"),
        [
            ("rnaseq", "library_preparation_kit", "Custom", {"rnaseq_assay_method": "SNARE-Seq2-RNA"}),
            ("rnaseq", "umi_read", "Read 2 (R2)", {"cell_barcode_read": "R2"}),
            (
                "atacseq",
                "sequencing_reagent_kit",
                "Illumina; NovaSeq 6000 S4 Reagent Kit v1.5 (300 cycles); PN 20028312",
                {"sequencing_reagent_kit": "NovaSeq 6000 S4 Reagent"},
            ),
            ("codex", "preparation_instrument_vendor", "Akoya Biosciences", {"preparation_instrument_vendor": "CODEX"}),
            ("af", "analyte_class", "Endogenous fluorophore", {"analyte_class": "", "assay_type": "AF"}),
            ("atacseq", "umi_read", "Not applicable", {"assay_type": "bulkATACseq"}),
        ],
    )
    def test_context_alone_does_not_establish_the_omitted_reference(
        self, tmp_path: Path, assay: str, field: str, reference: str, legacy: dict
    ) -> None:
        _case(tmp_path, "r", assay=assay, gold={field: reference}, predicted={field: None}, legacy=legacy)
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (OMISSIONS, EXTERNAL_GAP)
        assert row["omission_basis"] == "legacy hint insufficient to establish reference"
        assert not row["legacy_support_sources"]
        assert row["legacy_hint_sources"]
        assert not row["legacy_sources"]

    @pytest.mark.parametrize("legacy_value", [None, "", "  ", "Unknown", "not specified"])
    def test_empty_or_unknown_source_fields_are_not_hints(self, tmp_path: Path, legacy_value: str | None) -> None:
        _case(
            tmp_path,
            "r",
            gold={"acquisition_instrument_model": "NovaSeq 6000"},
            predicted={"acquisition_instrument_model": None},
            legacy={"acquisition_instrument_model": legacy_value},
        )
        row = _one(tmp_path)
        assert row["subcategory"] == EXTERNAL_GAP
        assert row["omission_basis"] == "no supporting legacy value detected"
        assert not row["legacy_hint_sources"]

    def test_unrelated_populated_fields_do_not_rule_out_an_external_gap(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"nuclear_marker_or_stain": "DAPI"},
            predicted={"nuclear_marker_or_stain": None},
            legacy={"assay_type": "Cell DIVE", "donor_id": "DAPI-study", "number_of_channels": 3},
            assay="celldive",
        )
        assert _one(tmp_path)["subcategory"] == EXTERNAL_GAP

    def test_numeric_zero_is_a_real_context_hint(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"barcode_offset": "Not applicable"},
            predicted={"barcode_offset": None},
            legacy={"cell_barcode_offset": 0},
        )
        row = _one(tmp_path)
        assert row["legacy_hint_sources"] == {"cell_barcode_offset": 0}
        assert row["subcategory"] == EXTERNAL_GAP
        assert not row["legacy_support_sources"]

    def test_nested_target_field_provides_a_context_hint(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"acquisition_instrument_model": "NovaSeq 6000"},
            predicted={"acquisition_instrument_model": None},
            legacy={"other_metadata": {"acquisition_instrument_model": "NovaSeq"}},
        )
        row = _one(tmp_path)
        assert row["subcategory"] == EXTERNAL_GAP
        assert row["legacy_hint_sources"] == {"other_metadata.acquisition_instrument_model": "NovaSeq"}

    def test_omission_hints_do_not_depend_on_the_agent_noticing_them(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"ms_scan_mode": "MS2"},
            predicted={"ms_scan_mode": None},
            legacy={"ms_scan_mode": "MS2"},
        )
        _write(
            tmp_path / "atacseq" / "output" / "m" / "sys" / "decisions" / "r.json",
            [{"key": "ms_scan_mode", "value": None, "legacy_fields": [], "reasoning": "No source found."}],
        )
        row = _one(tmp_path)
        assert row["subcategory"] == MISSED_VALUE
        assert row["legacy_support_sources"] == {"ms_scan_mode": "MS2"}

    def test_overestimated_when_the_run_wrote_a_value_the_record_holds(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": ""}, predicted={"title": "lung"}, legacy={"tissue": "lung"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (INSERTIONS, UNEXPECTED_COPY)

    @pytest.mark.parametrize(
        ("field", "reference", "legacy"),
        [
            ("umi_size", "12", {"library_pcr_cycles": "12", "cell_barcode_size": "12"}),
            ("umi_offset", "16", {"cell_barcode_size": "16"}),
            ("umi_read", "Read 1 (R1)", {"cell_barcode_read": "R1"}),
            ("umi_offset", "16", {"umi_offset": "17", "cell_barcode_size": "16"}),
            ("umi_read", "Not applicable", {"assay_type": "bulkATACseq"}),
            ("lc_column_vendor", "Waters", {"lc_instrument_vendor": "Waters"}),
        ],
    )
    def test_different_properties_and_conflicting_offsets_cannot_support_omissions(
        self, tmp_path: Path, field: str, reference: str, legacy: dict
    ) -> None:
        _case(tmp_path, "r", gold={field: reference}, predicted={field: None}, legacy=legacy)
        row = _one(tmp_path)
        assert row["subcategory"] == EXTERNAL_GAP
        assert not row["legacy_support_sources"]

    @pytest.mark.parametrize(
        ("field", "reference", "legacy", "source"),
        [
            ("umi_size", "12", {"umi_size": 12}, "umi_size"),
            ("umi_offset", "0", {"umi_offset": 0}, "umi_offset"),
            ("umi_read", "Read 1 (R1)", {"umi_read": "R1"}, "umi_read"),
            ("barcode_read", "Read 2 (R2)", {"cell_barcode_read": "R2"}, "cell_barcode_read"),
            ("barcode_size", "16", {"cell_barcode_size": "16"}, "cell_barcode_size"),
            ("umi_size", "12", {"other_metadata": {"umi_size": "12"}}, "other_metadata.umi_size"),
        ],
    )
    def test_direct_umi_evidence_and_valid_renames_support_omissions(
        self, tmp_path: Path, field: str, reference: str, legacy: dict, source: str
    ) -> None:
        _case(tmp_path, "r", gold={field: reference}, predicted={field: None}, legacy=legacy)
        row = _one(tmp_path)
        assert row["subcategory"] == MISSED_VALUE
        assert source in row["legacy_support_sources"]
        assert row["omission_basis"] == "reference established by legacy source"

    def test_renamed_source_with_a_different_value_does_not_establish_the_reference(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"barcode_read": "Read 2 (R2)"},
            predicted={"barcode_read": None},
            legacy={"cell_barcode_read": "I5"},
        )
        assert _one(tmp_path)["subcategory"] == EXTERNAL_GAP

    def test_too_optimistic_when_the_record_holds_nothing(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"title": ""}, predicted={"title": "from nowhere"}, legacy={"tissue": "lung"})
        row = _one(tmp_path)
        assert (row["category"], row["subcategory"]) == (INSERTIONS, UNEXPECTED_FILL)

    def test_the_two_pairs_mirror_each_other(self, tmp_path: Path) -> None:
        # One question -- did the record hold the value? -- asked of gold's for a deletion
        # and of the run's for an insertion.  The mirror is the point of the names.
        _case(tmp_path, "a", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "lung"})
        _case(tmp_path, "b", gold={"title": ""}, predicted={"title": "lung"}, legacy={"organ": "lung"})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert set(errors["subcategory"]) == {MISSED_VALUE, UNEXPECTED_COPY}


@pytest.mark.parametrize(
    ("legacy", "source"),
    [
        ({"other_metadata": {"Seq_run": "NS-1699"}}, "other_metadata.Seq_run"),
        ({"other_metadata": {"runs": [{"Seq_run": "NS-1699"}]}}, "other_metadata.runs[0].Seq_run"),
        ({"other_metadata": {"Seq_run": [None, "", "NS-1699"]}}, "other_metadata.Seq_run[2]"),
    ],
)
@pytest.mark.parametrize(
    ("reference", "prediction", "subcategory"),
    [
        (None, "NS-1699", UNEXPECTED_COPY),
        ("NS-1699", None, MISSED_VALUE),
        ("other run", "NS-1699", WRONG_MAPPING),
    ],
)
def test_nested_legacy_values_determine_provenance(
    tmp_path: Path, legacy: dict, source: str, reference: str | None, prediction: str | None, subcategory: str
) -> None:
    _case(
        tmp_path,
        "r",
        gold={"sequencing_batch_id": reference},
        predicted={"sequencing_batch_id": prediction},
        legacy=legacy,
    )
    assert _one(tmp_path)["subcategory"] == subcategory
    assert _legacy_fields_carrying("NS-1699", legacy) == {source}


@pytest.mark.parametrize(
    "legacy",
    [
        {"other_metadata": {"Seq_run": "NS-16990"}},
        {"other_metadata": {"NS-1699": "unrelated"}},
    ],
)
def test_nested_lookup_does_not_match_substrings_or_keys(tmp_path: Path, legacy: dict) -> None:
    _case(tmp_path, "r", gold={"title": None}, predicted={"title": "NS-1699"}, legacy=legacy)
    assert _one(tmp_path)["subcategory"] == UNEXPECTED_FILL


def test_nested_lookup_preserves_whole_container_matches() -> None:
    value = {"Seq_run": "NS-1699"}
    assert _legacy_fields_carrying(value, {"other_metadata": value}) == {"other_metadata"}


@pytest.mark.parametrize("recorded", ["225 pM", "225", "225.0"])
def test_numeric_values_recorded_with_units_are_available(tmp_path: Path, recorded: str) -> None:
    _case(
        tmp_path,
        "r",
        gold={"count": None},
        predicted={"count": 225.0},
        legacy={"other_metadata": {"concentration": recorded}},
    )
    row = _one(tmp_path)
    assert row["subcategory"] == UNEXPECTED_COPY
    assert row["legacy_sources"] == {"other_metadata.concentration": recorded}


@pytest.mark.parametrize("recorded", ["NS-225", "225 pM in pooled sample", "2250 pM"])
def test_numeric_availability_does_not_match_identifiers_or_free_text(tmp_path: Path, recorded: str) -> None:
    _case(tmp_path, "r", gold={"count": None}, predicted={"count": 225.0}, legacy={"other": recorded})
    assert _one(tmp_path)["subcategory"] == UNEXPECTED_FILL


@pytest.mark.parametrize(
    ("prediction", "recorded"),
    [(-5, "5"), (5, "-5"), (-5, "5 mM"), (9007199254740993, "9007199254740992")],
)
def test_legacy_availability_preserves_numeric_sign_and_precision(
    tmp_path: Path, prediction: int, recorded: str
) -> None:
    _case(tmp_path, "r", gold={"count": None}, predicted={"count": prediction}, legacy={"other": recorded})
    assert _one(tmp_path)["subcategory"] == UNEXPECTED_FILL
    assert _one(tmp_path)["legacy_sources"] == {}


class TestPartition:
    def _mixed(self, tmp_path: Path):
        _case(
            tmp_path,
            "mixed",
            gold={"tissue": "lung", "title": "", "count": 5, "ms_scan_mode": "MS1"},
            predicted={"tissue": "kidney", "title": "spurious", "count": None, "ms_scan_mode": "MS1"},
            legacy={},
        )
        return collect_field_errors(str(tmp_path), "m", "sys")

    def test_one_row_per_error(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        assert len(errors) == errors["pointer"].nunique() == 3  # substitution + insertion + deletion

    def test_every_counted_error_is_recovered(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        counts = reconcile_with_confusion(errors, str(tmp_path), "m", "sys")
        assert counts["FP"]["counted"] == counts["FP"]["categorised"] == 2  # substitution + insertion
        assert counts["FN"]["counted"] == counts["FN"]["categorised"] == 2  # substitution + deletion

    def test_the_category_is_the_confusion_case(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        expected = {"substitution": SUBSTITUTIONS, "deletion": OMISSIONS, "insertion": INSERTIONS}
        assert [expected[case] for case in errors["case"]] == list(errors["category"])

    def test_costs_follows_the_case(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path).set_index("case")
        assert errors.loc["insertion", "costs"] == "precision"
        assert errors.loc["deletion", "costs"] == "recall"
        assert errors.loc["substitution", "costs"] == "precision and recall"

    def test_the_confusion_cells_match_what_each_category_costs(self) -> None:
        # The figure braces the categories with these labels, so a category naming the wrong
        # cells would contradict the reconciliation counted right beside it.
        assert CONFUSION_CELLS_BY_CATEGORY == {
            SUBSTITUTIONS: "FP + FN",
            OMISSIONS: "FN",
            INSERTIONS: "FP",
        }

    def test_the_cells_agree_with_the_costs_column(self, tmp_path: Path) -> None:
        errors = self._mixed(tmp_path)
        in_words = {"precision and recall": "FP + FN", "recall": "FN", "precision": "FP"}
        for _index, row in errors.iterrows():
            assert CONFUSION_CELLS_BY_CATEGORY[row["category"]] == in_words[row["costs"]]

    def test_every_subcategory_has_a_category(self) -> None:
        assert set(SUBCATEGORIES) == set(CATEGORY_BY_SUBCATEGORY)
        assert set(CATEGORY_BY_SUBCATEGORY.values()) == set(CATEGORIES)

    def test_the_subcategory_order_nests_inside_the_category_order(self) -> None:
        # The two tables print one above the other, so the finer order has to run in the
        # coarser one's order for the roll-up to be checkable by eye.
        walked = [CATEGORY_BY_SUBCATEGORY[name] for name in SUBCATEGORIES]
        assert walked == sorted(walked, key=CATEGORIES.index)

    def test_levels_that_contradict_each_other_are_refused(self, tmp_path: Path) -> None:
        # The category comes from the case and the sub-category from the classifier, by two
        # routes that could drift apart without anything raising.
        errors = self._mixed(tmp_path)
        errors.loc[errors.index[0], "category"] = OMISSIONS
        errors.loc[errors.index[0], "subcategory"] = NEAR_MATCH
        with pytest.raises(ValueError, match="wrong category"):
            _check_levels_agree(errors)

    def test_matches_and_blanks_produce_no_rows(self, tmp_path: Path) -> None:
        _case(
            tmp_path, "r", gold={"tissue": "lung", "title": ""}, predicted={"tissue": "lung", "title": None}, legacy={}
        )
        assert collect_field_errors(str(tmp_path), "m", "sys").empty


class TestNearMatchReasons:
    def test_formatting_equivalence_has_a_specific_reason(self) -> None:
        reasons = near_match_reasons("Orbitrap", "orbitrap ", "tissue", {"tissue": "Orbitrap"}, None)
        assert reasons == [SAME_VALUE_OTHER_SHAPE]

    def test_no_reasons_means_not_a_near_match(self) -> None:
        assert near_match_reasons("lung", "kidney", "tissue", {}, None) == []

    def test_rows_that_are_not_close_carry_no_reasons(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={})
        assert _one(tmp_path)["near_match_reasons"] == ""


class TestBookkeeping:
    def test_pointer_locates_the_field(self, tmp_path: Path) -> None:
        _case(tmp_path, "rec01", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert set(errors["pointer"]) == {"atacseq/rec01#tissue"}

    def test_the_run_s_own_reasoning_is_attached_when_kept(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        _write(
            tmp_path / "atacseq" / "output" / "m" / "sys" / "decisions" / "r.json",
            [{"key": "tissue", "resolution": "harmonized", "reasoning": "picked the nearest term"}],
        )
        row = _one(tmp_path)
        assert row["resolution"] == "harmonized"
        assert row["reasoning"] == "picked the nearest term"

    def test_no_decision_log_leaves_those_columns_blank(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        row = _one(tmp_path)
        assert row["resolution"] is None
        assert row["reasoning"] is None

    def test_summary_shares_sum_to_one(self, tmp_path: Path) -> None:
        for index in range(4):
            _case(
                tmp_path,
                f"r{index}",
                gold={"tissue": "lung"},
                predicted={"tissue": "kidney" if index else None},
                legacy={},
            )
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        for table in (summarize_error_categories(errors), summarize_error_subcategories(errors)):
            assert abs(table["share"].sum() - 1.0) < 1e-9

    def test_a_run_without_predictions_gives_an_empty_frame(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={})
        assert collect_field_errors(str(tmp_path), "m", "absent").empty

    def test_a_prediction_without_its_legacy_input_is_refused(self, tmp_path: Path) -> None:
        # Not a partial corpus -- the run read that input to write the prediction.  Left to
        # default to an empty record, every provenance test quietly answers "no".
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "kidney"}, legacy={"tissue": "lung"})
        (tmp_path / "atacseq" / "input" / "r.json").unlink()
        with pytest.raises(FileNotFoundError, match="no legacy record"):
            collect_field_errors(str(tmp_path), "m", "sys")


class TestDeduplication:
    def _repetitive(self, tmp_path: Path):
        # The same mistake in three records, and a different one in a fourth.
        for index in range(3):
            _case(
                tmp_path,
                f"same{index}",
                gold={"tissue": "lung"},
                predicted={"tissue": "kidney"},
                legacy={"title": "unrelated"},
            )
        _case(tmp_path, "other", gold={"tissue": "lung"}, predicted={"tissue": "spleen"}, legacy={})
        return collect_field_errors(str(tmp_path), "m", "sys")

    def test_a_repeated_mistake_collapses_to_one_row(self, tmp_path: Path) -> None:
        errors = self._repetitive(tmp_path)
        assert len(errors) == 4
        assert len(deduplicate_errors(errors)) == 2

    def test_the_instances_are_counted_not_discarded(self, tmp_path: Path) -> None:
        errors = self._repetitive(tmp_path)
        distinct = deduplicate_errors(errors)
        assert distinct["n_instances"].sum() == len(errors)
        assert sorted(distinct["n_instances"]) == [1, 3]

    def test_the_two_levels_still_agree_afterwards(self, tmp_path: Path) -> None:
        _check_levels_agree(deduplicate_errors(self._repetitive(tmp_path)))

    def test_the_same_values_labelled_differently_stay_apart(self, tmp_path: Path) -> None:
        # Same gold and same asserted value, but one record's legacy holds gold's value and
        # the other's does not, so the taxonomy calls them different things -- which makes
        # them different mistakes, and the sub-category has to be part of the key.
        _case(tmp_path, "a", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={"organ": "lung"})
        _case(tmp_path, "b", gold={"tissue": "lung"}, predicted={"tissue": None}, legacy={})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert set(errors["subcategory"]) == {MISSED_VALUE, EXTERNAL_GAP}
        assert len(deduplicate_errors(errors)) == 2

    def test_an_unhashable_value_does_not_raise(self, tmp_path: Path) -> None:
        # Gold and prediction hold JSON of any type, and a list would make a plain groupby
        # raise rather than group.
        _case(tmp_path, "r", gold={"tissue": ["lung", "left"]}, predicted={"tissue": ["lung"]}, legacy={})
        assert len(deduplicate_errors(collect_field_errors(str(tmp_path), "m", "sys"))) == 1

    def test_an_empty_frame_comes_back_empty(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={})
        distinct = deduplicate_errors(collect_field_errors(str(tmp_path), "m", "sys"))
        assert distinct.empty
        assert "n_instances" in distinct.columns


class TestFieldTypeSummary:
    def test_counts_partition_the_deduplicated_cases(self, tmp_path: Path) -> None:
        _case(
            tmp_path,
            "r",
            gold={"ms_scan_mode": "MS2", "title": "expected"},
            predicted={"ms_scan_mode": None, "title": None},
            legacy={"ms_scan_mode": "MS2"},
        )
        cases = deduplicate_errors(collect_field_errors(tmp_path, "m", "sys"))
        table = summarize_error_subcategories_by_field_type(cases).set_index("subcategory")
        assert table.loc[MISSED_VALUE, "ontology"] == 1
        assert table.loc[EXTERNAL_GAP, "non_ontology"] == 1
        assert table["total"].sum() == len(cases) == 2
        assert table["total"].equals(table[["ontology", "non_ontology"]].sum(axis=1))
        assert list(table.index) == list(SUBCATEGORIES)

    def test_no_disagreements_reports_zero_for_every_subcategory(self, tmp_path: Path) -> None:
        _case(tmp_path, "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={})
        table = summarize_error_subcategories_by_field_type(collect_field_errors(tmp_path, "m", "sys"))
        assert list(table.subcategory) == list(SUBCATEGORIES)
        assert table[["ontology", "non_ontology", "total"]].to_numpy().sum() == 0


class TestCategoryShares:
    def _corpus(self, tmp_path: Path):
        for index in range(3):
            _case(
                tmp_path,
                f"r{index}",
                gold={"tissue": "lung", "title": "", "count": 5},
                predicted={"tissue": "kidney", "title": "x", "count": None},
                legacy={},
            )
        return collect_field_errors(str(tmp_path), "m", "sys")

    def test_shares_sum_to_one_within_each_assay(self, tmp_path: Path) -> None:
        shares = category_shares(self._corpus(tmp_path))
        for _assay, frame in shares.groupby("assay", observed=True):
            assert abs(frame["share"].sum() - 1.0) < 1e-9

    def test_the_pooled_row_counts_every_error(self, tmp_path: Path) -> None:
        errors = self._corpus(tmp_path)
        shares = category_shares(errors)
        pooled = shares[shares["assay"] == POOLED_ASSAY]
        assert pooled["n"].sum() == len(errors)

    def test_a_category_with_no_errors_is_kept_as_a_zero(self, tmp_path: Path) -> None:
        # A figure draws one segment per category; one dropped for being empty in an assay
        # and present in another would silently reorder the stack between rows.
        shares = category_shares(self._corpus(tmp_path))
        pooled = shares[shares["assay"] == POOLED_ASSAY]
        assert list(pooled["category"]) == list(CATEGORIES)
