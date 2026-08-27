"""Tests for the two levels the error tables are printed at.

The load-bearing property is that the category table is a faithful roll-up of the
sub-category table printed beneath it: the two are shown together so a reader can check the
correspondence down a column, and a taxonomy that disagreed with itself would print
plausible totals that happen to be wrong.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from analysis.data_analysis import (
    CATEGORIES,
    CATEGORY_BY_SUBCATEGORY,
    collect_field_errors,
    summarize_error_categories,
    summarize_error_subcategories,
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


def _case(root: Path, assay: str, name: str, *, gold: dict, predicted: dict, legacy: dict) -> None:
    _write(root / "schemas" / f"{assay}.json", SCHEMA)
    _write(root / assay / "gold" / f"{name}.json", gold)
    _write(root / assay / "input" / f"{name}.json", legacy)
    _write(root / assay / "output" / "m" / "sys" / f"{name}.json", predicted)


def _corpus(root: Path):
    """Two assays, between them reaching every category on both sides."""
    # A wrong value the record carries elsewhere, and a hedge in a field gold left blank.
    _case(
        root,
        "atacseq",
        "r1",
        gold={"tissue": "lung", "title": ""},
        predicted={"tissue": "kidney", "title": "Custom"},
        legacy={"tissue": "lung tissue", "extra": "kidney"},
    )
    # A right answer in the wrong shape, and a gold value the record never held.
    _case(
        root,
        "atacseq",
        "r2",
        gold={"tissue": "Lung Biopsy", "title": "a study"},
        predicted={"tissue": "lung  biopsy", "title": None},
        legacy={},
    )
    # A gold value sitting in the record under another name, left blank anyway.
    _case(
        root,
        "rnaseq",
        "r3",
        gold={"tissue": "spleen", "title": "kept"},
        predicted={"tissue": None, "title": "kept"},
        legacy={"organ": "spleen"},
    )
    return collect_field_errors(str(root), "m", "sys")


class TestTheTwoLevels:
    def test_the_category_table_rolls_up_the_subcategory_table_exactly(self, tmp_path: Path) -> None:
        errors = _corpus(tmp_path)
        if True:
            categories = summarize_error_categories(errors).set_index("category")
            subcategories = summarize_error_subcategories(errors)
            for category, row in categories.iterrows():
                members = [name for name in subcategories["subcategory"] if CATEGORY_BY_SUBCATEGORY[name] == category]
                expected = subcategories[subcategories["subcategory"].isin(members)]["total"].sum()
                assert row["total"] == expected, category

    def test_both_levels_total_the_side_s_error_count(self, tmp_path: Path) -> None:
        errors = _corpus(tmp_path)
        if True:
            expected = len(errors)
            assert summarize_error_categories(errors)["total"].sum() == expected
            assert summarize_error_subcategories(errors)["total"].sum() == expected

    def test_each_row_s_assay_columns_add_up_to_its_total(self, tmp_path: Path) -> None:
        errors = _corpus(tmp_path)
        if True:
            for table, level in (
                (summarize_error_categories(errors), "category"),
                (summarize_error_subcategories(errors), "subcategory"),
            ):
                assays = [column for column in table.columns if column not in (level, "total", "share")]
                assert table[assays].sum(axis=1).tolist() == table["total"].tolist()

    def test_the_subcategory_rows_run_in_their_categories_order(self, tmp_path: Path) -> None:
        # This is what makes the roll-up checkable down a column rather than by hunting.
        errors = _corpus(tmp_path)
        for order in (CATEGORIES,):
            walked = [CATEGORY_BY_SUBCATEGORY[name] for name in summarize_error_subcategories(errors)["subcategory"]]
            assert walked == sorted(walked, key=order.index), walked

    def test_shares_sum_to_one_at_both_levels(self, tmp_path: Path) -> None:
        errors = _corpus(tmp_path)
        if True:
            for table in (
                summarize_error_categories(errors),
                summarize_error_subcategories(errors),
            ):
                assert abs(table["share"].sum() - 1.0) < 5e-3

    def test_narrowing_to_a_field_type_narrows_the_counts(self, tmp_path: Path) -> None:
        errors = _corpus(tmp_path)
        whole = summarize_error_categories(errors)["total"].sum()
        halves = sum(
            summarize_error_categories(errors, field_type=field_type)["total"].sum()
            for field_type in ("ontology", "non_ontology")
        )
        assert halves == whole

    def test_no_errors_gives_an_empty_frame_rather_than_raising(self, tmp_path: Path) -> None:
        _case(tmp_path, "atacseq", "r", gold={"tissue": "lung"}, predicted={"tissue": "lung"}, legacy={})
        errors = collect_field_errors(str(tmp_path), "m", "sys")
        assert summarize_error_categories(errors).empty
        assert list(summarize_error_categories(errors).columns) == ["category", "total", "share"]
        assert list(summarize_error_subcategories(errors).columns) == [
            "subcategory",
            "total",
            "share",
        ]
