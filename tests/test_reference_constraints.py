"""Objective gold exclusions must affect entire assays and both scoring units."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.reference_constraints import _value_violations, audit_reference_constraints
from analysis.significance import (
    build_per_assay_precision_recall_table,
    build_pre_post_precision_recall_table,
    build_precision_recall_table,
    collect_deduplicated_outcomes,
    collect_paired_data,
    deduplicated_paired_tests,
)
from notebook_utils import (
    show_post_adjudication_evaluation,
    show_pre_post_adjudication_comparison,
    show_reference_exclusions,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import CaptureFixture


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _schema(root: Path, assay: str, children: list[dict]) -> None:
    _write(root / "schemas" / f"{assay}.json", {"children": children})


@pytest.mark.parametrize("blank", [None, "", " \t\n"])
def test_required_blanks_are_unknowns(blank: object) -> None:
    assert not _value_violations(blank, {"type": "integer", "required": True, "pattern": "^[0-9]+$"})


@pytest.mark.parametrize(
    ("field_type", "value", "valid"),
    [
        ("integer", 0, True),
        ("integer", 2.0, True),
        ("integer", 1.5, False),
        ("integer", "2", False),
        ("integer", True, False),
        ("decimal", 0, True),
        ("decimal", 1.5, True),
        ("decimal", "1.5", False),
        ("decimal", False, False),
        ("decimal", float("inf"), False),
        ("decimal", float("nan"), False),
        ("string", "False", True),
        ("string", False, False),
        ("string", 1, False),
        ("boolean", False, True),
        ("boolean", 0, False),
        ("link", "https://example.org/", True),
        ("link", {}, False),
    ],
)
def test_json_types_without_coercion(field_type: str, value: object, valid: bool) -> None:
    violations = _value_violations(value, {"type": field_type})
    assert (not violations) == valid


def test_patterns_use_search_unless_anchored() -> None:
    assert not _value_violations("prefix12suffix", {"type": "string", "pattern": "[0-9]+"})
    assert _value_violations("prefix12suffix", {"type": "string", "pattern": "^[0-9]+$"})
    assert not _value_violations("HBM123.ABCD.456", {"type": "string", "pattern": r"^HBM\d{3}\.[A-Z]{4}\.\d{3}$"})


def test_permissible_values_are_exact_alternatives() -> None:
    field = {
        "type": "string",
        "permissible_values": [
            {"type": "branch", "options": ["Yes"]},
            {"type": "literal", "options": ["No"]},
        ],
    }
    for valid in ("Yes", "No"):
        assert not _value_violations(valid, field)
    for invalid in ("yes", "Yes ", "Maybe"):
        assert _value_violations(invalid, field) == [("permissible_values", ["Yes", "No"])]


@pytest.mark.parametrize("open_constraint", [{"type": "ontology"}, {"type": "branch", "options": []}])
def test_open_alternative_does_not_establish_nonmembership(open_constraint: dict) -> None:
    field = {"permissible_values": [{"type": "branch", "options": ["lung"]}, open_constraint]}
    assert not _value_violations("liver", field)


def test_multiplicity_checks_all_items() -> None:
    field = {"type": "string", "multivalued": True, "permissible_values": [{"options": ["a", "b"]}]}
    assert not _value_violations(["a", "b"], field)
    assert _value_violations(["a", "c"], field) == [("permissible_values", ["a", "b"])]
    assert _value_violations("a", field) == [("multivalued", True)]
    assert _value_violations(["a"], {"type": "string", "multivalued": False}) == [("multivalued", False)]


def _corpus(root: Path) -> None:
    children = [
        {"name": "tissue", "type": "string", "permissible_values": [{"type": "branch", "options": ["lung"]}]},
        {"name": "title", "type": "string"},
        {"name": "required_unknown", "type": "decimal", "required": True},
    ]
    for assay in ("atacseq", "rnaseq"):
        _schema(root, assay, children)
        for name, tissue in (("valid", "lung"), ("blank", None)):
            gold = {"tissue": tissue, "title": "kept", "required_unknown": None}
            _write(root / assay / "gold" / f"{name}.json", gold)
            for condition in ("baseline", "arms-agent"):
                # Both conditions disagree on tissue, including insertion at blank.
                _write(
                    root / assay / "output" / "m" / condition / "run-2" / f"{name}.json",
                    {"tissue": "liver", "title": "kept", "required_unknown": None},
                )
    # An unpredicted record still excludes the field in ATACseq, and only there.
    _write(root / "atacseq" / "gold" / "unpredicted.json", {"tissue": "invalid", "title": "kept"})


def test_exclusion_is_gold_only_and_assay_wide(tmp_path: Path) -> None:
    _corpus(tmp_path)
    audit = audit_reference_constraints(tmp_path)
    assert audit.excluded_fields_by_assay == {"atacseq": frozenset({"tissue"}), "rnaseq": frozenset()}
    row = audit.fields.iloc[0]
    assert row["status"] == "excluded"
    assert row["field_type"] == "ontology"
    assert row["n_invalid_records"] == 1
    assert row["n_gold_records"] == row["n_excluded_comparisons"] == 3
    assert json.loads(audit.violations.iloc[0]["records"]) == ["unpredicted.json"]
    assert audit.coverage["n_excluded_fields"].tolist() == [1, 0]

    original = collect_paired_data(tmp_path, "m", "atacseq", run=2)
    filtered = collect_paired_data(tmp_path, "m", "atacseq", run=2, excluded_fields={"tissue"})
    assert len(original.record_confusion["ontology"]) == 2
    assert not filtered.record_confusion["ontology"]
    assert filtered.record_confusion["all"] == filtered.record_confusion["non_ontology"]
    assert filtered.record_confusion["all"] == [(1, 0, 0, 1, 0, 0)] * 2
    assert collect_paired_data(tmp_path, "m", "atacseq", run=2).record_confusion == original.record_confusion


def test_same_exclusions_remove_recall_and_precision_items(tmp_path: Path) -> None:
    _corpus(tmp_path)
    audit = audit_reference_constraints(tmp_path)
    outcomes = collect_deduplicated_outcomes(
        tmp_path,
        "m",
        baseline="baseline",
        system="arms-agent",
        run=2,
        excluded_fields_by_assay=audit.excluded_fields_by_assay,
    )
    assert ("atacseq", "tissue") not in outcomes.field_types
    assert ("atacseq", "tissue") not in outcomes.asserted_by_field
    assert not any(key[:2] == ("atacseq", "tissue") for key in outcomes.recall_clusters)
    assert ("rnaseq", "tissue") in outcomes.asserted_by_field
    assert any(key[:2] == ("rnaseq", "tissue") for key in outcomes.recall_clusters)
    assert not collect_paired_data(tmp_path, "m", "atacseq", run=1).record_confusion["all"]


def test_reports_keep_all_witnesses_and_count_invalid_records_once(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    _schema(
        tmp_path,
        "atacseq",
        [{"name": "code", "type": "string", "pattern": "^A$", "permissible_values": [{"options": ["A"]}]}],
    )
    for name in ("r1", "r2"):
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", {"code": "<script>", "metadata_schema_id": "id"})
    audit = show_reference_exclusions(tmp_path, report_dir=tmp_path / "report")
    assert audit.fields.iloc[0]["n_invalid_records"] == 2
    assert len(audit.violations) == 2
    for records in audit.violations["records"]:
        assert json.loads(records) == ["r1.json", "r2.json"]
    assert json.loads(audit.coverage.iloc[0]["unchecked_fields"]) == ["metadata_schema_id"]
    html = (tmp_path / "report" / "reference_exclusions.html").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "r1.json" in html and "r2.json" in html and "<details>" in html
    assert "gold" not in html
    assert "n_reference_records" in html and "reference_value" in html
    assert "excluded" in capsys.readouterr().out
    assert (tmp_path / "report" / "reference_exclusions_violations.csv").is_file()


def test_exclusions_propagate_to_pooled_per_assay_and_notebook_tables(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    _corpus(tmp_path)
    exclusions = audit_reference_constraints(tmp_path).excluded_fields_by_assay
    per_assay = build_per_assay_precision_recall_table(tmp_path, "m", run=2, excluded_fields_by_assay=exclusions)
    assert not ((per_assay["assay"] == "ATACseq") & (per_assay["category"] == "Ontology-constrained")).any()
    pooled = build_precision_recall_table(tmp_path, "m", run=2, excluded_fields_by_assay=exclusions)
    ontology = pooled[pooled["category"] == "Ontology-constrained"]
    assert ontology["n_records"].tolist() == [2, 2, 2]  # RNAseq only
    result = show_post_adjudication_evaluation(
        tmp_path,
        "m",
        baseline="baseline",
        system="arms-agent",
        run=2,
        n_resamples=100,
        report_dir=tmp_path / "report",
    )
    assert result["audit"].excluded_fields_by_assay == exclusions
    assert "All assays" in result["field_level"]["assay"].tolist()
    assert not result["deduplicated"].empty
    output = capsys.readouterr().out
    assert "post-adjudication per-assay" in output
    assert "post-adjudication pooled" in output
    assert "post-adjudication deduplicated" in output
    assert (tmp_path / "report" / "post_adjudication_m_baseline_vs_arms-agent_run-2_field_level.csv").is_file()


def test_all_fields_excluded_returns_empty_items_and_zero_records(tmp_path: Path) -> None:
    _schema(tmp_path, "atacseq", [{"name": "count", "type": "integer"}])
    for directory in ("gold", "output/m/baseline", "output/m/arms-agent"):
        _write(tmp_path / "atacseq" / directory / "r.json", {"count": "invalid"})
    result = show_post_adjudication_evaluation(tmp_path, "m", baseline="baseline", system="arms-agent")
    assert result["field_level"]["n_records"].tolist() == [0] * 6
    assert result["per_assay_tests"].empty
    assert result["deduplicated"].empty
    assert not deduplicated_paired_tests(
        tmp_path,
        "m",
        baseline="baseline",
        system="arms-agent",
        excluded_fields_by_assay=result["audit"].excluded_fields_by_assay,
    )


def test_no_exclusions_preserves_original_results(tmp_path: Path) -> None:
    _schema(tmp_path, "atacseq", [{"name": "count", "type": "integer"}])
    for directory in ("gold", "output/m/baseline", "output/m/arms-agent"):
        _write(tmp_path / "atacseq" / directory / "r.json", {"count": 1})
    audit = audit_reference_constraints(tmp_path)
    assert audit.fields.empty and audit.violations.empty
    expected = build_precision_recall_table(tmp_path, "m")
    filtered = build_precision_recall_table(tmp_path, "m", excluded_fields_by_assay=audit.excluded_fields_by_assay)
    assert expected.equals(filtered)


@pytest.mark.parametrize("schema", [{}, {"children": [{"name": "nested", "type": "element", "children": []}]}])
def test_unsupported_template_fails_explicitly(tmp_path: Path, schema: dict) -> None:
    _write(tmp_path / "schemas" / "atacseq.json", schema)
    _write(tmp_path / "atacseq" / "gold" / "r.json", {})
    with pytest.raises(ValueError):
        audit_reference_constraints(tmp_path)


def test_missing_template_cannot_silently_skip_gold(tmp_path: Path) -> None:
    _write(tmp_path / "atacseq" / "gold" / "r.json", {"a": 1})
    with pytest.raises(FileNotFoundError, match="Missing template"):
        audit_reference_constraints(tmp_path)


def test_empty_corpus_has_stable_report_columns(tmp_path: Path) -> None:
    audit = audit_reference_constraints(tmp_path)
    assert not audit.excluded_fields_by_assay
    assert audit.fields.empty and audit.violations.empty and audit.coverage.empty
    assert "field" in audit.fields.columns and "records" in audit.violations.columns


def test_pre_post_changes_are_within_method_and_use_unrounded_scores(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    _schema(
        tmp_path,
        "atacseq",
        [
            {"name": "kept", "type": "string"},
            {"name": "excluded", "type": "string", "permissible_values": [{"options": ["valid"]}]},
        ],
    )
    for name, reference, system_value in (("r1", "invalid", "wrong"), ("r2", "valid", None)):
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", {"kept": "good", "excluded": reference})
        _write(
            tmp_path / "atacseq" / "output" / "m" / "baseline" / f"{name}.json",
            {"kept": "wrong", "excluded": reference},
        )
        _write(
            tmp_path / "atacseq" / "output" / "m" / "arms-agent" / f"{name}.json",
            {"kept": "good", "excluded": system_value},
        )
    # A baseline-only record is excluded from both views of the comparison.
    _write(tmp_path / "atacseq" / "gold" / "unpaired.json", {"kept": "good", "excluded": "valid"})
    _write(tmp_path / "atacseq" / "output" / "m" / "baseline" / "unpaired.json", {"kept": "good", "excluded": "valid"})
    post = show_post_adjudication_evaluation(
        tmp_path,
        "m",
        baseline="baseline",
        system="arms-agent",
        n_resamples=100,
    )
    comparison = show_pre_post_adjudication_comparison(
        tmp_path,
        "m",
        excluded_fields=post["audit"].excluded_fields_by_assay,
        post_deduplicated=post["deduplicated"],
        baseline="baseline",
        system="arms-agent",
        n_resamples=100,
    )
    all_fields = comparison["per_assay"].query("assay == 'ATACseq' and category == 'All fields'").set_index("metric")
    assert all_fields.loc["precision", "n_records_pre"] == all_fields.loc["precision", "n_records_post"] == 2
    assert all_fields.loc["precision", "baseline change"] == -0.5
    assert all_fields.loc["recall", "baseline change"] == -0.5
    assert all_fields.loc["precision", "arms-agent change"] == pytest.approx(1 / 3)
    assert all_fields.loc["recall", "arms-agent change"] == 0.5
    dedup = comparison["deduplicated"].query("field_type == 'All fields'").set_index("metric")
    assert dedup.loc["recall", "n_items_pre"] == 3
    assert dedup.loc["recall", "n_items_post"] == 1
    assert dedup.loc["recall", "baseline change"] == pytest.approx(-2 / 3)
    assert dedup.loc["recall", "arms-agent change"] == pytest.approx(2 / 3)
    assert dedup.loc["precision", "arms-agent change"] == 0.5
    output = capsys.readouterr().out
    assert "change = post minus pre" in output


def test_pre_post_comparison_does_not_score_an_entirely_excluded_category_as_zero(tmp_path: Path) -> None:
    _corpus(tmp_path)
    table = build_pre_post_precision_recall_table(
        tmp_path,
        "m",
        run=2,
        excluded_fields=audit_reference_constraints(tmp_path).excluded_fields_by_assay,
    )
    excluded = table.query("assay == 'ATACseq' and category == 'Ontology-constrained'")
    assert excluded["n_records_pre"].tolist() == [2, 2]
    assert excluded["n_records_post"].tolist() == [0, 0]
    assert excluded["baseline post"].isna().all()
    assert excluded["arms-agent change"].isna().all()


@pytest.mark.parametrize("assays", [(), ("atacseq",)])
def test_excluded_pr_plot_matches_paired_table_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assays: tuple[str, ...],
) -> None:
    import matplotlib.pyplot as plt

    from analysis.metrics import precision_recall_f1
    from plots import pr_space

    _schema(tmp_path, "atacseq", [{"name": "kept", "type": "integer"}, {"name": "excluded", "type": "integer"}])
    for name, kept, excluded, system_kept in (("r1", 1, "invalid", 2), ("r2", 2, 1, 2)):
        _write(tmp_path / "atacseq" / "gold" / f"{name}.json", {"kept": kept, "excluded": excluded})
        for condition, value in (("baseline", kept), ("arms-agent", system_kept)):
            _write(
                tmp_path / "atacseq" / "output" / "m" / condition / "run-2" / f"{name}.json",
                {"kept": value, "excluded": "different"},
            )
    # This baseline-only record must affect neither plotted method nor n_records.
    _write(tmp_path / "atacseq" / "gold" / "unpaired.json", {"kept": 3, "excluded": 1})
    _write(tmp_path / "atacseq" / "output" / "m" / "baseline" / "run-2" / "unpaired.json", {"kept": 3, "excluded": 1})
    audit = audit_reference_constraints(tmp_path)
    figures = []
    monkeypatch.setattr(pr_space, "_finish", lambda fig, _save_path: figures.append(fig))
    pr_space.plot_pr_condition_comp(
        str(tmp_path),
        "m",
        assays=assays,
        field_types=("all",),
        run=2,
        excluded_fields=audit.excluded_fields_by_assay,
        paired=True,
        show_f1_contours=False,
    )
    try:
        ax = figures[0].axes[0]
        plotted = sorted(
            (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
            for line in ax.get_lines()
            if line.get_linestyle() == "None"
            and len(line.get_xdata()) == 1
            and line.get_gid() != pr_space.STACK_COPY_GID
        )
        data = collect_paired_data(tmp_path, "m", "atacseq", run=2, excluded_fields={"excluded"})
        expected = []
        for offset in (0, 3):
            counts = {
                key: sum(row[offset + i] for row in data.record_confusion["all"])
                for i, key in enumerate(("TP", "FP", "FN"))
            }
            scores = precision_recall_f1(counts)
            expected.append((scores["recall"], scores["precision"]))
        assert plotted == sorted(expected) == [(0.5, 0.5), (1.0, 1.0)]
        assert "n=2" in ax.get_title()
    finally:
        for fig in figures:
            plt.close(fig)
