"""Tests for the template material the prompt-only arm is given."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conditions.prompt_only import baseline, template_spec
from conditions.prompt_only.template_spec import collect_constraint_lines, collect_field_names

TEMPLATE: dict[str, Any] = {
    "type": "template",
    "name": "demo",
    "children": [
        {"name": "sample_name", "type": "string"},
        {
            "name": "tissue",
            "type": "string",
            "permissible_values": [{"type": "branch", "ontology_acronym": "UBERON", "branch_iri": "http://x/UBERON_1"}],
        },
        {
            "name": "is_targeted",
            "type": "string",
            "permissible_values": [{"type": "literal", "options": ["Yes", "No"]}],
        },
    ],
}

NESTED: dict[str, Any] = {
    "type": "template",
    "name": "nested",
    "children": [
        {
            "name": "address",
            "type": "element",
            "children": [
                {"name": "city", "type": "string"},
                {
                    "name": "country",
                    "type": "string",
                    "permissible_values": [
                        {"type": "branch", "ontology_acronym": "GAZ", "branch_iri": "http://x/GAZ_1"}
                    ],
                },
            ],
        }
    ],
}

# The prompt the baseline arm must produce, captured from the implementation that
# predated the experiment. It is the published baseline condition, so it must not drift.
BASELINE_PROMPT = (
    'Given the following legacy metadata: {\n  "sample_name": "S1"\n}.\n'
    "Report a new and corrected metadata sample where the following template is as complete as possible:\n"
    "sample_name, tissue, is_targeted.\n"
    "Check if the field values and field names make sense. If no match is found for a field name, "
    "match it to an ontology. As far as possible, make field values adhere to ontology restrictions.\n"
    "- tissue: value should be one of the UBERON ontology concepts\n"
    "- Missing values: use null\n\n"
    "Do not provide any explanation. Output only the corrected record in Python dict format"
)

LEGACY = {"sample_name": "S1"}


@pytest.fixture(autouse=True)
def _stub_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve TEMPLATE instead of calling CEDAR."""
    monkeypatch.setattr(template_spec, "fetch_cedar_template", lambda _iri: TEMPLATE)


class TestCollectFieldNames:
    def test_flat_template(self) -> None:
        assert collect_field_names(TEMPLATE["children"]) == ["sample_name", "tissue", "is_targeted"]

    def test_nested_elements_are_dotted(self) -> None:
        assert collect_field_names(NESTED["children"]) == ["address.city", "address.country"]


class TestBaselineArm:
    def test_names_the_ontology_without_listing_values(self) -> None:
        assert collect_constraint_lines(TEMPLATE) == ["- tissue: value should be one of the UBERON ontology concepts"]

    def test_nested_field_names_are_dotted(self) -> None:
        assert collect_constraint_lines(NESTED) == [
            "- address.country: value should be one of the GAZ ontology concepts"
        ]

    def test_template_without_constraints_yields_no_lines(self) -> None:
        assert collect_constraint_lines({"children": [{"name": "x", "type": "string"}]}) == []

    def test_literal_options_are_not_stated(self) -> None:
        """The baseline arm is told where a value comes from, never which values are permitted."""
        assert not any("Yes" in line for line in collect_constraint_lines(TEMPLATE))

    def test_prompt_is_unchanged_from_before_the_experiment(self) -> None:
        """The baseline arm is the published condition and must not drift."""
        assert baseline.build_user_prompt(LEGACY, "iri") == BASELINE_PROMPT


class TestTheArmSharesTheRecord:
    def test_the_legacy_record_is_carried_verbatim(self) -> None:
        assert json.dumps(LEGACY, indent=2) in baseline.build_user_prompt(LEGACY, "iri")
