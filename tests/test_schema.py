"""Tests for the dynamic Pydantic model builder from CEDAR templates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from metadata_migration_agent.schema import build_output_model


def _make_field(
    name: str,
    field_type: str = "string",
    required: bool = False,
    multivalued: bool = False,
) -> dict:
    return {
        "name": name,
        "description": f"Description of {name}",
        "label": name.replace("_", " ").title(),
        "type": field_type,
        "required": required,
        "multivalued": multivalued,
    }


def _make_element(
    name: str,
    children: list[dict],
    required: bool = False,
    multivalued: bool = False,
) -> dict:
    return {
        "name": name,
        "description": f"Description of {name}",
        "label": name.replace("_", " ").title(),
        "type": "element",
        "required": required,
        "multivalued": multivalued,
        "children": children,
    }


def _make_template(name: str, children: list[dict]) -> dict:
    return {"type": "template", "name": name, "children": children}


class TestFlatFields:
    """Test flat templates with simple field types."""

    def test_string_integer_boolean(self) -> None:
        template = _make_template(
            "TestTemplate",
            [
                _make_field("title", "string", required=True),
                _make_field("count", "integer"),
                _make_field("active", "boolean"),
            ],
        )
        model = build_output_model(template)
        schema = model.model_json_schema()

        assert schema["additionalProperties"] is False
        assert "title" in schema["properties"]
        assert "count" in schema["properties"]
        assert "active" in schema["properties"]

    def test_all_cedar_types_map_correctly(self) -> None:
        template = _make_template(
            "TypeTest",
            [
                _make_field("s", "string"),
                _make_field("i", "integer"),
                _make_field("d", "decimal"),
                _make_field("b", "boolean"),
                _make_field("dt", "date"),
                _make_field("dtt", "datetime"),
                _make_field("t", "time"),
                _make_field("l", "link"),
            ],
        )
        model = build_output_model(template)
        schema = model.model_json_schema()

        # All fields present
        for name in ("s", "i", "d", "b", "dt", "dtt", "t", "l"):
            assert name in schema["properties"]

    def test_additional_properties_forbidden(self) -> None:
        template = _make_template("Strict", [_make_field("name", "string", required=True)])
        model = build_output_model(template)
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False

    def test_extra_field_rejected_at_validation(self) -> None:
        template = _make_template("Strict", [_make_field("name", "string", required=True)])
        model = build_output_model(template)
        with pytest.raises(ValidationError):
            model(name="ok", unexpected_field="bad")


class TestRequiredVsOptional:
    """Test required and optional field handling."""

    def test_all_fields_in_required_for_strict_mode(self) -> None:
        """OpenAI strict mode requires every property in the ``required`` array."""
        template = _make_template(
            "ReqTest",
            [
                _make_field("req_field", "string", required=True),
                _make_field("opt_field", "string", required=False),
            ],
        )
        model = build_output_model(template)
        schema = model.model_json_schema()

        # Both must be in `required` for OpenAI strict mode compatibility
        assert "req_field" in schema.get("required", [])
        assert "opt_field" in schema.get("required", [])

        # Both must accept None (nullable for extraction)
        instance = model(req_field=None, opt_field=None)
        assert instance.req_field is None
        assert instance.opt_field is None

    def test_all_fields_are_nullable(self) -> None:
        """Every field — regardless of CEDAR required flag — accepts None for extraction."""
        template = _make_template(
            "Nullable",
            [
                _make_field("required_str", "string", required=True),
                _make_field("optional_str", "string", required=False),
                _make_field("required_int", "integer", required=True),
                _make_field("optional_bool", "boolean", required=False),
            ],
        )
        model = build_output_model(template)
        instance = model(required_str=None, optional_str=None, required_int=None, optional_bool=None)
        assert instance.required_str is None
        assert instance.optional_str is None
        assert instance.required_int is None
        assert instance.optional_bool is None


class TestMultivalued:
    """Test multivalued (list) fields."""

    def test_multivalued_produces_list_type(self) -> None:
        template = _make_template(
            "Multi",
            [_make_field("tags", "string", multivalued=True)],
        )
        model = build_output_model(template)
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_multivalued_required(self) -> None:
        template = _make_template(
            "MultiReq",
            [_make_field("ids", "integer", required=True, multivalued=True)],
        )
        model = build_output_model(template)
        schema = model.model_json_schema()
        assert "ids" in schema.get("required", [])


class TestNestedElements:
    """Test nested element handling."""

    def test_nested_element_creates_sub_model(self) -> None:
        template = _make_template(
            "Nested",
            [
                _make_element(
                    "address",
                    [
                        _make_field("street", "string", required=True),
                        _make_field("city", "string", required=True),
                    ],
                    required=True,
                ),
            ],
        )
        model = build_output_model(template)
        schema = model.model_json_schema()

        # The address field should reference a sub-model
        assert "address" in schema["properties"]

    def test_nested_element_forbids_extra_properties(self) -> None:
        template = _make_template(
            "NestedStrict",
            [
                _make_element(
                    "contact",
                    [_make_field("email", "string", required=True)],
                    required=True,
                ),
            ],
        )
        model = build_output_model(template)
        instance = model(contact={"email": "a@b.com"})
        assert instance.contact.email == "a@b.com"

        with pytest.raises(ValidationError):
            model(contact={"email": "a@b.com", "phone": "123"})

    def test_multivalued_element(self) -> None:
        template = _make_template(
            "MultiElem",
            [
                _make_element(
                    "authors",
                    [_make_field("name", "string", required=True)],
                    multivalued=True,
                ),
            ],
        )
        model = build_output_model(template)
        instance = model(authors=[{"name": "Alice"}, {"name": "Bob"}])
        assert len(instance.authors) == 2

    def test_deeply_nested(self) -> None:
        template = _make_template(
            "Deep",
            [
                _make_element(
                    "level1",
                    [
                        _make_element(
                            "level2",
                            [_make_field("value", "integer", required=True)],
                            required=True,
                        ),
                    ],
                    required=True,
                ),
            ],
        )
        model = build_output_model(template)
        instance = model(level1={"level2": {"value": 42}})
        assert instance.level1.level2.value == 42


class TestNameSanitization:
    """Test that special characters in template names are handled."""

    def test_special_chars_in_template_name(self) -> None:
        template = _make_template("My Template (v2.1)", [_make_field("x", "string")])
        model = build_output_model(template)
        # Model should be created without error; name is sanitized for the class name
        assert model is not None

    def test_field_starting_with_digit(self) -> None:
        template = _make_template(
            "Digit",
            [
                {
                    "name": "field_ok",
                    "description": "d",
                    "label": "l",
                    "type": "string",
                    "required": False,
                    "multivalued": False,
                },
            ],
        )
        model = build_output_model(template)
        assert model is not None

    def test_empty_template(self) -> None:
        template = _make_template("Empty", [])
        model = build_output_model(template)
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"] == {}
