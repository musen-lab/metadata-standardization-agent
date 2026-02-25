"""Build a dynamic Pydantic model from a CEDAR template for structured output enforcement."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

# CEDAR type → Python type mapping
_CEDAR_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "decimal": float,
    "boolean": bool,
    "date": str,
    "datetime": str,
    "time": str,
    "link": str,
}

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_]")


class _StrictBase(BaseModel):
    """Base model that forbids extra fields."""

    model_config = ConfigDict(extra="forbid")


def _sanitize_name(name: str) -> str:
    """Sanitize a CEDAR field name into a valid Python identifier for model class names."""
    sanitized = _SANITIZE_RE.sub("_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"f_{sanitized}"
    return sanitized or "unnamed"


def _build_child_model(child: dict[str, Any], prefix: str) -> tuple[str, tuple[type, ...]]:
    """Build a (field_name, field_info_tuple) pair for a single CEDAR child definition.

    Returns:
        A tuple of (field_name, pydantic_field_tuple) suitable for passing to create_model.
    """
    name = child["name"]
    child_type = child.get("type", "string")
    multivalued = child.get("multivalued", False)

    if child_type == "element":
        # Recursively build a nested model
        python_type = _build_element_model(child, prefix)
    else:
        python_type = _CEDAR_TYPE_MAP.get(child_type, str)

    if multivalued:
        python_type = list[python_type]  # type: ignore[valid-type]

    # Always nullable: in the extraction context the LLM must be able to express
    # "unknown" as null.  OpenAI strict mode still requires every property in the
    # `required` array, which is handled by the `...` default below.
    python_type = python_type | None  # type: ignore[assignment]

    # Always use `...` (required) so every field appears in the JSON schema `required` list.
    # Optional fields already allow None via the union type above.
    field_tuple: tuple[type, ...] = (python_type, ...)

    return name, field_tuple


def _build_element_model(element: dict[str, Any], prefix: str) -> type[BaseModel]:
    """Recursively build a Pydantic model for a CEDAR ElementDefinition."""
    model_name = f"{prefix}_{_sanitize_name(element['name'])}"
    children = element.get("children", [])

    fields: dict[str, Any] = {}
    for child in children:
        field_name, field_tuple = _build_child_model(child, model_name)
        fields[field_name] = field_tuple

    return create_model(model_name, __base__=_StrictBase, **fields)


def build_output_model(template: dict[str, Any]) -> type[BaseModel]:
    """Build a dynamic Pydantic model from a CEDAR template dict.

    The template dict is the output of ``clean_template_response()`` — a serialized
    ``SimplifiedTemplate`` with ``type``, ``name``, and ``children`` keys.

    Args:
        template: The cleaned CEDAR template as a dict.

    Returns:
        A Pydantic model class with ``extra="forbid"`` that mirrors the template structure.
    """
    template_name = _sanitize_name(template.get("name", "CedarMetadata"))
    children = template.get("children", [])

    fields: dict[str, Any] = {}
    for child in children:
        field_name, field_tuple = _build_child_model(child, template_name)
        fields[field_name] = field_tuple

    return create_model(template_name, __base__=_StrictBase, **fields)
