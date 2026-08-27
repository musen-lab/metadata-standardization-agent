"""Build the template material the prompt-only arm is given.

``baseline`` is told the field names, plus a note on each field whose values come
from an ontology branch saying which ontology it is.  The permissible values
themselves are not given, so the model has to supply them from its own knowledge.

The fetch lives here rather than in the condition itself because it is the same
fetch the tool arm makes, under the same cache key.  The rest is what the template
does not supply directly: the flat field-name list and the ontology notes the
baseline prompt is assembled from.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from cedar_mcp.external_api import get_template
from cedar_mcp.processing import clean_template_response

from arms_agent.cache import SqliteCache

if TYPE_CHECKING:
    from collections.abc import Iterator


_cache: SqliteCache | None = None


def _get_cache() -> SqliteCache:
    """Return the module-level cache instance, creating it on first use."""
    global _cache  # noqa: PLW0603
    if _cache is None:
        _cache = SqliteCache()
    return _cache


def _get_api_key(name: str) -> str:
    """Return the named API key from the environment."""
    key = os.environ.get(name, "")
    if not key:
        msg = f"{name} environment variable is not set"
        raise ValueError(msg)
    return key


def fetch_cedar_template(template_id: str) -> dict[str, Any]:
    """Fetch a CEDAR template by its ID or full URL and return its cleaned structure.

    The same fetch every arm makes, cached under the key the tool arm uses so the
    two share a single copy of each template.

    Args:
        template_id: The template ID (UUID) or full CEDAR URL.

    Returns:
        The cleaned template, or CEDAR's error payload if the fetch failed.
    """
    cached = _get_cache().get("get_cedar_template", template_id=template_id)
    if cached is not None:
        return cached

    template_data = get_template(template_id, _get_api_key("CEDAR_API_KEY"))
    if "error" in template_data:
        return template_data

    result = clean_template_response(template_data)
    _get_cache().set("get_cedar_template", result, template_id=template_id)
    return result


def collect_field_names(children: list[dict[str, Any]], prefix: str = "") -> list[str]:
    """Recursively collect dot-notation field names from template children.

    Args:
        children: The ``children`` list from a CEDAR template or element.
        prefix: Dot-notation prefix for nested elements (e.g. ``"address."``).

    Returns:
        A flat list of field names such as ``["sample_name", "address.city"]``.
    """
    names: list[str] = []
    for child in children:
        name = child.get("name", "")
        if child.get("type") == "element" and "children" in child:
            names.extend(collect_field_names(child["children"], prefix=f"{prefix}{name}."))
        else:
            names.append(f"{prefix}{name}")
    return names


def _branch_constraints(children: list[dict[str, Any]], prefix: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(dotted name, constraint)`` for every branch-constrained field, depth first."""
    for child in children:
        name = child.get("name", "")
        if child.get("type") == "element" and "children" in child:
            yield from _branch_constraints(child["children"], prefix=f"{prefix}{name}.")
            continue
        for constraint in child.get("permissible_values") or []:
            if constraint.get("type") == "branch":
                yield f"{prefix}{name}", constraint


def collect_constraint_lines(template: dict[str, Any]) -> list[str]:
    """Return the ontology notes the ``baseline`` prompt carries.

    A branch constraint is named by its ontology and nothing more: the baseline
    arm is told where a value comes from, never which values are permitted.

    Args:
        template: A cleaned CEDAR template.

    Returns:
        Instruction lines such as
        ``["- tissue: value should be one of the UBERON ontology concepts"]``.
    """
    return [
        f"- {name}: value should be one of the {branch.get('ontology_acronym', '')} ontology concepts"
        for name, branch in _branch_constraints(template.get("children", []))
    ]
