"""Baseline workflow: single LLM call (no tools) followed by structured extraction."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from cedar_mcp.external_api import get_template
from cedar_mcp.processing import clean_template_response
from langgraph.graph import END, START, StateGraph

from metadata_migration_agent.cache import SqliteCache

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from evaluation.prompts import BASELINE_SYSTEM_PROMPT
from metadata_migration_agent.state import AgentState
from metadata_migration_agent.utils import extract_output_metadata

logger = logging.getLogger(__name__)


def build_baseline_workflow(model: str) -> CompiledStateGraph:
    """Build the baseline workflow: single LLM migrate followed by structured extraction.

    Args:
        model: LLM model identifier used for the migration call.

    Returns:
        A compiled LangGraph with ``START -> migrate -> extract -> END``.
    """
    from functools import partial

    graph = StateGraph(AgentState)
    graph.add_node("migrate", partial(baseline_migrate_node, model=model))
    graph.add_node("extract", extract_output_metadata)
    graph.add_edge(START, "migrate")
    graph.add_edge("migrate", "extract")
    graph.add_edge("extract", END)
    logger.debug("Compiling baseline workflow graph: migrate -> extract")
    return graph.compile()


_cache: SqliteCache | None = None


def _get_cache() -> SqliteCache:
    """Return the module-level cache instance, creating it on first use."""
    global _cache  # noqa: PLW0603
    if _cache is None:
        _cache = SqliteCache()
    return _cache


def _get_cedar_api_key() -> str:
    """Return the CEDAR API key from the environment."""
    key = os.environ.get("CEDAR_API_KEY", "")
    if not key:
        raise ValueError("CEDAR_API_KEY environment variable is not set")
    return key


def _fetch_cedar_template(template_id: str) -> dict[str, Any]:
    """Fetch a CEDAR template by its ID or full URL and return its cleaned structure."""
    cached = _get_cache().get("get_cedar_template", template_id=template_id)
    if cached is not None:
        return cached

    cedar_api_key = _get_cedar_api_key()

    template_data = get_template(template_id, cedar_api_key)
    if "error" in template_data:
        return template_data

    result = clean_template_response(template_data)
    _get_cache().set("get_cedar_template", result, template_id=template_id)
    return result


def build_user_prompt_v1(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    """Build the original user prompt for the baseline workflow."""
    template = _fetch_cedar_template(template_iri)
    field_names = _collect_field_names(template["children"])
    ontology_lines = _collect_ontology_constraints(template["children"])

    field_list = ", ".join(field_names)
    prompt = (
        f"Given the following legacy metadata: {json.dumps(legacy_metadata, indent=2)}.\n"
        "Report a new and corrected metadata sample where the following "
        f"template is as complete as possible:\n{field_list}.\n"
        "Check if the field values and field names make sense. If no match "
        "is found for a field name, match it to an ontology. As far as possible, "
        "make field values adhere to ontology restrictions.\n"
    )
    if ontology_lines:
        prompt += "\n".join(ontology_lines) + "\n"
    prompt += "- Missing values: use null\nDo not provide any explanation"
    return prompt


def build_user_prompt_v2(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    """Build the second version user prompt for the baseline workflow."""
    template = _fetch_cedar_template(template_iri)
    field_names = _collect_field_names(template["children"])
    ontology_lines = _collect_ontology_constraints(template["children"])

    field_list = ", ".join(field_names)
    prompt = (
        f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
        f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n\n```"
        f"Target field list:\n{field_list}.\n\n"
        f"Ontology-constrained fields:\n"
    )
    if ontology_lines:
        prompt += "\n".join(ontology_lines) + "\n"
    return prompt


def baseline_migrate_node(state: AgentState, model: str) -> dict[str, Any]:
    """Perform a single LLM call to migrate legacy metadata without tools.

    The user message (built by ``build_user_prompt``) already contains all
    necessary context. A system message is prepended to guide the LLM's
    behaviour during migration.

    Args:
        state: The current agent state containing messages.
        model: LLM model identifier.

    Returns:
        A partial state update appending the AI response to messages.
    """
    from langchain_core.messages import AIMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    messages = [SystemMessage(content=BASELINE_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [AIMessage(content=response.content)]}


def _collect_field_names(children: list[dict[str, Any]], prefix: str = "") -> list[str]:
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
            names.extend(_collect_field_names(child["children"], prefix=f"{prefix}{name}."))
        else:
            names.append(f"{prefix}{name}")
    return names


def _collect_ontology_constraints(children: list[dict[str, Any]], prefix: str = "") -> list[str]:
    """Recursively collect ontology constraint instructions from template children.

    For each field whose ``permissible_values`` contain a ``"branch"`` constraint,
    a human-readable instruction line is emitted.

    Args:
        children: The ``children`` list from a CEDAR template or element.
        prefix: Dot-notation prefix for nested elements.

    Returns:
        A list of instruction strings such as
        ``["- tissue: value should be one of the UBERON ontology concepts"]``.
    """
    lines: list[str] = []
    for child in children:
        name = child.get("name", "")
        if child.get("type") == "element" and "children" in child:
            lines.extend(_collect_ontology_constraints(child["children"], prefix=f"{prefix}{name}."))
        else:
            for constraint in child.get("permissible_values", []) or []:
                if constraint.get("type") == "branch":
                    acronym = constraint.get("ontology_acronym", "")
                    lines.append(f"- {prefix}{name}: value should be one of the {acronym} ontology concepts")
    return lines
