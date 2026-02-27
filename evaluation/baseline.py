"""Baseline workflow: single LLM call (no tools) followed by structured extraction."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from metadata_migration_agent.state import AgentState
from metadata_migration_agent.utils import extract_output_metadata

logger = logging.getLogger(__name__)


def build_baseline_workflow() -> CompiledStateGraph:
    """Build the baseline workflow: single LLM migrate followed by structured extraction.

    Returns:
        A compiled LangGraph with ``START -> migrate -> extract -> END``.
    """
    graph = StateGraph(AgentState)
    graph.add_node("migrate", baseline_migrate_node)
    graph.add_node("extract", extract_output_metadata)
    graph.add_edge(START, "migrate")
    graph.add_edge("migrate", "extract")
    graph.add_edge("extract", END)
    logger.debug("Compiling baseline workflow graph: migrate -> extract")
    return graph.compile()


def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    """Build the user prompt for the baseline workflow.

    Fetches the CEDAR template identified by *template_iri* and dynamically
    generates the field list and ontology constraint instructions from the
    template's ``children`` structure.
    """
    from metadata_migration_agent.tools import get_cedar_template

    template = get_cedar_template.invoke({"template_id": template_iri})
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


def baseline_migrate_node(state: AgentState) -> dict[str, Any]:
    """Perform a single LLM call to migrate legacy metadata without tools.

    The user message (built by ``build_user_prompt``) already contains all
    necessary context, so no system message or template fetch is needed.

    Args:
        state: The current agent state containing messages.

    Returns:
        A partial state update appending the AI response to messages.
    """
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])

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
