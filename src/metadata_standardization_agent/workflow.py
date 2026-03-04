"""LangGraph workflow wiring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from metadata_standardization_agent.agent import build_migration_agent
from metadata_standardization_agent.state import AgentState
from metadata_standardization_agent.utils import extract_output_metadata

logger = logging.getLogger(__name__)


def build_workflow(model: str) -> CompiledStateGraph:
    """Build the full migration workflow: migrate followed by extract.

    The returned graph runs the ReAct migration agent followed by a structured
    extraction node that produces a schema-conformant metadata dict.

    Args:
        model: LLM model identifier passed to the migration agent.

    Returns:
        A compiled LangGraph that can be invoked with ``AgentState``.
    """
    migration_agent = build_migration_agent(model=model)

    graph = StateGraph(AgentState)
    graph.add_node("migrate", migration_agent)
    graph.add_node("extract", extract_output_metadata)
    graph.add_edge(START, "migrate")
    graph.add_edge("migrate", "extract")
    graph.add_edge("extract", END)
    logger.debug("Compiling workflow graph: migrate -> extract")
    return graph.compile()
