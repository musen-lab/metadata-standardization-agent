"""LangGraph workflow wiring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from langchain_core.runnables.base import RunnableLike
    from langgraph.graph.state import CompiledStateGraph

from arms_agent.state import AgentState
from arms_agent.utils import extract_output_metadata

logger = logging.getLogger(__name__)


def build_workflow(agent: RunnableLike) -> CompiledStateGraph:
    """Wire *agent* into the migration workflow: migrate followed by extract.

    The graph is the same for every way of producing a migration, so the agent is
    supplied rather than built here: the shipped CLI and the tool arm pass the ReAct
    agent from ``build_migration_agent``, the prompt-only evaluation conditions pass a
    single-call node.  Nothing below depends on which it was.

    Args:
        agent: The node that performs the migration.  It must leave the answer in state
            for the extraction node to read: either ``structured_response`` holding a
            validated object, or a final AI message for the extraction node to parse.

    Returns:
        A compiled LangGraph that can be invoked with ``AgentState``.
    """
    graph = StateGraph(AgentState)
    graph.add_node("migrate", agent)
    graph.add_node("extract", extract_output_metadata)
    graph.add_edge(START, "migrate")
    graph.add_edge("migrate", "extract")
    graph.add_edge("extract", END)
    logger.debug("Compiling workflow graph: migrate -> extract")
    return graph.compile()
