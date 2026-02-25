"""LangGraph ReAct agent for metadata migration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents import create_agent

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from metadata_migration_agent.prompts import SYSTEM_PROMPT
from metadata_migration_agent.state import AgentState
from metadata_migration_agent.tools import all_tools

logger = logging.getLogger(__name__)


def build_migration_agent(model: str = "gpt-5-mini") -> CompiledStateGraph:
    """Build the ReAct migration agent with tools.

    Args:
        model: The OpenAI model identifier to use.

    Returns:
        A compiled ReAct agent graph.
    """
    from langchain_openai import ChatOpenAI

    logger.debug("Building migration agent with model=%s, tools=%d", model, len(all_tools))
    llm = ChatOpenAI(model=model, temperature=0)
    return create_agent(llm, tools=all_tools, system_prompt=SYSTEM_PROMPT, state_schema=AgentState)
