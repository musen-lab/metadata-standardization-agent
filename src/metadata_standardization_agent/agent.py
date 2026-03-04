"""LangGraph ReAct agent for metadata migration."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from metadata_standardization_agent.prompts import SYSTEM_PROMPT
from metadata_standardization_agent.state import AgentState
from metadata_standardization_agent.tools import all_tools

logger = logging.getLogger(__name__)

# OpenAI o-series reasoning models don't support parallel_tool_calls
_O_SERIES = re.compile(r"^o\d")


def build_migration_agent(model: str) -> CompiledStateGraph:
    """Build the ReAct migration agent with tools.

    Args:
        model: The OpenAI model identifier to use.

    Returns:
        A compiled ReAct agent graph.
    """
    from langchain_openai import ChatOpenAI

    logger.debug("Building migration agent with model=%s, tools=%d", model, len(all_tools))
    model_kwargs: dict[str, Any] = {}
    if not _O_SERIES.match(model):
        model_kwargs["parallel_tool_calls"] = True
    llm = ChatOpenAI(model=model, temperature=0, model_kwargs=model_kwargs)
    return create_agent(llm, tools=all_tools, system_prompt=SYSTEM_PROMPT, state_schema=AgentState)
