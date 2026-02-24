"""LangGraph ReAct agent for metadata migration."""

from __future__ import annotations

from langchain.agents import create_agent

from metadata_migration_agent.prompts import SYSTEM_PROMPT
from metadata_migration_agent.state import AgentState
from metadata_migration_agent.tools import all_tools


def build_agent(model: str = "gpt-4o"):
    """Construct and return the compiled ReAct migration agent.

    Args:
        model: The OpenAI model identifier to use.

    Returns:
        A compiled LangGraph agent that can be invoked with ``AgentState``.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model)
    return create_agent(
        llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=AgentState,
    )
