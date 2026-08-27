"""Agent state definitions for the metadata standardization agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Message-based state for the ReAct migration agent.

    Inherits a ``messages`` list from ``MessagesState`` which stores the
    full conversation history including tool calls and responses.

    ``structured_response`` is written by ``create_agent`` when the agent was built
    with a response format: it holds the validated final answer.  It has to be
    declared here because LangGraph keeps only the keys its state schema names.
    """

    cedar_template_iri: str
    metadata: dict[str, Any]
    decisions: list[dict[str, Any]]
    structured_response: Any
