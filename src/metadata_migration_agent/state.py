"""Agent state definitions for the metadata migration agent."""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Message-based state for the ReAct migration agent.

    Inherits a ``messages`` list from ``MessagesState`` which stores the
    full conversation history including tool calls and responses.
    """
