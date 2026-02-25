"""Agent state definitions for the metadata migration agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Message-based state for the ReAct migration agent.

    Inherits a ``messages`` list from ``MessagesState`` which stores the
    full conversation history including tool calls and responses.
    """

    cedar_template_iri: str
    metadata: dict[str, Any]
