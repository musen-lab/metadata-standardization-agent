"""LangGraph ReAct agent for metadata migration."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage

from metadata_migration_agent.prompts import SYSTEM_PROMPT
from metadata_migration_agent.schema import build_output_model
from metadata_migration_agent.state import AgentState
from metadata_migration_agent.tools import all_tools, get_cedar_template

logger = logging.getLogger(__name__)


def _extract_final_response(messages: list[AnyMessage]) -> str:
    """Extract the final assistant text from a list of agent messages.

    In a ReAct agent, message ``content`` can be a plain string or a list of
    content blocks.  This walks the messages in reverse to find the last AI
    message that contains text.

    Args:
        messages: The full message list returned by the agent graph.

    Returns:
        The extracted text content.

    Raises:
        ValueError: If no AI message with text content is found.
    """
    for message in reversed(messages):
        if message.type != "ai":
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts = [block["text"] for block in content if isinstance(block, dict) and block.get("text")]
            if text_parts:
                return "\n".join(text_parts)
    msg = "Agent produced no text response."
    raise ValueError(msg)


def _extract_metadata(state: AgentState) -> dict[str, Any]:
    """Post-processing node that extracts structured JSON from the agent's response.

    Reads the agent's final text output, builds a dynamic Pydantic schema from
    the CEDAR template, then uses a JSON-mode LLM call to produce a
    schema-conformant metadata dict.

    Args:
        state: The current agent state containing messages and cedar_template_iri.

    Returns:
        A partial state update with the ``metadata`` key populated.
    """
    from langchain_openai import ChatOpenAI

    final_text = _extract_final_response(state["messages"])
    logger.debug("Raw agent response:\n%s", final_text)

    template_dict = get_cedar_template.invoke({"template_id": state["cedar_template_iri"]})
    output_model = build_output_model(template_dict)

    model_kwargs: dict[str, Any] = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cedar_metadata",
                "strict": True,
                "schema": output_model.model_json_schema(),
            },
        },
    }
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, model_kwargs=model_kwargs)
    result = llm.invoke(
        f"Extract the JSON metadata object from the following text. "
        f"Return only the JSON object, nothing else. "
        f"Use null (not empty strings) for any field whose value is unknown, "
        f"missing, or empty.\n\n{final_text}"
    )
    return {"metadata": json.loads(result.content)}


def build_agent(model: str = "gpt-5-mini"):
    """Construct and return the compiled migration agent with extraction post-processing.

    The returned graph runs the ReAct migration agent followed by a structured
    extraction node that produces a schema-conformant metadata dict.

    Args:
        model: The OpenAI model identifier to use.

    Returns:
        A compiled LangGraph that can be invoked with ``AgentState``.
    """
    from langchain_openai import ChatOpenAI

    logger.debug("Building agent with model=%s", model)
    llm = ChatOpenAI(model=model, temperature=0)
    react_agent = create_agent(
        llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=AgentState,
    )

    graph = StateGraph(AgentState)
    graph.add_node("agent", react_agent)
    graph.add_node("extract", _extract_metadata)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "extract")
    graph.add_edge("extract", END)
    return graph.compile()
