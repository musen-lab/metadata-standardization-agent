"""Extraction utilities for post-processing agent output."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage
    from langchain_openai import ChatOpenAI

    from metadata_migration_agent.state import AgentState

from metadata_migration_agent.schema import build_output_model
from metadata_migration_agent.tools import get_cedar_template

logger = logging.getLogger(__name__)

_extraction_llm: ChatOpenAI | None = None


def _get_extraction_llm() -> ChatOpenAI:
    """Return a shared extraction LLM client, creating it on first use."""
    global _extraction_llm  # noqa: PLW0603
    if _extraction_llm is None:
        from langchain_openai import ChatOpenAI as _ChatOpenAI

        _extraction_llm = _ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    return _extraction_llm


def extract_output_metadata(state: AgentState) -> dict[str, Any]:
    """Post-processing node that extracts structured JSON from the agent's response.

    Reads the agent's final text output, builds a dynamic Pydantic schema from
    the CEDAR template, then uses a JSON-mode LLM call to produce a
    schema-conformant metadata dict.

    Args:
        state: The current agent state containing messages and cedar_template_iri.

    Returns:
        A partial state update with the ``metadata`` key populated.
    """
    final_text = extract_agent_final_response(state["messages"])
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
    llm = _get_extraction_llm()
    result = llm.invoke(
        f"Extract the JSON metadata object from the following text. "
        f"Return only the JSON object, nothing else. "
        f"Use null (not empty strings) for any field whose value is unknown, "
        f"missing, or empty.\n\n{final_text}",
        **model_kwargs,
    )
    metadata = json.loads(result.content)
    logger.debug("Extracted metadata with %d top-level keys", len(metadata))
    return {"metadata": metadata}


def extract_agent_final_response(messages: list[AnyMessage]) -> str:
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
    logger.debug("Scanning %d messages for final agent response", len(messages))
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
