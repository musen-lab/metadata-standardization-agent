"""Entry point for running the metadata migration agent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI

from metadata_migration_agent.agent import build_agent
from metadata_migration_agent.logging_config import configure_logging
from metadata_migration_agent.schema import build_output_model
from metadata_migration_agent.tools import get_cedar_template

if TYPE_CHECKING:
    from pydantic import BaseModel

# Load environment variables from .env (project root)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env", override=True)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the migration agent with a legacy metadata record and CEDAR template IRI."""
    parser = argparse.ArgumentParser(
        description="Migrate a legacy metadata record to a CEDAR template format.",
    )
    parser.add_argument("legacy_metadata", help="Path to the legacy metadata JSON file.")
    parser.add_argument("cedar_template_iri", help="IRI of the CEDAR template to migrate to.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    configure_logging(args.debug)

    logger.debug("Legacy metadata file: %s", args.legacy_metadata)
    logger.debug("CEDAR template IRI: %s", args.cedar_template_iri)

    with open(args.legacy_metadata) as f:
        legacy_metadata = json.load(f)

    user_message = (
        f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
        f"CEDAR Template IRI: {args.cedar_template_iri}\n\n"
        f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
    )

    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    final_text = _extract_final_response(result["messages"])
    logger.debug("Raw agent response:\n%s", final_text)

    # Build a strict output schema from the CEDAR template
    template_dict = get_cedar_template.invoke({"template_id": args.cedar_template_iri})
    output_model = build_output_model(template_dict)

    # Parse the response into a guaranteed valid JSON object
    metadata = _extract_metadata_json(final_text, output_schema=output_model)
    print(json.dumps(metadata, indent=2))


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
        SystemExit: If no AI message with text content is found.
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
    logger.error("Agent produced no text response.")
    raise SystemExit(1)


def _extract_metadata_json(
    text: str,
    model: str = "gpt-5-mini",
    output_schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Extract a valid JSON metadata object from the agent's free-text response.

    Uses a JSON-mode LLM call to reliably parse the JSON object,
    even when the response contains extra prose or markdown fencing.

    Args:
        text: The raw text response from the migration agent.
        model: The OpenAI model identifier to use for extraction.
        output_schema: Optional Pydantic model to enforce structured output.
            When provided, uses ``json_schema`` response format for strict enforcement.
            When ``None``, falls back to ``json_object`` mode.

    Returns:
        The extracted metadata as a Python dict.
    """
    if output_schema is not None:
        model_kwargs: dict[str, Any] = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cedar_metadata",
                    "strict": True,
                    "schema": output_schema.model_json_schema(),
                },
            },
        }
    else:
        model_kwargs = {"response_format": {"type": "json_object"}}

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        model_kwargs=model_kwargs,
    )
    result = llm.invoke(
        f"Extract the JSON metadata object from the following text. "
        f"Return only the JSON object, nothing else. "
        f"Use null (not empty strings) for any field whose value is unknown, "
        f"missing, or empty.\n\n{text}"
    )
    return json.loads(result.content)


if __name__ == "__main__":
    main()
