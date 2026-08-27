"""Extraction utilities for post-processing agent output."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_openai import ChatOpenAI

    from arms_agent.state import AgentState

from arms_agent.schema import build_output_model
from arms_agent.tools import get_cedar_template
from arms_agent.tracing import record_migrated_record, record_processing_log

logger = logging.getLogger(__name__)

_extraction_llm: ChatOpenAI | None = None


def _fenced_block_re(marker: str) -> re.Pattern[str]:
    """Return a pattern matching a ```json <marker> ... ``` block."""
    return re.compile(rf"```json[ \t]+{re.escape(marker)}[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.DOTALL)


_RECORD_BLOCK_RE = _fenced_block_re("record")
_LOG_BLOCK_RE = _fenced_block_re("log")


def _parse_fenced_json(text: str, pattern: re.Pattern[str], label: str) -> Any | None:
    """Return the JSON parsed from the last block matching *pattern*, or ``None``.

    The last match wins so that a block the agent restated or corrected later in
    its message supersedes an earlier one.  A missing block and an unparseable one
    are both reported as ``None``, distinguished only in the log, since neither is
    worth failing a migration over.
    """
    matches = pattern.findall(text)
    if not matches:
        logger.debug("No ```json %s block found in agent response", label)
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        logger.warning("Found a ```json %s block but could not parse it as JSON", label)
        return None


def _coerce_record(parsed: Any, output_model: type[BaseModel]) -> dict[str, Any] | None:
    """Validate *parsed* against *output_model*, filling absent fields with null.

    Returns the validated dict, or ``None`` when *parsed* is not a JSON object or
    does not satisfy the template's schema — in which case the caller falls back
    to the LLM extraction path.
    """
    if not isinstance(parsed, dict):
        logger.warning("Record block is a %s, not a JSON object", type(parsed).__name__)
        return None
    filled = {name: parsed.get(name) for name in output_model.model_fields}
    extra = set(parsed) - set(output_model.model_fields)
    if extra:
        logger.warning("Record block has %d field(s) absent from the template: %s", len(extra), sorted(extra))
        return None
    try:
        return output_model.model_validate(filled).model_dump()
    except ValueError as exc:
        logger.warning("Record block failed template validation: %s", exc)
        return None


def _extract_log(final_text: str) -> list[dict[str, Any]]:
    """Return the processing-log entries from the agent's response.

    Non-object entries are dropped.  An absent or unparseable block yields an
    empty list: the log is diagnostic, so losing it must not fail a migration.
    """
    parsed = _parse_fenced_json(final_text, _LOG_BLOCK_RE, "log")
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        logger.warning("Processing log is a %s, not an array", type(parsed).__name__)
        return []
    entries = [entry for entry in parsed if isinstance(entry, dict)]
    if len(entries) != len(parsed):
        logger.warning("Dropped %d processing-log entries that were not objects", len(parsed) - len(entries))
    return entries


def _get_extraction_llm() -> ChatOpenAI:
    """Return a shared extraction LLM client, creating it on first use.

    Built against the same endpoint as the migration agent, since a gateway that serves
    one serves the other.  The model is overridable because an endpoint need not offer
    this one: the Stanford AI API Gateway, for instance, lists ``gpt-4.1`` but no
    ``gpt-4.1-mini``, and an unavailable model here would only surface on the fallback
    path, long after the run started.
    """
    global _extraction_llm  # noqa: PLW0603
    if _extraction_llm is None:
        from langchain_openai import ChatOpenAI as _ChatOpenAI

        from arms_agent.agent import resolve_base_url

        model = os.environ.get("OPENAI_EXTRACTION_MODEL", "").strip() or "gpt-4.1-mini"
        logger.debug("Creating the extraction client with model=%s", model)
        _extraction_llm = _ChatOpenAI(model=model, temperature=0, base_url=resolve_base_url())
    return _extraction_llm


def extract_output_metadata(
    state: AgentState,
    # LangGraph matches this annotation against a fixed set of spellings to decide
    # whether to inject the run config, and `RunnableConfig | None` is not among them:
    # it would leave the parameter at None on every call, silently untraced.
    config: Optional[RunnableConfig] = None,  # noqa: UP045
) -> dict[str, Any]:
    """Post-processing node that extracts the migrated record and processing log.

    When the agent was built with a response format its answer is already a
    validated object, so this reads it straight out of state: no text parsing and
    no second model between the agent and the recorded result.

    Without a response format — the prompt-only evaluation conditions, which answer
    in free text — it falls back to parsing the ```json record and ```json log
    blocks out of the response, and then to an extraction LLM call if the record
    block is missing or does not satisfy the template.

    Args:
        state: The current agent state containing messages and cedar_template_iri.
        config: The run config, forwarded to the extraction LLM so that a fallback
            call is counted and traced like the rest of the run.

    Returns:
        A partial state update with ``metadata`` and ``decisions`` populated.
    """
    structured = state.get("structured_response")
    if structured is not None:
        return _from_structured_response(structured)
    return _from_response_text(state, config)


def _from_structured_response(structured: Any) -> dict[str, Any]:
    """Split a validated agent answer into the record and the processing log."""
    payload = structured.model_dump() if isinstance(structured, BaseModel) else dict(structured)
    metadata = payload.get("record") or {}
    decisions = payload.get("log") or []
    logger.debug("Read a validated response with %d field(s) and %d log entr(ies)", len(metadata), len(decisions))
    record_migrated_record(metadata, source="structured_response")
    record_processing_log(decisions, source="structured_response")
    return {"metadata": metadata, "decisions": decisions}


def _from_response_text(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Recover the record and log from an agent that answered in free text."""
    final_text = extract_agent_final_response(state["messages"])
    logger.debug("Raw agent response:\n%s", final_text)

    template_dict = get_cedar_template.invoke({"template_id": state["cedar_template_iri"]})
    output_model = build_output_model(template_dict)

    decisions = _extract_log(final_text)
    logger.debug("Parsed %d processing-log entries", len(decisions))

    parsed_record = _parse_fenced_json(final_text, _RECORD_BLOCK_RE, "record")
    if parsed_record is not None:
        metadata = _coerce_record(parsed_record, output_model)
        if metadata is not None:
            logger.debug("Used the record block directly; skipped the extraction call")
            record_migrated_record(metadata, source="record_block")
            record_processing_log(decisions, source="record_block")
            return {"metadata": metadata, "decisions": decisions}

    # Loud, not debug: past this point the recorded record is a second model's
    # reconstruction rather than the agent's own answer, which is worth knowing when
    # reading a run's results.
    logger.warning("No usable record block; reconstructing the record with the extraction LLM")
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
        # Without the run config this call is invisible to the token tracker and to
        # Langfuse, so its cost and content go unrecorded.
        config=config,
        **model_kwargs,
    )
    metadata = json.loads(result.content)
    logger.debug("Extracted metadata with %d top-level keys", len(metadata))
    record_migrated_record(metadata, source="extraction_llm")
    record_processing_log(decisions, source="extraction_llm")
    return {"metadata": metadata, "decisions": decisions}


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
