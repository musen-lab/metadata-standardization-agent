"""LangGraph agent for metadata migration."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

from arms_agent.schema import build_response_model
from arms_agent.state import AgentState
from arms_agent.tools import get_cedar_template

logger = logging.getLogger(__name__)

# OpenAI o-series reasoning models don't support parallel_tool_calls
_O_SERIES = re.compile(r"^o\d")

# Which models take reasoning settings.  Not gpt-5.6 alone: the o-series and the whole
# gpt-5 family accept them, and a non-reasoning model such as gpt-4.1 rejects the
# request outright with "'reasoning.effort' is not supported with this model".
_REASONING_MODELS = re.compile(r"^(o\d|gpt-5)")

# The values the API accepts, as it reports them when given anything else.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
ReasoningMode = Literal["standard", "pro"]

# Where the OpenAI-compatible API lives.  ``OPENAI_BASE_URL`` is the OpenAI SDK's own
# name for this, so a gateway configured for any other OpenAI client works here
# unchanged; ``OPENAI_API_BASE`` is accepted as well because langchain reads that one.
_BASE_URL_VARS = ("OPENAI_BASE_URL", "OPENAI_API_BASE")


def resolve_base_url() -> str | None:
    """Return the API endpoint to call, or ``None`` for OpenAI's own.

    Set one of the variables in :data:`_BASE_URL_VARS` to route every call through a
    gateway, such as the Stanford API Gateway, instead of ``api.openai.com``.  The key
    in ``OPENAI_API_KEY`` then has to be the one that gateway issues.
    """
    for name in _BASE_URL_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _reasoning_kwargs(
    model: str,
    reasoning_effort: ReasoningEffort,
    reasoning_mode: ReasoningMode,
) -> dict[str, Any]:
    """Return the reasoning settings to hand ``ChatOpenAI``.

    A model that does not reason gets nothing, since it rejects the request outright.
    """
    if not _REASONING_MODELS.match(model):
        return {}
    return {"reasoning": {"effort": reasoning_effort, "mode": reasoning_mode}}


def build_response_format(template_iri: str) -> ProviderStrategy:
    """Return the provider-enforced response format for *template_iri*.

    Raises:
        ValueError: If the template cannot be fetched, which would otherwise bind an
            empty schema and silently produce empty records for a whole run.
    """
    template_dict = get_cedar_template.invoke({"template_id": template_iri})
    if "error" in template_dict or not template_dict.get("children"):
        msg = f"Cannot build a response schema for {template_iri}: {template_dict.get('error', 'no fields returned')}"
        raise ValueError(msg)
    return ProviderStrategy(build_response_model(template_dict), strict=True)


def build_migration_agent(
    model: str,
    system_prompt: str,
    response_format: ProviderStrategy | None,
    tools: Sequence[BaseTool],
    reasoning_effort: ReasoningEffort = "low",
    reasoning_mode: ReasoningMode = "standard",
) -> CompiledStateGraph:
    """Build the agent that performs the migration.

    Args:
        model: The OpenAI model identifier to use.
        system_prompt: The system prompt carrying the agent's policy.
        response_format: The schema the final answer is bound to.
            ``None`` leaves the answer unconstrained, as free text the extraction node then has to parse.
        tools: The tools this agent may call.
        reasoning_effort: How much reasoning the model spends before answering.
            Ignored by models that do not reason.
        reasoning_mode: Which reasoning behaviour to use.  Ignored by models that
            do not reason.

    Returns:
        A compiled agent graph.
    """
    from langchain_openai import ChatOpenAI

    base_url = resolve_base_url()
    reasoning_kwargs = _reasoning_kwargs(model, reasoning_effort, reasoning_mode)

    logger.info(
        "Building migration agent with model=%s, tools=%d, structured_output=%s, reasoning=%s, endpoint=%s",
        model,
        len(tools),
        response_format is not None,
        reasoning_kwargs or None,
        base_url or "api.openai.com",
    )
    model_kwargs: dict[str, Any] = {}
    if tools and not _O_SERIES.match(model):
        model_kwargs["parallel_tool_calls"] = True
    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        temperature=0,
        model_kwargs=model_kwargs,
        **reasoning_kwargs,
    )
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        state_schema=AgentState,
        response_format=response_format,
    )
