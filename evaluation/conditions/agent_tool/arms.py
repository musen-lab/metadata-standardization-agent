"""The ``agent-tool`` condition: ARMS itself, which reaches the template and the
vocabularies through tools rather than being handed them in the prompt.

The only arm built with a non-empty tool list, and the only one whose system prompt is
the shipped one, in :mod:`arms_agent.prompts`.  Everything else --
the model, the schema its answer is bound to, the graph it runs in -- it shares with the
prompt-only condition.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from arms_agent.agent import build_migration_agent, build_response_format
from arms_agent.prompts import SYSTEM_PROMPT
from arms_agent.tools import all_tools
from arms_agent.workflow import build_workflow

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_agent_tool_workflow(model: str, template_iri: str | None = None) -> CompiledStateGraph:
    """Build this condition's workflow: the shipped ReAct agent followed by extraction.

    Args:
        model: LLM model identifier forwarded to ``build_migration_agent``.
        template_iri: The CEDAR template the sweep targets.  When given, the agent's
            answer is constrained to it.

    Returns:
        A compiled LangGraph produced by ``arms_agent.workflow.build_workflow``.
    """
    return build_workflow(
        build_migration_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            response_format=build_response_format(template_iri) if template_iri else None,
            tools=all_tools,
            reasoning_effort="high",
            reasoning_mode="standard",
        )
    )


def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    """Build the user prompt: the CEDAR template IRI and the legacy record.

    The IRI rather than the template itself: fetching it is the agent's first tool call.
    """
    return (
        f"Migrate the following legacy metadata record to follow the format of the metadata template.\n\n"
        f"Metadata template IRI: {template_iri}\n\n"
        f"Legacy metadata record:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
    )
