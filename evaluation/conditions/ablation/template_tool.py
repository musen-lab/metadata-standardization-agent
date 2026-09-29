"""The ``template-tool`` condition: ARMS with the template tool and no term search.

It is handed what ARMS is handed, the template IRI and the legacy record, and fetches the
template the same way.  It cannot query a vocabulary, so an ontology-constrained value
comes from the model's own knowledge.  Against ``arms-agent`` it measures what the term
search adds; against ``baseline``, what the full template specification adds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arms_agent.agent import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SAMPLING,
    ReasoningEffort,
    Sampling,
    build_migration_agent,
    build_response_format,
)
from arms_agent.tools import get_cedar_template
from arms_agent.workflow import build_workflow
from conditions.ablation.prompts.template_tool import SYSTEM_PROMPT
from conditions.agent_tool.arms import build_user_prompt
from conditions.registry import Condition

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

#: The one tool this arm has.
TOOLS = (get_cedar_template,)


def build_template_tool_workflow(
    model: str,
    template_iri: str | None = None,
    reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    sampling: Sampling = DEFAULT_SAMPLING,
) -> CompiledStateGraph:
    """Build this condition's workflow: the ReAct agent with the template tool only.

    Args:
        model: LLM model identifier forwarded to ``build_migration_agent``.
        template_iri: The CEDAR template the sweep targets.  When given, the agent's
            answer is constrained to it.
        reasoning_effort: How much the model reasons before answering; a model on another
            server may accept other levels than OpenAI's.
        sampling: How the model picks each token (default: greedy, temperature 0).

    Returns:
        A compiled LangGraph produced by ``arms_agent.workflow.build_workflow``.
    """
    return build_workflow(
        build_migration_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            response_format=build_response_format(template_iri) if template_iri else None,
            tools=TOOLS,
            reasoning_effort=reasoning_effort,
            reasoning_mode="standard",
            sampling=sampling,
        )
    )


#: What the harness runs this module as.  The template comes from CEDAR, which every
#: condition needs, and no vocabulary is consulted, so it declares no key of its own.
CONDITION = Condition(
    name="template-tool",
    build_workflow=build_template_tool_workflow,
    build_user_prompt=build_user_prompt,
    order=40,
)
