"""The ``term-tool`` condition: the baseline's information, plus the term search tools.

Its user message is the baseline's, byte for byte: the field names, and the vocabulary
each constrained field draws from.  It has no template tool, so it never sees the
specification.  What it can do that the baseline cannot is search the vocabularies.

That message names a vocabulary but no branch within it, so only one of the two search
tools is usable as given.  Both are offered anyway, as ARMS is offered both, and the
system prompt tells the model what each needs; which one it calls is then its own choice
rather than something this condition decided for it.  Against ``arms-agent`` it measures
what the template specification adds; against ``baseline``, what the term search adds.
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
from arms_agent.tools import term_search_from_branch, term_search_from_ontology
from arms_agent.workflow import build_workflow
from conditions.ablation.prompts.term_tool import SYSTEM_PROMPT
from conditions.prompt_only.baseline import build_user_prompt
from conditions.registry import Condition

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

#: The two tools this arm has: ARMS's searches, without its template fetch.
TOOLS = (term_search_from_branch, term_search_from_ontology)


def build_term_tool_workflow(
    model: str,
    template_iri: str | None = None,
    reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    sampling: Sampling = DEFAULT_SAMPLING,
) -> CompiledStateGraph:
    """Build this condition's workflow: the ReAct agent with the term search tools only.

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


#: What the harness runs this module as.  Its searches reach BioPortal, so it declares
#: the key, as ARMS does.
CONDITION = Condition(
    name="term-tool",
    build_workflow=build_term_tool_workflow,
    build_user_prompt=build_user_prompt,
    requires_keys=("BIOPORTAL_API_KEY",),
    order=60,
)
