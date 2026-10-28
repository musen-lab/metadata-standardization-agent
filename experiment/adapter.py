"""A gold-blind adapter to the existing ARMS workflow and user prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from arms_agent.tracing import instrument
from arms_agent.workflow import RECURSION_LIMIT
from conditions.agent_tool.arms import build_agent_tool_workflow, build_user_prompt
from experiment.dataset import TEMPLATE_IRI

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def make_task(model: str) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Build ARMS once; each invocation receives only legacy metadata and template.

    A fresh callback handler per item nests under the SDK's active task span.
    The SDK owns concurrency, failure isolation and the enclosing trace.
    """
    workflow = build_agent_tool_workflow(model=model, template_iri=TEMPLATE_IRI)

    async def task(*, item: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        agent_input = item["input"]
        if set(agent_input) != {"legacy_metadata", "template_iri"} or agent_input["template_iri"] != TEMPLATE_IRI:
            raise ValueError("Expected only RNA-seq legacy_metadata and template_iri")
        prompt = build_user_prompt(agent_input["legacy_metadata"], agent_input["template_iri"])
        result = await workflow.ainvoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "cedar_template_iri": agent_input["template_iri"],
            },
            config=instrument({"recursion_limit": RECURSION_LIMIT, "run_name": "arms-rnaseq-migrate"}),
        )
        output = result.get("metadata")
        if not isinstance(output, dict) or not output:
            raise ValueError("ARMS did not return a nonempty metadata object")
        return output

    return task
