"""The ``baseline`` condition: field names and vocabulary names, no tools.

The information set of the published baseline. Its system prompt lives in
:mod:`conditions.prompt_only.prompts.baseline`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from arms_agent.agent import build_migration_agent, build_response_format
from arms_agent.workflow import build_workflow
from conditions.prompt_only import template_spec
from conditions.prompt_only.prompts.baseline import SYSTEM_PROMPT
from conditions.registry import Condition

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_baseline_workflow(model: str, template_iri: str | None = None) -> CompiledStateGraph:
    """Build this condition's workflow: single LLM migrate followed by structured extraction.

    Args:
        model: LLM model identifier used for the migration call.
        template_iri: The CEDAR template the sweep targets.  When given, the migration
            call's answer is validated against it, so the extraction node reads an
            object instead of parsing the response text.

    Returns:
        A compiled LangGraph produced by ``arms_agent.workflow.build_workflow``.
    """
    return build_workflow(
        build_migration_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            response_format=build_response_format(template_iri) if template_iri else None,
            tools=(),  # No tools
            reasoning_effort="high",
            reasoning_mode="standard",
        )
    )


def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    """Build the user prompt: the legacy record, the field names, and vocabulary names.

    This is the published baseline's user message and must not drift.

    Args:
        legacy_metadata: The legacy record to migrate.
        template_iri: IRI or ID of the target CEDAR template.
    """
    # Called through the module so tests can stub the fetch.
    template = template_spec.fetch_cedar_template(template_iri)
    field_names = template_spec.collect_field_names(template["children"])
    ontology_lines = template_spec.collect_constraint_lines(template)

    field_list = ", ".join(field_names)
    prompt = (
        f"Given the following legacy metadata: {json.dumps(legacy_metadata, indent=2)}.\n"
        "Report a new and corrected metadata sample where the following "
        f"template is as complete as possible:\n{field_list}.\n"
        "Check if the field values and field names make sense. If no match "
        "is found for a field name, match it to an ontology. As far as possible, "
        "make field values adhere to ontology restrictions.\n"
    )
    if ontology_lines:
        prompt += "\n".join(ontology_lines) + "\n"
    prompt += "- Missing values: use null\n\n"
    prompt += "Do not provide any explanation. Output only the corrected record in Python dict format"
    return prompt


#: What the harness runs this module as.  ``baseline`` needs no key of its own: the
#: template comes from CEDAR, which every condition needs, and the vocabularies are
#: never consulted.
CONDITION = Condition(
    name="baseline",
    build_workflow=build_baseline_workflow,
    build_user_prompt=build_user_prompt,
    order=0,
)
