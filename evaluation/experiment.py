"""Experiment workflow factory for the evaluation framework."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_experiment_workflow(model: str) -> CompiledStateGraph:
    """Build the experiment workflow by delegating to the shipped ``build_workflow``.

    This gives the evaluation framework a parallel factory alongside
    ``build_baseline_workflow``, while ``build_workflow`` stays in
    ``src/`` for the shipped CLI.

    Args:
        model: LLM model identifier forwarded to ``build_workflow``.

    Returns:
        A compiled LangGraph produced by ``metadata_migration_agent.workflow.build_workflow``.
    """
    from metadata_migration_agent.workflow import build_workflow

    return build_workflow(model=model)


def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    """Build the user prompt for the experiment workflow.

    The experiment prompt includes the CEDAR template IRI and the legacy
    metadata, asking the agent to migrate the record.
    """
    return (
        f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
        f"CEDAR Template IRI: {template_iri}\n\n"
        f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
    )
