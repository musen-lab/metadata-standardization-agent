"""Attach Langfuse to a LangChain run.

The callback handler traces the graph itself -- every node, model call and tool call --
without the agent code knowing.  What it cannot do is name the enclosing trace or
record from inside a node body; those are :mod:`spans` and :mod:`observations`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arms_agent.tracing.client import tracing_enabled

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler


def tracing_callbacks() -> list[BaseCallbackHandler]:
    """Return the Langfuse callback handlers to attach to a run, or an empty list when tracing is off."""
    if not tracing_enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def instrument(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a LangChain run *config* with Langfuse tracing attached.

    The Langfuse callback handler is appended to ``callbacks``, preserving any
    handler the caller already registered.  ``run_name`` names the root
    observation on its own, but the enclosing trace is named only from the
    reserved ``langfuse_trace_name`` metadata key, so ``run_name`` is mirrored
    there to keep traces titled the way the runs are.  ``tags`` and ``metadata``
    are read by the handler as-is and need no translation.

    Returns *config* unchanged when tracing is off.
    """
    handlers = tracing_callbacks()
    if not handlers:
        return config

    traced = dict(config)
    traced["callbacks"] = [*(traced.get("callbacks") or []), *handlers]

    run_name = traced.get("run_name")
    if run_name:
        metadata = dict(traced.get("metadata") or {})
        metadata.setdefault("langfuse_trace_name", run_name)
        traced["metadata"] = metadata

    return traced
