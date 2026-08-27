"""One span around a whole run, opened where the run's own code can see it."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from arms_agent.tracing.client import tracing_enabled

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def traced_run(name: str, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
    """Open one Langfuse span around a run, yielding it (or ``None`` when tracing is off).

    The LangChain callback handler nests its own spans correctly, but it attaches the
    tracing context in the callback rather than in the code the graph is running.
    Under ``ainvoke`` those are different tasks, so a node body asking Langfuse
    "which trace am I in?" gets nothing.  Wrapping the invocation here puts a span in
    the context the graph inherits, which is what lets a node record into the trace.

    Concurrent runs each need their own ``traced_run``, entered inside their own task,
    so that the context one run attaches is not visible to the others.
    """
    if not tracing_enabled():
        yield None
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(name=name, metadata=metadata) as span:
        yield span
