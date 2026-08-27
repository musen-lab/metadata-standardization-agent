"""Langfuse tracing for agent and evaluation runs.

Tracing is optional.  It is active only when ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are both set and ``LANGFUSE_TRACING_ENABLED`` is not
``false``.  Otherwise every helper here is a no-op, so the agent runs unchanged
for anyone without Langfuse credentials.

Which Langfuse project a trace lands in is decided by the key pair, not by a
per-run setting.  To separate runs inside one project -- evaluation sweeps from
production migrations, say -- set ``LANGFUSE_TRACING_ENVIRONMENT``, which the
Langfuse client reads directly and which the Langfuse UI can filter on.

Four modules, split by how each reaches Langfuse, because they reach it in
genuinely different ways:

:mod:`~arms_agent.tracing.client`
    The credential gate every other module asks first, the client to record into,
    and the flush a short-lived CLI owes its buffered spans.

:mod:`~arms_agent.tracing.instrumentation`
    Hands the callback handler to LangChain, which then traces the graph -- nodes,
    model calls, tool calls -- without the agent code taking part.

:mod:`~arms_agent.tracing.spans`
    Opens one span around a run in the context the run's own code can see, which
    the callback handler cannot do for it.

:mod:`~arms_agent.tracing.observations`
    Records what the run produced: the migrated record and the processing log.
    These need the span above, and are the reason it exists.
"""

from arms_agent.tracing.client import flush_tracing, tracing_enabled
from arms_agent.tracing.instrumentation import instrument, tracing_callbacks
from arms_agent.tracing.observations import (
    MIGRATED_RECORD_OBSERVATION,
    PROCESSING_LOG_OBSERVATION,
    record_migrated_record,
    record_processing_log,
)
from arms_agent.tracing.spans import traced_run

__all__ = [
    "MIGRATED_RECORD_OBSERVATION",
    "PROCESSING_LOG_OBSERVATION",
    "flush_tracing",
    "instrument",
    "record_migrated_record",
    "record_processing_log",
    "traced_run",
    "tracing_callbacks",
    "tracing_enabled",
]
