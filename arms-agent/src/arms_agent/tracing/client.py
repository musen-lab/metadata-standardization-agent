"""Whether to reach Langfuse at all, how to reach it, and making sure it arrives.

Every other module in this package asks here first, so the rule that tracing is
optional is stated once rather than repeated at each call site.  The Langfuse import
stays inside the functions: a project without the credentials never pays for it.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from importlib.util import find_spec
from typing import Any

logger = logging.getLogger(__name__)


@cache
def _langfuse_installed() -> bool:
    """Return True when the Langfuse SDK is importable.

    Langfuse ships in the ``tracing`` extra, so credentials alone do not mean the
    SDK is there.  Checking here keeps the four import sites unguarded and warns
    once, rather than raising ImportError in the middle of a run.
    """
    if find_spec("langfuse") is not None:
        return True
    logger.warning(
        "Langfuse credentials are set but the SDK is not installed, so nothing will be traced; "
        "install it with: pip install 'arms-agent[tracing]'"
    )
    return False


def tracing_enabled() -> bool:
    """Return True when Langfuse is installed, credentialed, and not switched off."""
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return False
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False
    return _langfuse_installed()


def recording_client(what: str) -> Any | None:
    """Return the Langfuse client to record *what* into, or ``None`` when it cannot be recorded.

    An event attaches to whatever span is current in the calling code's context, so
    recording without one would strand it in a trace of its own rather than the run's.
    """
    if not tracing_enabled():
        return None
    from langfuse import get_client

    client = get_client()
    if client.get_current_trace_id() is None:
        logger.warning("No active span, so the %s was not traced; wrap the run in traced_run() to record it", what)
        return None
    return client


def flush_tracing() -> None:
    """Send buffered traces to Langfuse.

    The SDK batches spans in the background, so a short-lived CLI has to flush
    before it exits or the last traces are lost.  Failures are logged and
    swallowed: losing observability must not fail a completed migration.
    """
    if not tracing_enabled():
        return
    from langfuse import get_client

    try:
        get_client().flush()
    except Exception:  # noqa: BLE001 - tracing must never break a completed run
        logger.warning("Failed to flush traces to Langfuse", exc_info=True)
