"""Logging configuration and utilities for the metadata migration agent."""

from __future__ import annotations

import functools
import json
import logging
import sys
from typing import Any


def configure_logging(debug: bool = False) -> None:
    """Configure logging for the agent.

    Args:
        debug: When True, sets the ``metadata_migration_agent`` and key
            LangChain/LangGraph loggers to DEBUG. Otherwise only WARNING
            and above are shown.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    if debug:
        for name in ("metadata_migration_agent", "langchain", "langgraph"):
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)


_MAX_LOG_CHARS = 500


def _summarize(value: Any) -> str:
    """Return a truncated string representation of *value*."""
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > _MAX_LOG_CHARS:
        return text[:_MAX_LOG_CHARS] + "..."
    return text


def log_tool_call(func):
    """Decorator that logs tool function calls, results, and exceptions."""
    _logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _logger.debug("Calling %s with %s", func.__name__, kwargs)
        try:
            result = func(*args, **kwargs)
        except Exception:
            _logger.exception("Exception in %s", func.__name__)
            raise
        if isinstance(result, dict) and result.get("_cached"):
            _logger.debug(
                "%s cache hit (age=%.1fs): %s",
                func.__name__,
                result.get("_cache_age_seconds", 0),
                _summarize(result),
            )
        else:
            _logger.debug("%s returned: %s", func.__name__, _summarize(result))
        return result

    return wrapper
