"""Logging configuration and utilities for the metadata standardization agent."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import sys
from typing import Any


def configure_logging(debug: bool = False) -> None:
    """Configure logging for the agent.

    Args:
        debug: When True, sets the ``arms_agent`` and key
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
        for name in ("arms_agent", "langchain", "langgraph"):
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


#: What :meth:`~arms_agent.cache.SqliteCache.get` adds to a cache hit.  Logged here, then
#: dropped, so a cached answer reaches the model exactly as the fresh one did.
_CACHE_METADATA = ("_cached", "_cache_age_seconds")


def _log_result(logger: logging.Logger, name: str, result: Any) -> Any:
    """Log a tool's *result* and return it as the model should see it.

    A cache hit is logged with its age, then returned without the cache's own
    bookkeeping keys: the model must not be able to tell a cached answer from a fresh
    one, or two runs of the same record would see different tool output.
    """
    if isinstance(result, dict) and result.get("_cached"):
        logger.debug("%s cache hit (age=%.1fs): %s", name, result.get("_cache_age_seconds", 0), _summarize(result))
        return {key: value for key, value in result.items() if key not in _CACHE_METADATA}
    logger.debug("%s returned: %s", name, _summarize(result))
    return result


def log_tool_call(func):
    """Decorator that logs tool function calls, results, and exceptions."""
    _logger = logging.getLogger(func.__module__)

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger.debug("Calling %s with %s", func.__name__, kwargs)
            try:
                result = await func(*args, **kwargs)
            except Exception:
                _logger.exception("Exception in %s", func.__name__)
                raise
            return _log_result(_logger, func.__name__, result)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _logger.debug("Calling %s with %s", func.__name__, kwargs)
        try:
            result = func(*args, **kwargs)
        except Exception:
            _logger.exception("Exception in %s", func.__name__)
            raise
        return _log_result(_logger, func.__name__, result)

    return wrapper
