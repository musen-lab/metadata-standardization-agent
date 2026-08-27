"""Shared pytest configuration for the agent's test suite.

This file deliberately repeats the environment fixture in the harness's ``tests/conftest.py``
rather than importing it.  The package has to be testable on its own, from a checkout of
``arms-agent/`` alone, so nothing here may reach above this directory.

There is no ``__init__.py`` beside this file, and there must not be.  The harness's
``tests/`` is a package, and a second one by the same name would shadow it on ``sys.path``.
"""

from __future__ import annotations

import pytest

# Every variable the agent reads from the environment.  ``__main__`` calls ``load_dotenv``
# at import time, and pytest imports every test module during collection, so importing the
# CLI test puts a developer's real ``.env`` into ``os.environ`` for the whole session --
# including the tests that never touch the CLI.
_AGENT_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_EXTRACTION_MODEL",
    "OPENAI_COST_MULTIPLIER",
    "OPENAI_COST_CACHE_DISCOUNT",
    "CEDAR_API_KEY",
    "BIOPORTAL_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_TRACING_ENABLED",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "ARMS_CACHE_DIR",
    "ARMS_CACHE_TTL_SECONDS",
)


@pytest.fixture(autouse=True)
def _unconfigured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an environment no ``.env`` has configured.

    What the suite measures has to be the code, not the machine it runs on.  Without
    this, a developer billed at half list price by a gateway sees the pricing tests
    fail on rates they never set, and one with Langfuse keys exercises the tracing-on
    path where CI exercises the tracing-off one.

    A test needing a variable sets it itself, which still works: this runs first, and
    ``monkeypatch`` restores the real environment afterwards either way.
    """
    for name in _AGENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
