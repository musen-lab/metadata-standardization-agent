"""Shared pytest configuration for the evaluation harness's test suite.

The agent's own tests live in ``arms-agent/tests/`` and have their own ``conftest``, so
the package stays testable on its own.  What is left here tests ``evaluation/``.

The modules under ``evaluation/`` import each other relative to that directory rather
than to the project root (e.g. ``from analysis.metrics import ...``), matching how they
are used from the demo notebook and the CLI.  Putting ``evaluation/`` on ``sys.path``
lets the tests import them by exactly the same names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"
if str(_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATION_DIR))

# Every variable the harness reads from the environment.  The evaluation CLI calls
# ``load_dotenv(override=True)`` at import time, and pytest imports every test module
# during collection, so importing the CLI test puts a developer's real ``.env`` into
# ``os.environ`` for the whole session -- including the tests that never touch a CLI.
_PROJECT_ENV_VARS = (
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
    for name in _PROJECT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
