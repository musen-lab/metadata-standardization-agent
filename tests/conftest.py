"""Shared pytest configuration for the test suite.

The analysis modules under ``evaluation/`` (``metrics``, ``data_analysis``,
``plots``, ``significance``, ``error_causes``) import each other with bare module
names (e.g. ``from metrics import ...``), matching how they are used from the demo
notebook and CLI.  Putting the ``evaluation/`` directory on ``sys.path`` lets the
tests import those modules with the same bare names.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"
if str(_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATION_DIR))
