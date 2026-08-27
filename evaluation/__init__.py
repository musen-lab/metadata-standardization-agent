"""Evaluation framework for the metadata standardization agent."""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)
