"""Local-data Langfuse experiments for the unpublished ARMS evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

# Existing evaluation modules import their siblings by directory, not package.
EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
