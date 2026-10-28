"""Small synthetic corpus fixtures; no service credentials or real agent calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture(autouse=True)
def unconfigured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore developer credentials, including dotenv loaded by other test collectors."""
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_TRACING_ENABLED",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "CEDAR_API_KEY",
        "BIOPORTAL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Callable[..., None]]:
    """Provide a minimal corpus and a filename-based pair writer."""
    for directory in ("rnaseq/input", "rnaseq/gold", "schemas"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "schemas/rnaseq.json").write_text(
        json.dumps({"children": [{"name": name, "permissible_values": [{"type": "ontology"}]} for name in ("a", "c")]})
    )

    def pair(name: str, legacy: Any = None, gold: Any = None) -> None:
        (tmp_path / "rnaseq/input" / name).write_text(json.dumps({"legacy": name} if legacy is None else legacy))
        (tmp_path / "rnaseq/gold" / name).write_text(json.dumps({"a": "X"} if gold is None else gold))

    return tmp_path, pair
