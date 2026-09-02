"""The version has one source: pyproject.toml, read back through the distribution."""

from __future__ import annotations

import importlib
import importlib.metadata
import tomllib
from pathlib import Path

import pytest

import arms_agent


def test_version_matches_the_installed_distribution() -> None:
    """__version__ reports whatever the installed distribution declares."""
    assert arms_agent.__version__ == importlib.metadata.version("arms-agent")


def test_version_matches_pyproject() -> None:
    """The installed version agrees with pyproject.toml, so nothing has drifted."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("no pyproject.toml beside the tests")
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert arms_agent.__version__ == declared


def test_version_falls_back_when_the_package_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing from an uninstalled source tree gives a placeholder, not an error."""

    def not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_found)
    try:
        assert importlib.reload(arms_agent).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(arms_agent)
