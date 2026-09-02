"""Metadata Standardization Agent: A LangGraph agent for migrating legacy metadata to CEDAR template format."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # pyproject.toml is the only place the version is written.  Reading it back
    # from the installed distribution keeps a second copy from drifting.
    __version__ = version("arms-agent")
except PackageNotFoundError:
    # A source tree that was never installed still has to import.
    __version__ = "0+unknown"
