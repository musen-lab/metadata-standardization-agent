"""The experimental conditions, grouped by how each reaches the template.

:mod:`conditions.prompt_only` holds the conditions handed the template in their prompt,
together with the single LLM call they share and the material their prompts are built
from.  :mod:`conditions.agent_tool` holds ARMS itself, which fetches the template and
looks up terms through tools instead.

Neither this module nor any other holds a list of what the conditions are.  A module
under either package that declares a ``CONDITION`` is one, and :mod:`conditions.registry`
finds it -- so a condition is added by dropping a file in, and removed by taking it out.
:mod:`conditions.registry` documents what such a module declares.

:func:`build_condition` turns a name into the two builders it stands for.  The CLI and
the notebook sweep both dispatch through it, so neither can drift from the other on what
a condition is made of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from conditions.registry import (
    Condition,
    build_condition,
    condition_names,
    discover,
    get_condition,
)

if TYPE_CHECKING:
    CONDITIONS: tuple[str, ...]

__all__ = [
    "CONDITIONS",
    "Condition",
    "build_condition",
    "condition_names",
    "discover",
    "get_condition",
]


def __getattr__(name: str) -> Any:
    """Answer ``CONDITIONS`` by scanning, so importing this package scans nothing.

    The names cannot be known without importing the condition modules, and those import
    the agent.  Deferring the scan to the first read keeps a module free to import a
    sibling condition's helpers without the package importing it back.
    """
    if name == "CONDITIONS":
        return condition_names()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
