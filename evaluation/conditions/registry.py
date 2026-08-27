"""How a module becomes a condition the harness can run.

A condition is any module under a family package here -- :mod:`conditions.prompt_only`
or :mod:`conditions.agent_tool` -- that declares a module-level :data:`CONDITION`.  Drop
such a module in and the CLI, the sweep and the notebook all see it; nothing else has to
be edited, because nothing else holds a list of what the conditions are.

A module is asked for four things, and the last two only when it needs them::

    from conditions.registry import Condition

    CONDITION = Condition(
        name="schema+vocab",                    # what the CLI takes and the output dir is called
        build_workflow=build_schema_vocab_workflow,
        build_user_prompt=build_user_prompt,
        requires_keys=("BIOPORTAL_API_KEY",),   # checked before a sweep spends anything
        order=20,                               # where it sits in the reported order
    )

The name is declared rather than taken from the filename because the two need not agree:
``schema+vocab`` is not a legal module name.  ``requires_keys`` is declared rather than
looked up because only the condition knows what it calls; a condition that needs a key
the environment lacks stops the sweep in :func:`sweep.plan_sweep`, before any run starts.

A module without a :data:`CONDITION` is not a condition -- :mod:`prompt_only.template_spec`
is the material the prompts are built from, not an arm of the study -- so discovery passes
over it rather than guessing.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from langgraph.graph.state import CompiledStateGraph

#: The module-level name a condition declares itself with.
DESCRIPTOR = "CONDITION"

_PACKAGE = __package__ or "conditions"
_PACKAGE_PATH = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Condition:
    """One arm of the study: what to build, what it needs, and where it is reported.

    Attributes:
        name: The name the CLI takes, the sweep prints and the output directory carries.
        build_workflow: Takes ``model`` and ``template_iri``, returns a compiled graph.
        build_user_prompt: Takes the legacy record and the template IRI, returns the
            user message.
        requires_keys: Environment variables this condition calls out to, beyond the
            ones every condition needs.  Given as any iterable of names.
        order: Where the condition sits in the reported order; ties break by name.
        family: The package it was found in, filled in by discovery.
        module: Its import path, filled in by discovery.
    """

    name: str
    build_workflow: Callable[..., CompiledStateGraph]
    build_user_prompt: Callable[[dict[str, Any], str], str]
    requires_keys: frozenset[str] = frozenset()
    order: int = 0
    family: str = ""
    module: str = ""

    def __post_init__(self) -> None:
        """Normalise *requires_keys*, so a module may declare it as any iterable."""
        object.__setattr__(self, "requires_keys", frozenset(self.requires_keys))


_registry: dict[str, Condition] | None = None


def discover(*, refresh: bool = False) -> dict[str, Condition]:
    """Return every declared condition, keyed by name, in reported order.

    The scan imports each candidate module, since the descriptor is what says whether a
    module is a condition at all.  That costs nothing worth saving: the conditions share
    their heavy imports, so the second module and the tenth are a few milliseconds each.

    The result is cached.  Pass *refresh* after adding a module to the package in a
    running process -- a notebook kernel, or a test.

    Args:
        refresh: Rescan rather than answer from the cache.

    Returns:
        ``{name: Condition}``, ordered by each condition's ``order`` then its name.

    Raises:
        TypeError: If a module declares a ``CONDITION`` that is not a :class:`Condition`.
        ValueError: If two modules declare the same name.
    """
    global _registry  # noqa: PLW0603
    if _registry is None or refresh:
        found = sorted(_iter_declared(), key=lambda condition: (condition.order, condition.name))
        _registry = {}
        for condition in found:
            if condition.name in _registry:
                first = _registry[condition.name].module
                raise ValueError(
                    f"Two modules declare the condition {condition.name!r}: {first} and {condition.module}"
                )
            _registry[condition.name] = condition
    return _registry


def _iter_declared() -> Iterator[Condition]:
    """Yield the condition every module under every family package declares."""
    for family in _families():
        package = importlib.import_module(f"{_PACKAGE}.{family}")
        for info in pkgutil.iter_modules(package.__path__):
            if info.ispkg or info.name.startswith("_"):
                continue
            path = f"{_PACKAGE}.{family}.{info.name}"
            declared = getattr(importlib.import_module(path), DESCRIPTOR, None)
            if declared is None:
                continue
            if not isinstance(declared, Condition):
                raise TypeError(f"{path}.{DESCRIPTOR} is {type(declared).__name__}, not a Condition")
            yield replace(declared, family=family, module=path)


def _families() -> list[str]:
    """The family packages to scan, by directory name.

    Every package here is a family, so a third one is added the same way a condition is:
    by dropping it in.  Private names are skipped, which is what keeps ``__pycache__``
    out.
    """
    return sorted(
        info.name for info in pkgutil.iter_modules([str(_PACKAGE_PATH)]) if info.ispkg and not info.name.startswith("_")
    )


def condition_names() -> tuple[str, ...]:
    """Every condition name the experiment knows, in reported order."""
    return tuple(discover())


def get_condition(name: str) -> Condition:
    """Return the condition called *name*.

    Args:
        name: One of :func:`condition_names`.

    Returns:
        The declared :class:`Condition`.

    Raises:
        ValueError: If no module declares that name.  A name outside the set is a typo,
            and a typo left to fall through to an agent would be an expensive one.
    """
    registry = discover()
    if name not in registry:
        raise ValueError(f"Unknown run type {name!r}; expected one of {', '.join(registry)}")
    return registry[name]


def build_condition(
    run_type: str,
) -> tuple[Callable[..., CompiledStateGraph], Callable[[dict[str, Any], str], str]]:
    """Return the workflow builder and the user-prompt builder *run_type* is run with.

    Args:
        run_type: One of :func:`condition_names`.

    Returns:
        ``(build_workflow, build_user_prompt)``.  The workflow builder takes ``model``
        and ``template_iri``; the prompt builder takes the legacy record and that same
        template IRI.

    Raises:
        ValueError: If *run_type* is not a declared condition.
    """
    condition = get_condition(run_type)
    return condition.build_workflow, condition.build_user_prompt
