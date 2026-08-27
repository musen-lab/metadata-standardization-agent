"""Tests that a condition is added by dropping a module in, and nothing else.

The registry's whole promise is that no list of conditions is written down anywhere, so
these tests write real modules into the package on disk and scan for them.  A stub would
test the descriptor and not the promise: what has to hold is that a file appearing under
``conditions/`` is found, run and reported like the ones already there.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from conditions import registry
from conditions.registry import Condition, build_condition, condition_names, discover, get_condition

if TYPE_CHECKING:
    from collections.abc import Iterator

_FAMILIES = Path(registry.__file__).resolve().parent

# A whole condition module, of the shape a real one has: builders, then the declaration.
_MODULE = textwrap.dedent(
    '''
    """A condition written by a test."""

    from __future__ import annotations

    from typing import Any

    from conditions.registry import Condition


    def build_workflow(model: str, template_iri: str | None = None) -> str:
        """Stand in for a compiled graph, which needs no LLM to be identified."""
        return f"{model}|{template_iri}"


    def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str:
        """Stand in for the user message."""
        return f"{sorted(legacy_metadata)}|{template_iri}"


    CONDITION = Condition(
        name="{name}",
        build_workflow=build_workflow,
        build_user_prompt=build_user_prompt,
        requires_keys={requires_keys!r},
        order={order},
    )
    '''
)


@pytest.fixture
def drop_in() -> Iterator[object]:
    """Write condition modules into the package, and take them out again afterwards.

    The registry is rescanned on the way in and on the way out, so a test that adds a
    module cannot leave it in the cache for the next one.
    """
    written: list[Path] = []

    def _drop(family: str, module: str, *, name: str, requires_keys: tuple[str, ...] = (), order: int = 50) -> None:
        path = _FAMILIES / family / f"{module}.py"
        assert not path.exists(), f"{path} already exists; pick another module name"
        source = _MODULE.replace("{name}", name).replace("{requires_keys!r}", repr(requires_keys))
        path.write_text(source.replace("{order}", str(order)))
        written.append(path)
        importlib.invalidate_caches()
        discover(refresh=True)

    yield _drop

    for path in written:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"conditions.{path.parent.name}.{path.stem}", None)
    importlib.invalidate_caches()
    discover(refresh=True)


class TestWhatIsAlreadyThere:
    def test_both_shipped_conditions_are_found(self) -> None:
        assert condition_names() == ("baseline", "arms-agent")

    def test_each_carries_the_family_it_was_found_in(self) -> None:
        assert get_condition("baseline").family == "prompt_only"
        assert get_condition("arms-agent").family == "agent_tool"

    def test_only_arms_declares_a_vocabulary_key(self) -> None:
        """The key check in plan_sweep reads this, so a wrong answer costs a failed run."""
        assert get_condition("arms-agent").requires_keys == {"BIOPORTAL_API_KEY"}
        assert get_condition("baseline").requires_keys == frozenset()

    def test_a_helper_module_is_not_a_condition(self) -> None:
        """template_spec sits beside the conditions and declares nothing, so it is passed over."""
        assert "template_spec" not in condition_names()
        assert all(condition.module != "conditions.prompt_only.template_spec" for condition in discover().values())

    def test_build_condition_returns_the_declared_builders(self) -> None:
        build_workflow, build_user_prompt = build_condition("baseline")
        condition = get_condition("baseline")
        assert build_workflow is condition.build_workflow
        assert build_user_prompt is condition.build_user_prompt


class TestDroppingOneIn:
    def test_a_new_prompt_only_module_is_found(self, drop_in: object) -> None:
        drop_in("prompt_only", "dropped_in", name="dropped-in")
        assert "dropped-in" in condition_names()
        assert get_condition("dropped-in").family == "prompt_only"

    def test_a_new_agent_tool_module_is_found(self, drop_in: object) -> None:
        drop_in("agent_tool", "dropped_in_agent", name="dropped-in-agent")
        assert get_condition("dropped-in-agent").family == "agent_tool"

    def test_its_name_need_not_be_its_module_name(self, drop_in: object) -> None:
        """``schema+vocab`` is the name that made this necessary: no module may be called it."""
        drop_in("prompt_only", "schema_vocab_like", name="schema+vocab")
        assert "schema+vocab" in condition_names()
        assert get_condition("schema+vocab").module == "conditions.prompt_only.schema_vocab_like"

    def test_its_builders_are_the_ones_that_run(self, drop_in: object) -> None:
        drop_in("prompt_only", "dropped_in", name="dropped-in")
        build_workflow, build_user_prompt = build_condition("dropped-in")
        assert build_workflow(model="gpt-4.1-mini", template_iri="iri") == "gpt-4.1-mini|iri"
        assert build_user_prompt({"a": 1}, "iri") == "['a']|iri"

    def test_order_decides_where_it_is_reported(self, drop_in: object) -> None:
        """Reported order is the module's to declare, since the tables read left to right."""
        drop_in("prompt_only", "dropped_in", name="dropped-in", order=10)
        assert condition_names() == ("baseline", "dropped-in", "arms-agent")

    def test_taking_it_out_again_removes_it(self, drop_in: object) -> None:
        drop_in("prompt_only", "dropped_in", name="dropped-in")
        (_FAMILIES / "prompt_only" / "dropped_in.py").unlink()
        discover(refresh=True)
        assert "dropped-in" not in condition_names()


class TestRefusals:
    def test_two_modules_may_not_claim_one_name(self, drop_in: object) -> None:
        drop_in("prompt_only", "dropped_in", name="dropped-in")
        with pytest.raises(ValueError, match="Two modules declare the condition 'dropped-in'"):
            drop_in("agent_tool", "dropped_in_agent", name="dropped-in")

    def test_a_descriptor_of_the_wrong_type_is_refused(self) -> None:
        """A module that declares something else has made a mistake, not a condition."""
        path = _FAMILIES / "prompt_only" / "not_a_condition.py"
        path.write_text('"""Wrong."""\n\nfrom __future__ import annotations\n\nCONDITION = "baseline"\n')
        try:
            importlib.invalidate_caches()
            with pytest.raises(TypeError, match="is str, not a Condition"):
                discover(refresh=True)
        finally:
            path.unlink()
            sys.modules.pop("conditions.prompt_only.not_a_condition", None)
            importlib.invalidate_caches()
            discover(refresh=True)

    def test_an_unknown_name_says_what_it_expected(self) -> None:
        with pytest.raises(ValueError, match="Unknown run type 'schema'; expected one of baseline, arms-agent"):
            get_condition("schema")


class TestDescriptor:
    def test_requires_keys_may_be_any_iterable(self) -> None:
        """A module writing a tuple or a list must not end up with a different check."""
        condition = Condition(
            name="x",
            build_workflow=lambda **_kwargs: "graph",
            build_user_prompt=lambda _record, _iri: "prompt",
            requires_keys=["A", "B", "A"],
        )
        assert condition.requires_keys == {"A", "B"}

    def test_a_condition_declares_nothing_it_need_not(self) -> None:
        """Only the first three are required; the rest have to carry sane defaults."""
        condition = Condition(
            name="x",
            build_workflow=lambda **_kwargs: "graph",
            build_user_prompt=lambda _record, _iri: "prompt",
        )
        assert condition.requires_keys == frozenset()
        assert (condition.order, condition.family, condition.module) == (0, "", "")
