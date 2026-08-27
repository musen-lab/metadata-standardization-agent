"""The two experimental conditions, grouped by how each reaches the template.

:mod:`conditions.prompt_only` holds ``baseline``, which is handed the template in its
prompt, together with the single LLM call it makes and its system prompt.
:mod:`conditions.agent_tool` holds ARMS itself, which fetches the template and looks up
terms through tools instead.

``arms-agent`` is spelled that way on the command line, in the output directory names
and in :mod:`analysis`; only the module name differs.

:func:`build_condition` turns one of those names into the two builders it names.  The
CLI and the notebook sweep both dispatch through it, so neither can drift from the other
on what a condition is made of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.graph.state import CompiledStateGraph

#: Every condition name the experiment knows, in the order the paper reports them.  A
#: name outside this set is a typo, and a typo left to fall through to the agent would
#: be an expensive one.
CONDITIONS = ("baseline", "arms-agent")


def build_condition(
    run_type: str,
) -> tuple[Callable[..., CompiledStateGraph], Callable[[dict[str, Any], str], str]]:
    """Return the workflow builder and the user-prompt builder *run_type* is run with.

    Each condition is imported only when it is asked for, so naming one does not pay for
    the other.

    Args:
        run_type: One of :data:`CONDITIONS`.

    Returns:
        ``(build_workflow, build_user_prompt)``.  The workflow builder takes ``model``
        and ``template_iri``; the prompt builder takes the legacy record and that same
        template IRI.

    Raises:
        ValueError: If *run_type* is not one of :data:`CONDITIONS`.
    """
    if run_type == "baseline":
        from conditions.prompt_only.baseline import build_baseline_workflow, build_user_prompt

        return build_baseline_workflow, build_user_prompt
    if run_type == "arms-agent":
        from conditions.agent_tool.arms import build_agent_tool_workflow, build_user_prompt

        return build_agent_tool_workflow, build_user_prompt
    raise ValueError(f"Unknown run type {run_type!r}; expected one of {', '.join(CONDITIONS)}")
