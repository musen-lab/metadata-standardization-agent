"""Mocked ARMS workflow: preserve prompts/config and never pass gold to the agent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from experiment import adapter
from experiment.dataset import TEMPLATE_IRI


@pytest.mark.parametrize("output", [{"a": "prediction"}, {}, None, []])
def test_gold_blind_adapter(monkeypatch: pytest.MonkeyPatch, output: object) -> None:
    """Only legacy/template reach prompt builder and workflow; missing output fails."""
    workflow = Mock(ainvoke=AsyncMock(return_value={"metadata": output, "decisions": ["not output"]}))
    build = Mock(return_value=workflow)
    prompt = Mock(return_value="UNCHANGED EXISTING PROMPT")
    instrument = Mock(side_effect=lambda config: {**config, "callbacks": ["fresh-handler"]})
    monkeypatch.setattr(adapter, "build_agent_tool_workflow", build)
    monkeypatch.setattr(adapter, "build_user_prompt", prompt)
    monkeypatch.setattr(adapter, "instrument", instrument)
    task = adapter.make_task("configured-model")
    item = {
        "input": {"legacy_metadata": {"old": "data"}, "template_iri": TEMPLATE_IRI},
        "expected_output": {"SECRET_GOLD": "must never reach the agent"},
        "metadata": {"gold_path": "not prompt data"},
    }
    if output:
        assert asyncio.run(task(item=item)) == output
    else:
        with pytest.raises(ValueError, match="nonempty metadata"):
            asyncio.run(task(item=item))
    build.assert_called_once_with(model="configured-model", template_iri=TEMPLATE_IRI)
    prompt.assert_called_once_with({"old": "data"}, TEMPLATE_IRI)
    state = workflow.ainvoke.call_args.args[0]
    assert set(state) == {"messages", "cedar_template_iri"}
    assert state["messages"][0].content == "UNCHANGED EXISTING PROMPT"
    config = workflow.ainvoke.call_args.kwargs["config"]
    assert config["recursion_limit"] == adapter.RECURSION_LIMIT
    assert config["callbacks"] == ["fresh-handler"]
    assert "SECRET_GOLD" not in repr(workflow.ainvoke.call_args)


def test_reject_injected_gold(monkeypatch: pytest.MonkeyPatch) -> None:
    """An accidental expected_output key inside input cannot be forwarded."""
    monkeypatch.setattr(adapter, "build_agent_tool_workflow", Mock())
    task = adapter.make_task("configured-model")
    with pytest.raises(ValueError, match="Expected only RNA-seq"):
        asyncio.run(task(item={"input": {"legacy_metadata": {}, "template_iri": TEMPLATE_IRI, "gold": {}}}))
