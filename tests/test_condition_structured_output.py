"""Tests that every arm is built alike and answers with a validated object.

The two arms differ in information access and nothing else, so they must also agree on
how their answer is carried: one ``{record, log}`` object, validated against the
template, read by the same extraction node.  Since every arm is now built by the one
``build_migration_agent``, that agreement is structural -- what these tests pin is that
each arm passes it the right things, and that only the tool list and the system prompt
differ.
"""

from __future__ import annotations

from typing import Any

import pytest

from arms_agent import agent as agent_module
from arms_agent.prompts import SYSTEM_PROMPT as ARMS_PROMPT
from arms_agent.schema import build_response_model
from arms_agent.state import AgentState
from arms_agent.tools import all_tools
from conditions.agent_tool import arms
from conditions.prompt_only import baseline

TEMPLATE: dict[str, Any] = {
    "type": "template",
    "name": "demo",
    "children": [
        {"name": "sample_name", "type": "string"},
        {"name": "channel_count", "type": "integer"},
    ],
}

ANSWER: dict[str, Any] = {
    "record": {"sample_name": "S1", "channel_count": 4},
    "log": [
        {
            "key": "sample_name",
            "value": "S1",
            "legacy_fields": ["name"],
            "legacy_values": ["S1"],
            "resolution": "copied",
            "candidates": [],
            "reasoning": "The legacy name states the sample name.",
        }
    ],
}

# Each arm's module and the builder that wires up its workflow.
PROMPT_ONLY = [
    pytest.param(baseline, "build_baseline_workflow", id="baseline"),
]
EVERY_ARM = [*PROMPT_ONLY, pytest.param(arms, "build_agent_tool_workflow", id="agent-tool")]


@pytest.fixture(autouse=True)
def stub_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve TEMPLATE instead of calling CEDAR, and pin the endpoint.

    The reasoning settings take a different form behind a gateway, so a developer with
    ``OPENAI_BASE_URL`` set would otherwise see different assertions pass than CI does.
    """
    monkeypatch.setattr(agent_module.get_cedar_template, "func", lambda template_id: TEMPLATE)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture what an arm passes to ``create_agent``, without constructing a model."""
    seen: dict[str, Any] = {}

    def fake_create_agent(llm: Any, **kwargs: Any) -> Any:
        seen["llm"] = llm
        seen.update(kwargs)
        # A node rather than a sentinel: the builders hand the result straight to
        # ``build_workflow``, which will only accept something LangGraph can run.
        return lambda _state: {}

    monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: kwargs)
    return seen


class TestResponseFormat:
    """Tests for turning a template IRI into the format every arm binds."""

    def test_carries_the_record_and_the_log(self) -> None:
        response_format = agent_module.build_response_format("iri")
        assert set(response_format.schema.model_fields) == {"record", "log"}

    def test_record_mirrors_the_template(self) -> None:
        json_schema = agent_module.build_response_format("iri").to_model_kwargs()["response_format"]["json_schema"]
        assert json_schema["strict"] is True
        assert set(json_schema["schema"]["$defs"]["demo"]["properties"]) == {"sample_name", "channel_count"}

    def test_a_failed_fetch_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Binding an empty schema would produce empty records for a whole sweep."""
        monkeypatch.setattr(agent_module.get_cedar_template, "func", lambda template_id: {"error": "404 not found"})
        with pytest.raises(ValueError, match="404 not found"):
            agent_module.build_response_format("iri")

    def test_a_template_without_fields_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            agent_module.get_cedar_template, "func", lambda template_id: {"type": "template", "name": "d"}
        )
        with pytest.raises(ValueError, match="no fields returned"):
            agent_module.build_response_format("iri")


@pytest.mark.parametrize(("module", "builder_name"), EVERY_ARM)
class TestEveryArmIsBuiltAlike:
    """What must hold for both arms, tool-using or not."""

    def test_the_answer_is_bound_to_the_templates_schema(
        self, module: Any, builder_name: str, built: dict[str, Any]
    ) -> None:
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        expected = build_response_model(TEMPLATE).model_json_schema()
        assert built["response_format"].to_model_kwargs()["response_format"]["json_schema"]["schema"] == expected

    def test_the_state_schema_carries_the_validated_answer(
        self, module: Any, builder_name: str, built: dict[str, Any]
    ) -> None:
        """``structured_response`` only survives the graph if the state declares it."""
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        assert built["state_schema"] is AgentState

    def test_without_a_template_the_answer_is_unconstrained(
        self, module: Any, builder_name: str, built: dict[str, Any]
    ) -> None:
        """A build with no template must keep working, so the schema stays optional."""
        getattr(module, builder_name)(model="gpt-4.1-mini")
        assert built["response_format"] is None

    def test_the_model_is_forwarded_at_temperature_zero(
        self, module: Any, builder_name: str, built: dict[str, Any]
    ) -> None:
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        assert built["llm"]["model"] == "gpt-4.1-mini"
        assert built["llm"]["temperature"] == 0

    def test_every_arm_reasons_alike(self, module: Any, builder_name: str, built: dict[str, Any]) -> None:
        """Reasoning is not the comparison's variable, so no arm may differ in it."""
        getattr(module, builder_name)(model="gpt-5.6-terra", template_iri="iri")
        assert built["llm"]["reasoning"] == {"effort": "high", "mode": "standard"}

    def test_a_non_reasoning_model_is_sent_no_reasoning(
        self, module: Any, builder_name: str, built: dict[str, Any]
    ) -> None:
        """gpt-4.1 rejects the request outright rather than ignoring the settings."""
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        assert "reasoning" not in built["llm"]
        assert "reasoning_effort" not in built["llm"]


@pytest.mark.parametrize(("module", "builder_name"), PROMPT_ONLY)
class TestOnlyInformationAccessDiffers:
    """The prompt-only arm differs from ARMS in tools and prompt, and nothing else."""

    def test_the_arm_is_built_with_no_tools(self, module: Any, builder_name: str, built: dict[str, Any]) -> None:
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        assert list(built["tools"]) == []

    def test_no_tool_calling_kwargs_are_sent(self, module: Any, builder_name: str, built: dict[str, Any]) -> None:
        """``parallel_tool_calls`` is meaningless without tools, so it must not be bound."""
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        assert built["llm"]["model_kwargs"] == {}

    def test_the_arm_sends_its_own_system_prompt(self, module: Any, builder_name: str, built: dict[str, Any]) -> None:
        getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        assert built["system_prompt"] == module.SYSTEM_PROMPT
        assert built["system_prompt"] != ARMS_PROMPT


class TestTheToolArmIsTheOneWithTools:
    """The contrast the experiment measures: same builder, different information access."""

    def test_arms_is_built_with_the_search_tools(self, built: dict[str, Any]) -> None:
        arms.build_agent_tool_workflow(model="gpt-4.1-mini", template_iri="iri")
        assert list(built["tools"]) == list(all_tools)

    def test_arms_sends_the_shipped_system_prompt(self, built: dict[str, Any]) -> None:
        arms.build_agent_tool_workflow(model="gpt-4.1-mini", template_iri="iri")
        assert built["system_prompt"] == ARMS_PROMPT

    def test_arms_asks_for_parallel_tool_calls(self, built: dict[str, Any]) -> None:
        arms.build_agent_tool_workflow(model="gpt-4.1-mini", template_iri="iri")
        assert built["llm"]["model_kwargs"] == {"parallel_tool_calls": True}

    def test_the_o_series_does_not(self, built: dict[str, Any]) -> None:
        """o-series models reject the flag outright."""
        arms.build_agent_tool_workflow(model="o3-mini", template_iri="iri")
        assert built["llm"]["model_kwargs"] == {}


@pytest.mark.parametrize(("module", "builder_name"), EVERY_ARM)
class TestTheWorkflowReachesTheRecord:
    """End to end: a schema-validated answer must arrive as metadata and decisions."""

    def test_the_extraction_node_reads_the_validated_answer(
        self, module: Any, builder_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answer = build_response_model(TEMPLATE).model_validate(ANSWER)

        def fake_build_migration_agent(**_kwargs: Any) -> Any:
            return lambda _state: {"structured_response": answer}

        monkeypatch.setattr(module, "build_migration_agent", fake_build_migration_agent)
        workflow = getattr(module, builder_name)(model="gpt-4.1-mini", template_iri="iri")
        result = workflow.invoke({"messages": [], "cedar_template_iri": "iri"})
        assert result["metadata"] == ANSWER["record"]
        assert result["decisions"] == ANSWER["log"]
