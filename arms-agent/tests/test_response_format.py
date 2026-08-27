"""Tests for binding the template's schema to the agent's answer.

The point of the response format is that a run's recorded record is the agent's own
validated answer, with no text parsing and no second model in between.  These tests
cover the wiring that makes that true, and the graph carrying it through to state.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from arms_agent import agent as agent_module
from arms_agent import utils
from arms_agent.schema import build_response_model
from arms_agent.state import AgentState
from arms_agent.utils import extract_output_metadata

TEMPLATE: dict[str, Any] = {
    "type": "template",
    "name": "demo",
    "children": [
        {"name": "manufacturer", "type": "string"},
        {"name": "channel_count", "type": "integer"},
    ],
}


@pytest.fixture
def stub_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve TEMPLATE instead of reaching CEDAR."""
    monkeypatch.setattr(agent_module.get_cedar_template, "func", lambda template_id: TEMPLATE)


class TestBuildResponseFormat:
    """Tests for turning a template IRI into a provider-enforced response format."""

    def test_binds_the_templates_schema(self, stub_template: None) -> None:
        response_format = agent_module.build_response_format("iri")
        json_schema = response_format.to_model_kwargs()["response_format"]["json_schema"]
        assert json_schema["strict"] is True
        assert set(json_schema["schema"]["properties"]) == {"record", "log"}
        assert set(json_schema["schema"]["$defs"]["demo"]["properties"]) == {"manufacturer", "channel_count"}

    def test_a_failed_fetch_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Binding an empty schema would produce empty records for a whole sweep."""
        monkeypatch.setattr(
            agent_module.get_cedar_template, "func", lambda template_id: {"error": "404 template not found"}
        )
        with pytest.raises(ValueError, match="404 template not found"):
            agent_module.build_response_format("iri")

    def test_a_template_without_fields_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            agent_module.get_cedar_template, "func", lambda template_id: {"type": "template", "name": "d"}
        )
        with pytest.raises(ValueError, match="no fields returned"):
            agent_module.build_response_format("iri")

    def test_the_builder_fetches_no_template_of_its_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The caller supplies the format, so a build must not reach CEDAR behind its back."""

        def _fail(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("the builder should not fetch a template")

        monkeypatch.setattr(agent_module, "build_response_format", _fail)
        captured: dict[str, Any] = {}

        def fake_create_agent(_llm: Any, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "compiled"

        monkeypatch.setattr(agent_module, "create_agent", fake_create_agent)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **_kwargs: "llm")
        built = agent_module.build_migration_agent(
            model="gpt-4.1-mini", system_prompt="a policy", response_format=None, tools=()
        )
        assert built == "compiled"
        assert captured["response_format"] is None
        assert captured["system_prompt"] == "a policy"
        assert list(captured["tools"]) == []


class TestReasoningSettings:
    """Which models are sent reasoning settings, and what the defaults are.

    The differentiator is the reasoning family rather than gpt-5.6 specifically: the
    o-series and the whole gpt-5 family accept these, and a non-reasoning model rejects
    the request outright rather than ignoring them.
    """

    @pytest.fixture(autouse=True)
    def _clear_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the direct-OpenAI form: a developer's own gateway must not decide these."""
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            monkeypatch.delenv(name, raising=False)

    @pytest.fixture
    def llm_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Capture what the builder hands ``ChatOpenAI``."""
        seen: dict[str, Any] = {}
        monkeypatch.setattr(agent_module, "create_agent", lambda _llm, **_kw: "compiled")
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: seen.update(kwargs))
        return seen

    def _build(self, model: str, **extra: Any) -> None:
        agent_module.build_migration_agent(model=model, system_prompt="p", response_format=None, tools=(), **extra)

    @pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5-mini", "o3-mini"])
    def test_reasoning_models_are_sent_the_settings(self, model: str, llm_kwargs: dict[str, Any]) -> None:
        self._build(model, reasoning_effort="high", reasoning_mode="pro")
        assert llm_kwargs["reasoning"] == {"effort": "high", "mode": "pro"}

    @pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-4o"])
    def test_non_reasoning_models_are_sent_nothing(self, model: str, llm_kwargs: dict[str, Any]) -> None:
        """Sent at all, they would be rejected outright rather than ignored."""
        self._build(model, reasoning_effort="high", reasoning_mode="standard")
        assert "reasoning" not in llm_kwargs
        assert "reasoning_effort" not in llm_kwargs

    def test_the_defaults_are_low_and_standard(self, llm_kwargs: dict[str, Any]) -> None:
        self._build("gpt-5.6-terra")
        assert llm_kwargs["reasoning"] == {"effort": "low", "mode": "standard"}


class TestStructuredResponseReachesTheOutput:
    """The validated answer has to survive the graph and land in the recorded output."""

    def test_the_graph_carries_it_from_the_agent_to_the_record(self) -> None:
        answer = build_response_model(TEMPLATE).model_validate(
            {
                "record": {"manufacturer": "Acme Corporation", "channel_count": 4},
                "log": [
                    {
                        "key": "manufacturer",
                        "value": "Acme Corporation",
                        "legacy_fields": ["vendor"],
                        "legacy_values": ["Acme"],
                        "resolution": "harmonized",
                        "candidates": ["Acme Corporation"],
                        "reasoning": "The vendor abbreviation denotes the full label.",
                    }
                ],
            }
        )

        def fake_migrate(_state: AgentState) -> dict[str, Any]:
            return {"structured_response": answer}

        graph = StateGraph(AgentState)
        graph.add_node("migrate", fake_migrate)
        graph.add_node("extract", extract_output_metadata)
        graph.add_edge(START, "migrate")
        graph.add_edge("migrate", "extract")
        graph.add_edge("extract", END)

        result = graph.compile().invoke({"messages": [], "cedar_template_iri": "iri"})
        assert result["metadata"] == {"manufacturer": "Acme Corporation", "channel_count": 4}
        assert result["decisions"][0]["resolution"] == "harmonized"
        # The record must be serialisable exactly as the agent gave it.
        assert json.loads(json.dumps(result["metadata"]))["channel_count"] == 4

    def test_the_node_is_given_the_run_config_by_the_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LangGraph injects the config only for certain annotations, and silently not for others.

        Without it the fallback extraction call is billed to nobody and traced nowhere,
        so this pins the wiring rather than trusting the signature.
        """
        seen: dict[str, Any] = {}

        def spy(state: AgentState, config: Any = None) -> dict[str, Any]:
            seen["config"] = config
            return {"metadata": {}, "decisions": []}

        monkeypatch.setattr(utils, "_from_response_text", spy)

        graph = StateGraph(AgentState)
        graph.add_node("extract", extract_output_metadata)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", END)
        graph.compile().invoke({"messages": [], "cedar_template_iri": "iri"}, config={"run_name": "migrate-record-1"})

        assert seen["config"] is not None, "the extraction node was not given the run config"
        assert "callbacks" in seen["config"]


class TestEndpointResolution:
    """Which endpoint the clients call, and how it is configured.

    ``OPENAI_BASE_URL`` is the OpenAI SDK's own name for this, so a gateway already
    configured for another OpenAI client needs no new variable here.
    """

    @pytest.fixture(autouse=True)
    def _clear_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A developer's own gateway setting must not decide what these tests see."""
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            monkeypatch.delenv(name, raising=False)

    def test_unset_means_openais_own_endpoint(self) -> None:
        assert agent_module.resolve_base_url() is None

    def test_the_sdk_variable_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.edu/v1")
        assert agent_module.resolve_base_url() == "https://gateway.example.edu/v1"

    def test_the_langchain_variable_is_accepted_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example.edu/v1")
        assert agent_module.resolve_base_url() == "https://gateway.example.edu/v1"

    def test_the_sdk_variable_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://sdk.example.edu/v1")
        monkeypatch.setenv("OPENAI_API_BASE", "https://langchain.example.edu/v1")
        assert agent_module.resolve_base_url() == "https://sdk.example.edu/v1"

    def test_an_empty_value_is_not_an_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A commented-out or blank .env line must not send calls to the empty string."""
        monkeypatch.setenv("OPENAI_BASE_URL", "   ")
        assert agent_module.resolve_base_url() is None

    def test_the_agent_is_built_against_the_gateway(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.edu/v1")
        seen: dict[str, Any] = {}
        monkeypatch.setattr(agent_module, "create_agent", lambda _llm, **_kw: "compiled")
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: seen.update(kwargs))
        agent_module.build_migration_agent(model="gpt-5.6-terra", system_prompt="p", response_format=None, tools=())
        assert seen["base_url"] == "https://gateway.example.edu/v1"

    def test_the_extraction_llm_follows_the_same_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A gateway that serves the agent serves the extraction call too."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.edu/v1")
        monkeypatch.setattr(utils, "_extraction_llm", None)
        seen: dict[str, Any] = {}
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: seen.update(kwargs))
        utils._get_extraction_llm()
        assert seen["base_url"] == "https://gateway.example.edu/v1"


class TestReasoningAcrossEndpoints:
    """The reasoning settings take the same form whatever the endpoint.

    `reasoning` moves the client to /v1/responses, which is the only way the tool arm
    can reason: both endpoints refuse function tools together with a reasoning effort on
    /v1/chat/completions. The Stanford AI API Gateway serves /v1/responses as well,
    though KB00019978 does not list it.
    """

    @pytest.fixture
    def llm_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(agent_module, "create_agent", lambda _llm, **_kw: "compiled")
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: seen.update(kwargs))
        return seen

    def _build(self, **extra: Any) -> None:
        agent_module.build_migration_agent(
            model="gpt-5.6-terra", system_prompt="p", response_format=None, tools=(), **extra
        )

    @pytest.mark.parametrize("endpoint", [None, "https://aiapi-prod.stanford.edu/v1"])
    def test_the_form_does_not_depend_on_the_endpoint(
        self, endpoint: str | None, monkeypatch: pytest.MonkeyPatch, llm_kwargs: dict[str, Any]
    ) -> None:
        """Sending effort the chat-completions way instead would cost the tool arm its reasoning."""
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        if endpoint is None:
            monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
        self._build(reasoning_effort="high", reasoning_mode="pro")
        assert llm_kwargs["reasoning"] == {"effort": "high", "mode": "pro"}
        assert "reasoning_effort" not in llm_kwargs
        assert llm_kwargs["base_url"] == endpoint


class TestExtractionModel:
    """The fallback extraction model, overridable for endpoints that do not offer it."""

    @pytest.fixture(autouse=True)
    def _fresh_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils, "_extraction_llm", None)
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_EXTRACTION_MODEL"):
            monkeypatch.delenv(name, raising=False)

    def test_the_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: seen.update(kwargs))
        utils._get_extraction_llm()
        assert seen["model"] == "gpt-4.1-mini"

    def test_the_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Stanford gateway lists gpt-4.1 but not gpt-4.1-mini."""
        monkeypatch.setenv("OPENAI_EXTRACTION_MODEL", "gpt-4.1")
        seen: dict[str, Any] = {}
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: seen.update(kwargs))
        utils._get_extraction_llm()
        assert seen["model"] == "gpt-4.1"
