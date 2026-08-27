"""Tests for parsing the agent's fenced record and processing-log blocks."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from arms_agent import utils
from arms_agent.schema import build_output_model, build_response_model

TEMPLATE: dict[str, Any] = {
    "type": "template",
    "name": "demo",
    "children": [
        {"name": "manufacturer", "type": "string"},
        {"name": "model", "type": "string"},
    ],
}

LOG_ENTRY: dict[str, Any] = {
    "key": "manufacturer",
    "value": "Acme Corporation",
    "legacy_fields": ["product_name"],
    "legacy_values": ["Acme X100 Analyzer"],
    "resolution": "derived",
    "candidates": ["Acme Corporation", "Acme Instruments Ltd"],
    "reasoning": "The product name identifies the maker.",
}


def _response(record: str | None = None, log: str | None = None, prose: str = "Done.") -> str:
    """Build an agent response with the given fenced blocks."""
    parts = [prose]
    if record is not None:
        parts.append(f"```json record\n{record}\n```")
    if log is not None:
        parts.append(f"```json log\n{log}\n```")
    return "\n\n".join(parts)


@pytest.fixture
def output_model() -> type:
    """The Pydantic model for TEMPLATE."""
    return build_output_model(TEMPLATE)


class TestParseFencedJson:
    """Tests for locating and parsing a fenced block."""

    def test_parses_record_block(self) -> None:
        text = _response(record='{"manufacturer": "Acme Corporation", "model": "X100"}')
        parsed = utils._parse_fenced_json(text, utils._RECORD_BLOCK_RE, "record")
        assert parsed == {"manufacturer": "Acme Corporation", "model": "X100"}

    def test_missing_block_returns_none(self) -> None:
        assert utils._parse_fenced_json("no blocks here", utils._RECORD_BLOCK_RE, "record") is None

    def test_malformed_json_returns_none(self) -> None:
        text = _response(record='{"manufacturer": ')
        assert utils._parse_fenced_json(text, utils._RECORD_BLOCK_RE, "record") is None

    def test_last_block_wins(self) -> None:
        text = _response(record='{"manufacturer": "First"}') + '\n\n```json record\n{"manufacturer": "Second"}\n```'
        parsed = utils._parse_fenced_json(text, utils._RECORD_BLOCK_RE, "record")
        assert parsed == {"manufacturer": "Second"}

    def test_record_and_log_markers_do_not_collide(self) -> None:
        text = _response(record='{"manufacturer": "Acme Corporation"}', log=json.dumps([LOG_ENTRY]))
        record = utils._parse_fenced_json(text, utils._RECORD_BLOCK_RE, "record")
        log = utils._parse_fenced_json(text, utils._LOG_BLOCK_RE, "log")
        assert record == {"manufacturer": "Acme Corporation"}
        assert isinstance(log, list)
        assert log[0]["key"] == "manufacturer"

    def test_plain_json_fence_is_not_matched(self) -> None:
        # An unmarked block belongs to neither, so it must not be mistaken for one.
        text = 'prose\n\n```json\n{"manufacturer": "Acme"}\n```'
        assert utils._parse_fenced_json(text, utils._RECORD_BLOCK_RE, "record") is None
        assert utils._parse_fenced_json(text, utils._LOG_BLOCK_RE, "log") is None


class TestExtractLog:
    """Tests for processing-log extraction."""

    def test_extracts_entries(self) -> None:
        assert utils._extract_log(_response(log=json.dumps([LOG_ENTRY]))) == [LOG_ENTRY]

    def test_missing_log_is_empty(self) -> None:
        assert utils._extract_log(_response(record="{}")) == []

    def test_malformed_log_is_empty(self) -> None:
        assert utils._extract_log(_response(log="[{oops}]")) == []

    def test_non_array_log_is_empty(self) -> None:
        assert utils._extract_log(_response(log='{"key": "manufacturer"}')) == []

    def test_non_object_entries_are_dropped(self) -> None:
        assert utils._extract_log(_response(log=json.dumps([LOG_ENTRY, "junk", 3]))) == [LOG_ENTRY]


class TestCoerceRecord:
    """Tests for validating a parsed record against the template."""

    def test_absent_fields_become_null(self, output_model: type) -> None:
        assert utils._coerce_record({"manufacturer": "Acme"}, output_model) == {
            "manufacturer": "Acme",
            "model": None,
        }

    def test_field_outside_the_template_is_rejected(self, output_model: type) -> None:
        assert utils._coerce_record({"manufacturer": "Acme", "surprise": 1}, output_model) is None

    def test_non_object_is_rejected(self, output_model: type) -> None:
        assert utils._coerce_record([1, 2], output_model) is None

    def test_wrong_datatype_is_rejected(self, output_model: type) -> None:
        assert utils._coerce_record({"manufacturer": {"nested": "object"}, "model": None}, output_model) is None


class TestStructuredResponse:
    """Tests for the path where the agent's answer was schema-enforced.

    Nothing here may reach for the template or the extraction LLM: a validated answer
    is the agent's own output and must be recorded as it stands.
    """

    @pytest.fixture(autouse=True)
    def _forbid_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("a validated answer must not be re-extracted")

        monkeypatch.setattr(utils, "_get_extraction_llm", _fail)
        monkeypatch.setattr(utils.get_cedar_template, "func", _fail)

    def _state(self, structured: Any) -> dict[str, Any]:
        return {"messages": [], "cedar_template_iri": "iri", "structured_response": structured}

    def test_splits_a_validated_model_into_record_and_log(self) -> None:
        model = build_response_model(TEMPLATE)
        answer = model.model_validate({"record": {"manufacturer": "Acme", "model": "X100"}, "log": [LOG_ENTRY]})
        result = utils.extract_output_metadata(self._state(answer))
        assert result["metadata"] == {"manufacturer": "Acme", "model": "X100"}
        assert result["decisions"] == [LOG_ENTRY]

    def test_accepts_a_plain_dict(self) -> None:
        """A checkpointer may hand the answer back as a dict rather than the model."""
        result = utils.extract_output_metadata(
            self._state({"record": {"manufacturer": "Acme", "model": None}, "log": [LOG_ENTRY]})
        )
        assert result["metadata"] == {"manufacturer": "Acme", "model": None}
        assert result["decisions"] == [LOG_ENTRY]

    def test_an_empty_log_yields_no_decisions(self) -> None:
        result = utils.extract_output_metadata(self._state({"record": {"manufacturer": "Acme"}, "log": []}))
        assert result["decisions"] == []

    def test_the_record_is_traced_before_the_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Langfuse orders sibling events by creation time, so the record has to be sent first."""
        traced: list[tuple[str, Any, str]] = []
        monkeypatch.setattr(
            utils, "record_migrated_record", lambda record, *, source: traced.append(("record", record, source))
        )
        monkeypatch.setattr(
            utils, "record_processing_log", lambda decisions, *, source: traced.append(("log", decisions, source))
        )
        utils.extract_output_metadata(self._state({"record": {"manufacturer": "Acme"}, "log": [LOG_ENTRY]}))
        assert traced == [
            ("record", {"manufacturer": "Acme"}, "structured_response"),
            ("log", [LOG_ENTRY], "structured_response"),
        ]


class TestExtractOutputMetadata:
    """Tests for the extraction node, including the LLM fallback."""

    @pytest.fixture(autouse=True)
    def _stub_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils.get_cedar_template, "func", lambda template_id: TEMPLATE)

    def _state(self, text: str) -> dict[str, Any]:
        return {"messages": [AIMessage(content=text)], "cedar_template_iri": "iri"}

    def test_uses_the_record_block_without_calling_the_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> None:
            raise AssertionError("the extraction LLM must not be called when the record block is usable")

        monkeypatch.setattr(utils, "_get_extraction_llm", boom)
        text = _response(record='{"manufacturer": "Acme Corporation", "model": "X100"}', log=json.dumps([LOG_ENTRY]))
        result = utils.extract_output_metadata(self._state(text))
        assert result["metadata"] == {"manufacturer": "Acme Corporation", "model": "X100"}
        assert result["decisions"] == [LOG_ENTRY]

    def test_falls_back_to_the_llm_when_the_record_block_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        class FakeLLM:
            def invoke(self, _prompt: str, **_kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                return AIMessage(content='{"manufacturer": "Acme Corporation", "model": null}')

        monkeypatch.setattr(utils, "_get_extraction_llm", lambda: FakeLLM())
        result = utils.extract_output_metadata(self._state(_response(log=json.dumps([LOG_ENTRY]))))
        assert calls == 1
        assert result["metadata"]["manufacturer"] == "Acme Corporation"
        # The log is still recovered on the fallback path.
        assert result["decisions"] == [LOG_ENTRY]

    def test_falls_back_when_the_record_block_violates_the_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        class FakeLLM:
            def invoke(self, _prompt: str, **_kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                return AIMessage(content='{"manufacturer": "Acme Corporation", "model": null}')

        monkeypatch.setattr(utils, "_get_extraction_llm", lambda: FakeLLM())
        text = _response(record='{"manufacturer": "Acme", "not_in_template": true}')
        result = utils.extract_output_metadata(self._state(text))
        assert calls == 1
        assert result["metadata"] == {"manufacturer": "Acme Corporation", "model": None}

    def test_the_fallback_call_is_given_the_run_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the config the reconstruction is invisible to the token tracker and to Langfuse."""
        seen: dict[str, Any] = {}

        class FakeLLM:
            def invoke(self, _prompt: str, config: Any = None, **_kwargs: Any) -> Any:
                seen["config"] = config
                return AIMessage(content='{"manufacturer": "Acme", "model": null}')

        monkeypatch.setattr(utils, "_get_extraction_llm", lambda: FakeLLM())
        config = {"callbacks": ["token-tracker"], "run_name": "migrate-record-1"}
        utils.extract_output_metadata(self._state(_response(prose="no blocks here")), config)
        assert seen["config"] is config

    def test_no_log_block_yields_no_decisions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(utils, "_get_extraction_llm", lambda: pytest.fail("should not be called"))
        result = utils.extract_output_metadata(self._state(_response(record='{"manufacturer": "Acme"}')))
        assert result["decisions"] == []
        assert result["metadata"] == {"manufacturer": "Acme", "model": None}
