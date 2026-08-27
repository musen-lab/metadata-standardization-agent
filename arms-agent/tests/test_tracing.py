"""Tests for the Langfuse tracing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arms_agent import tracing
from arms_agent.tracing import instrumentation

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a key pair so tracing reports itself as enabled."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")


@pytest.fixture
def stub_handler(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Replace the Langfuse callback handler with a sentinel.

    Constructing the real handler reaches for a Langfuse client, which these
    tests have no reason to build.  Patched on the module that defines it rather
    than on the package, since that is where ``instrument`` looks the name up.
    """
    sentinel = object()
    monkeypatch.setattr(instrumentation, "tracing_callbacks", lambda: [sentinel])
    yield sentinel


class TestTracingEnabled:
    """Tests for the credential and opt-out gate."""

    def test_disabled_without_credentials(self) -> None:
        assert tracing.tracing_enabled() is False

    def test_disabled_with_only_public_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        assert tracing.tracing_enabled() is False

    def test_disabled_with_only_secret_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        assert tracing.tracing_enabled() is False

    def test_enabled_with_both_keys(self, credentials: None) -> None:
        assert tracing.tracing_enabled() is True

    def test_explicit_opt_out_wins_over_credentials(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
        assert tracing.tracing_enabled() is False

    def test_opt_out_is_case_insensitive(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "FALSE")
        assert tracing.tracing_enabled() is False


class TestTracingCallbacks:
    """Tests for handler construction."""

    def test_no_handlers_when_disabled(self) -> None:
        assert tracing.tracing_callbacks() == []

    def test_builds_handler_when_enabled(self, credentials: None) -> None:
        from langfuse.langchain import CallbackHandler

        handlers = tracing.tracing_callbacks()
        assert len(handlers) == 1
        assert isinstance(handlers[0], CallbackHandler)


class TestInstrument:
    """Tests for attaching tracing to a LangChain run config."""

    def test_config_untouched_when_disabled(self) -> None:
        config: dict[str, Any] = {"callbacks": ["existing"], "run_name": "migrate-record-1"}
        assert tracing.instrument(config) is config
        assert config == {"callbacks": ["existing"], "run_name": "migrate-record-1"}

    def test_appends_handler_after_existing_callbacks(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"callbacks": ["token-tracker"], "run_name": "migrate-record-1"})
        assert traced["callbacks"] == ["token-tracker", stub_handler]

    def test_adds_handler_when_no_callbacks_configured(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"run_name": "migrate-record-1"})
        assert traced["callbacks"] == [stub_handler]

    def test_mirrors_run_name_onto_trace_name(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"run_name": "evaluate-record-1"})
        assert traced["metadata"]["langfuse_trace_name"] == "evaluate-record-1"

    def test_preserves_existing_metadata(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"run_name": "evaluate-record-1", "metadata": {"input_file": "record-1.json"}})
        assert traced["metadata"] == {
            "input_file": "record-1.json",
            "langfuse_trace_name": "evaluate-record-1",
        }

    def test_explicit_trace_name_is_not_overwritten(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"run_name": "evaluate-record-1", "metadata": {"langfuse_trace_name": "custom"}})
        assert traced["metadata"]["langfuse_trace_name"] == "custom"

    def test_no_trace_name_without_run_name(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"metadata": {"input_file": "record-1.json"}})
        assert "langfuse_trace_name" not in traced["metadata"]

    def test_other_config_keys_are_carried_through(self, credentials: None, stub_handler: object) -> None:
        traced = tracing.instrument({"recursion_limit": 30, "tags": ["evaluation", "experiment"]})
        assert traced["recursion_limit"] == 30
        assert traced["tags"] == ["evaluation", "experiment"]

    def test_caller_config_is_not_mutated(self, credentials: None, stub_handler: object) -> None:
        """Concurrent evaluation runs share a base config, so instrumenting must not write back into it."""
        config: dict[str, Any] = {
            "callbacks": ["token-tracker"],
            "run_name": "evaluate-record-1",
            "metadata": {"input_file": "record-1.json"},
        }
        tracing.instrument(config)
        assert config == {
            "callbacks": ["token-tracker"],
            "run_name": "evaluate-record-1",
            "metadata": {"input_file": "record-1.json"},
        }


class _RecordingClient:
    """A stand-in Langfuse client that records what a run tried to trace."""

    def __init__(self, trace_id: str | None = "trace-1") -> None:
        self.trace_id = trace_id
        self.events: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []

    def get_current_trace_id(self) -> str | None:
        return self.trace_id

    def create_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def start_as_current_observation(self, **kwargs: Any) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def _span() -> Iterator[str]:
            self.spans.append(kwargs)
            yield "span"

        return _span()


class TestTracedRun:
    """Tests for the explicit root span around a run."""

    def test_yields_none_and_builds_no_client_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail() -> None:
            raise AssertionError("traced_run must not touch Langfuse when tracing is off")

        monkeypatch.setattr("langfuse.get_client", _fail)
        with tracing.traced_run("migrate-record-1") as span:
            assert span is None

    def test_opens_a_named_span_with_metadata(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _RecordingClient()
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        with tracing.traced_run("migrate-record-1", {"input_file": "record-1.json"}) as span:
            assert span == "span"
        assert client.spans == [{"name": "migrate-record-1", "metadata": {"input_file": "record-1.json"}}]


class TestRecordProcessingLog:
    """Tests for storing the agent's processing log in the trace."""

    def test_nothing_sent_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail() -> None:
            raise AssertionError("record_processing_log must not touch Langfuse when tracing is off")

        monkeypatch.setattr("langfuse.get_client", _fail)
        tracing.record_processing_log([{"key": "title", "resolution": "copied"}], source="structured_response")

    def test_log_is_sent_as_one_observation(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _RecordingClient()
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        decisions = [
            {"key": "title", "resolution": "copied"},
            {"key": "assay", "resolution": "harmonized"},
            {"key": None, "resolution": "unmapped"},
            {"key": "extra", "resolution": "harmonized"},
        ]
        tracing.record_processing_log(decisions, source="structured_response")

        assert len(client.events) == 1
        event = client.events[0]
        assert event["name"] == tracing.PROCESSING_LOG_OBSERVATION
        assert event["output"] == decisions
        assert event["metadata"] == {
            "source": "structured_response",
            "entries": 4,
            "resolutions": {"copied": 1, "harmonized": 2, "unmapped": 1},
        }
        assert event["level"] == "DEFAULT"

    def test_an_empty_log_is_recorded_as_a_warning(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runs that produced no log are the ones worth finding, so they must be filterable."""
        client = _RecordingClient()
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        tracing.record_processing_log([], source="extraction_llm")

        event = client.events[0]
        assert event["level"] == "WARNING"
        assert event["metadata"]["entries"] == 0
        assert event["status_message"]

    def test_nothing_sent_without_an_active_span(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sending anyway would strand the log in a trace of its own, away from the run."""
        client = _RecordingClient(trace_id=None)
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        tracing.record_processing_log([{"key": "title", "resolution": "copied"}], source="record_block")
        assert client.events == []


class TestRecordMigratedRecord:
    """Tests for storing the migrated record in the trace."""

    def test_nothing_sent_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail() -> None:
            raise AssertionError("record_migrated_record must not touch Langfuse when tracing is off")

        monkeypatch.setattr("langfuse.get_client", _fail)
        tracing.record_migrated_record({"title": "A study"}, source="structured_response")

    def test_record_is_sent_as_one_observation(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _RecordingClient()
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        record = {"title": "A study", "assay": "ATAC-seq", "tissue": None}
        tracing.record_migrated_record(record, source="structured_response")

        assert len(client.events) == 1
        event = client.events[0]
        assert event["name"] == tracing.MIGRATED_RECORD_OBSERVATION
        assert event["output"] == record
        assert event["metadata"] == {"source": "structured_response", "fields": 3, "populated": 2}
        assert event["level"] == "DEFAULT"

    def test_an_empty_record_is_recorded_as_a_warning(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runs that produced no record are the ones worth finding, so they must be filterable."""
        client = _RecordingClient()
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        tracing.record_migrated_record({}, source="extraction_llm")

        event = client.events[0]
        assert event["level"] == "WARNING"
        assert event["metadata"] == {"source": "extraction_llm", "fields": 0, "populated": 0}
        assert event["status_message"]

    def test_nothing_sent_without_an_active_span(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sending anyway would strand the record in a trace of its own, away from the run."""
        client = _RecordingClient(trace_id=None)
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        tracing.record_migrated_record({"title": "A study"}, source="record_block")
        assert client.events == []

    @pytest.mark.parametrize(
        ("value", "empty"),
        [
            (None, True),
            ("", True),
            ("   ", True),
            ([], True),
            ({}, True),
            ({"child": None}, True),
            ([None, ""], True),
            ("A study", False),
            (0, False),
            (False, False),
            (["ATAC-seq"], False),
            ({"child": "value"}, False),
        ],
    )
    def test_populated_counts_only_fields_carrying_information(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch, value: Any, empty: bool
    ) -> None:
        """A nested element whose children are all null is empty; ``0`` and ``False`` are values."""
        client = _RecordingClient()
        monkeypatch.setattr("langfuse.get_client", lambda: client)
        tracing.record_migrated_record({"field": value}, source="structured_response")
        assert client.events[0]["metadata"]["populated"] == (0 if empty else 1)


class TestFlushTracing:
    """Tests for flushing buffered traces."""

    def test_no_client_built_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail() -> None:
            raise AssertionError("flush_tracing must not touch Langfuse when tracing is off")

        monkeypatch.setattr("langfuse.get_client", _fail)
        tracing.flush_tracing()

    def test_flushes_when_enabled(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        flushed: list[bool] = []

        class _Client:
            def flush(self) -> None:
                flushed.append(True)

        monkeypatch.setattr("langfuse.get_client", lambda: _Client())
        tracing.flush_tracing()
        assert flushed == [True]

    def test_flush_failure_is_swallowed(self, credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A completed migration must not fail because the trace export did."""

        class _Client:
            def flush(self) -> None:
                raise ConnectionError("langfuse unreachable")

        monkeypatch.setattr("langfuse.get_client", lambda: _Client())
        tracing.flush_tracing()
