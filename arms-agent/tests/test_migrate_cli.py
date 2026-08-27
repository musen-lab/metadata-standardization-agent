"""Tests for the shipped ``arms-migrate`` CLI entry point.

These run ``main()`` in-process with the workflow stubbed out.  The subprocess
approach used for the evaluation CLI cannot work here: the migrate CLI has no
equivalent of the empty-input-directory early return, so a real run would build
an LLM client and call out.  Stubbing the two builders is what lets the tests
assert the model actually reaches the agent -- the arity of those calls is the
thing most likely to drift, since nothing else in the suite constructs a workflow.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from arms_agent import __main__ as migrate_cli

if TYPE_CHECKING:
    from pathlib import Path

_TEMPLATE_IRI = "https://example.org/templates/test"


class _StubWorkflow:
    """Stands in for a compiled graph, recording the state it was invoked with."""

    def __init__(self) -> None:
        self.invoked_with: dict[str, Any] | None = None

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record the input state and return a minimal successful result."""
        self.invoked_with = state
        return {"metadata": {"title": "migrated"}, "decisions": [{"key": "title", "resolution": "copied"}]}


@pytest.fixture
def stub_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace both builders with recorders and disable tracing.

    The CLI builds the agent and then wires it into a graph, so the model and the
    template IRI are recorded off the agent builder while the graph builder returns
    the stub workflow the assertions invoke.

    ``__main__`` calls ``load_dotenv`` at import time, so a developer's real
    Langfuse keys would otherwise switch tracing on inside the test.
    """
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    recorded: dict[str, Any] = {"model": None, "template_iri": None, "agent": None, "workflow": None}

    # The CLI builds the response format from the template IRI, so recording the IRI
    # means recording what it asked for the format of.
    monkeypatch.setattr(migrate_cli, "build_response_format", lambda template_iri: f"format-for-{template_iri}")

    def fake_build_migration_agent(
        model: str,
        system_prompt: str,
        response_format: str | None,
        tools: Any,
        reasoning_effort: str,
        reasoning_mode: str,
    ) -> str:
        recorded["model"] = model
        recorded["system_prompt"] = system_prompt
        recorded["tools"] = list(tools)
        recorded["template_iri"] = (response_format or "").removeprefix("format-for-") or None
        # Positional-or-keyword without defaults, so the CLI omitting either would fail here.
        recorded["reasoning"] = {"effort": reasoning_effort, "mode": reasoning_mode}
        recorded["agent"] = f"agent-for-{model}"
        return recorded["agent"]

    def fake_build_workflow(agent: Any) -> _StubWorkflow:
        workflow = _StubWorkflow()
        # Pinned so the CLI cannot quietly wire up something other than the agent it built.
        assert agent == recorded["agent"]
        recorded["workflow"] = workflow
        return workflow

    monkeypatch.setattr(migrate_cli, "build_migration_agent", fake_build_migration_agent)
    monkeypatch.setattr(migrate_cli, "build_workflow", fake_build_workflow)
    return recorded


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> Path:
    """Invoke the CLI over a one-field legacy record, returning the output directory.

    The output directory is created up front: ``--output`` is only treated as a
    directory when it already exists, and is otherwise taken as a file path.
    """
    input_file = tmp_path / "record.json"
    input_file.write_text(json.dumps({"title": "a legacy title"}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    argv = [
        "arms-migrate",
        "--input",
        str(input_file),
        "--target-schema",
        _TEMPLATE_IRI,
        "--output",
        str(output_dir),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    migrate_cli.main()
    return output_dir


def test_cli_passes_the_default_model_to_the_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_build: dict[str, Any]
) -> None:
    """The CLI must build a workflow, which requires supplying a model."""
    _run(monkeypatch, tmp_path)
    assert stub_build["model"] == "gpt-5.6-terra"


def test_cli_model_flag_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_build: dict[str, Any]
) -> None:
    """``--model`` is free-form, so any identifier reaches the workflow unchanged."""
    _run(monkeypatch, tmp_path, "--model", "gpt-4.1-mini")
    assert stub_build["model"] == "gpt-4.1-mini"


def test_cli_writes_the_record_and_the_processing_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_build: dict[str, Any]
) -> None:
    """Output filenames derive from the input stem when ``--output`` is a directory."""
    output_dir = _run(monkeypatch, tmp_path)
    assert json.loads((output_dir / "record.json").read_text()) == {"title": "migrated"}
    decisions = json.loads((output_dir / "record.decisions.json").read_text())
    assert decisions == [{"key": "title", "resolution": "copied"}]


def test_cli_forwards_the_template_iri_in_the_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_build: dict[str, Any]
) -> None:
    """The graph needs the template IRI in state for the extraction node to fetch it."""
    _run(monkeypatch, tmp_path)
    state = stub_build["workflow"].invoked_with
    assert state["cedar_template_iri"] == _TEMPLATE_IRI
    assert "a legacy title" in state["messages"][0].content


def test_cli_forwards_the_template_iri_at_build_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_build: dict[str, Any]
) -> None:
    """The response schema is built from the template, so the build needs the IRI too."""
    _run(monkeypatch, tmp_path)
    assert stub_build["template_iri"] == _TEMPLATE_IRI


def test_cli_asks_for_high_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_build: dict[str, Any]
) -> None:
    """The shipped run reasons hard; the builder's own default is deliberately lower."""
    _run(monkeypatch, tmp_path)
    assert stub_build["reasoning"] == {"effort": "high", "mode": "standard"}
