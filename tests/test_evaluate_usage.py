"""Tests for the token usage the evaluation sweep records.

``run_experiment`` is driven with a stub workflow that reports token usage
through the callbacks it is handed, which is how the real graph reports it, so
no API call is involved.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.outputs import LLMResult

from evaluate import run_experiment

if TYPE_CHECKING:
    from pathlib import Path

_TEMPLATE_IRI = "https://example.org/templates/test"


class _StubWorkflow:
    """Reports usage to the run's callbacks, the way a real LLM call does."""

    def __init__(
        self,
        model: str = "gpt-5.6-terra",
        prompt_tokens: int = 1_000_000,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fire on_llm_end at every registered handler, then return a minimal result."""
        result = LLMResult(
            generations=[[]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
                "model_name": self.model,
            },
        )
        for handler in (config or {}).get("callbacks") or []:
            handler.on_llm_end(result)
        return {"metadata": {"title": "migrated"}, "decisions": []}


class _StubHandler:
    """A caller-supplied callback, to prove the tracker is added without mutating it."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Ignore the usage; this handler exists only to occupy the caller's list."""


def _prompt_builder(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    return f"{template_iri} {json.dumps(legacy_metadata)}"


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real Langfuse keys from activating an exporter."""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")


def _write_inputs(tmp_path: Path, count: int) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(count):
        (input_dir / f"record-{index}.json").write_text(json.dumps({"title": f"record {index}"}))
    return input_dir


def _run(tmp_path: Path, count: int, workflow: _StubWorkflow | None = None) -> Path:
    output_dir = tmp_path / "output"
    run_experiment(
        template_iri=_TEMPLATE_IRI,
        input_dir=_write_inputs(tmp_path, count),
        output_dir=output_dir,
        workflow_factory=lambda: workflow or _StubWorkflow(),
        user_prompt_builder=_prompt_builder,
    )
    return output_dir


def test_usage_is_recorded_per_input_file(tmp_path: Path) -> None:
    """Each record gets its own usage file, so an expensive outlier is visible."""
    output_dir = _run(tmp_path, 3)

    for index in range(3):
        usage = json.loads((output_dir / "usage" / f"record-{index}.json").read_text())
        assert usage["input_file"] == f"record-{index}.json"
        assert usage["prompt_tokens"] == 1_000_000
        # gpt-5.6-terra: $2.00/1M input
        assert usage["estimated_cost_usd"] == pytest.approx(2.00)


def test_sweep_total_sums_the_files(tmp_path: Path) -> None:
    output_dir = _run(tmp_path, 3)

    total = json.loads((output_dir / "usage" / "_sweep_total.json").read_text())
    assert total["files"] == 3
    assert total["prompt_tokens"] == 3_000_000
    assert total["estimated_cost_usd"] == pytest.approx(6.00)


def test_per_file_usage_is_not_shared_between_files(tmp_path: Path) -> None:
    """A tracker appended to the caller's own callbacks list would leak across files.

    ``_process_file`` shallow-copies *config*, so mutating ``config["callbacks"]``
    in place would give every file the same tracker and multiply the totals.  The
    caller's list is passed non-empty here to pin that.
    """
    caller_callbacks: list[Any] = [_StubHandler()]
    output_dir = tmp_path / "output"
    run_experiment(
        template_iri=_TEMPLATE_IRI,
        input_dir=_write_inputs(tmp_path, 4),
        output_dir=output_dir,
        workflow_factory=_StubWorkflow,
        user_prompt_builder=_prompt_builder,
        config={"callbacks": caller_callbacks},
    )

    for index in range(4):
        usage = json.loads((output_dir / "usage" / f"record-{index}.json").read_text())
        assert usage["prompt_tokens"] == 1_000_000, "usage leaked between files"
    assert len(caller_callbacks) == 1, "the caller's callbacks list was mutated"


def test_usage_stays_out_of_the_output_directory(tmp_path: Path) -> None:
    """*output_dir* must hold one file per input so it lines up with the gold standard."""
    output_dir = _run(tmp_path, 2)

    assert sorted(p.name for p in output_dir.glob("*.json")) == ["record-0.json", "record-1.json"]


def test_run_experiment_still_returns_output_paths(tmp_path: Path) -> None:
    """The public return type is unchanged by the per-file tracker."""
    output_dir = tmp_path / "output"
    written = run_experiment(
        template_iri=_TEMPLATE_IRI,
        input_dir=_write_inputs(tmp_path, 2),
        output_dir=output_dir,
        workflow_factory=_StubWorkflow,
        user_prompt_builder=_prompt_builder,
    )

    assert written == [output_dir / "record-0.json", output_dir / "record-1.json"]


def test_reasoning_tokens_are_recorded_per_file_and_in_the_sweep_total(tmp_path: Path) -> None:
    """Without this the sweep cannot say how much of its output spend was thinking."""
    workflow = _StubWorkflow(completion_tokens=1000, reasoning_tokens=600)
    output_dir = _run(tmp_path, 3, workflow)

    usage = json.loads((output_dir / "usage" / "record-0.json").read_text())
    assert usage["reasoning_tokens"] == 600
    assert usage["completion_tokens"] == 1000, "reasoning must not inflate the completion count"

    total = json.loads((output_dir / "usage" / "_sweep_total.json").read_text())
    assert total["reasoning_tokens"] == 1800


def test_unpriced_model_still_records_tokens(tmp_path: Path) -> None:
    """An unknown model must report its tokens, with cost left at zero."""
    output_dir = _run(tmp_path, 1, _StubWorkflow(model="gpt-5.6-vega"))

    usage = json.loads((output_dir / "usage" / "record-0.json").read_text())
    assert usage["prompt_tokens"] == 1_000_000
    assert usage["estimated_cost_usd"] == 0.0
