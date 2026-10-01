"""Tests for the token usage the evaluation sweep records.

``run_experiment`` is driven with a stub workflow that reports token usage
through the callbacks it is handed, which is how the real graph reports it, so
no API call is involved.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx
import openai
import pytest
from langchain_core.outputs import LLMResult

from evaluate import ExperimentResult, run_experiment

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


def test_run_experiment_returns_output_paths(tmp_path: Path) -> None:
    """What was written is returned in input order, so a caller need not glob for it."""
    output_dir = tmp_path / "output"
    written = run_experiment(
        template_iri=_TEMPLATE_IRI,
        input_dir=_write_inputs(tmp_path, 2),
        output_dir=output_dir,
        workflow_factory=_StubWorkflow,
        user_prompt_builder=_prompt_builder,
    )

    assert written.written == [output_dir / "record-0.json", output_dir / "record-1.json"]
    assert written.failed == []


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


class _FailingWorkflow(_StubWorkflow):
    """Fails on one record, the way a dropped connection stops a job partway."""

    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on
        self.seen: list[str] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        name = (config or {})["metadata"]["input_file"]
        self.seen.append(name)
        if name == self.fail_on:
            raise ConnectionError("Connection error.")
        return await super().ainvoke(state, config)


def _run_again(tmp_path: Path, workflow: _StubWorkflow, **kwargs: Any) -> ExperimentResult:
    return run_experiment(
        template_iri=_TEMPLATE_IRI,
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        workflow_factory=lambda: workflow,
        user_prompt_builder=_prompt_builder,
        **{"max_concurrency": 1, **kwargs},
    )


def test_resume_migrates_only_the_records_a_failed_run_left(tmp_path: Path) -> None:
    """A failure writes nothing for its record; the resumed call migrates that one alone."""
    _write_inputs(tmp_path, 3)
    with pytest.raises(ConnectionError):
        _run_again(tmp_path, _FailingWorkflow(fail_on="record-1.json"))
    output_dir = tmp_path / "output"
    finished = sorted(path.name for path in output_dir.glob("*.json"))
    assert "record-1.json" not in finished
    assert "record-0.json" in finished

    retry = _FailingWorkflow(fail_on="")
    written = _run_again(tmp_path, retry, resume=True)

    left = [f"record-{index}.json" for index in range(3) if f"record-{index}.json" not in finished]
    assert retry.seen == left
    assert [path.name for path in written.written] == left
    assert sorted(path.name for path in output_dir.glob("*.json")) == [f"record-{index}.json" for index in range(3)]


def test_a_resumed_runs_total_covers_every_record(tmp_path: Path) -> None:
    """The total is the whole run's, not just the part that finished last."""
    _write_inputs(tmp_path, 3)
    with pytest.raises(ConnectionError):
        _run_again(tmp_path, _FailingWorkflow(fail_on="record-2.json"))

    _run_again(tmp_path, _StubWorkflow(), resume=True)

    total = json.loads((tmp_path / "output" / "usage" / "_sweep_total.json").read_text())
    assert total["files"] == 3
    assert total["prompt_tokens"] == 3_000_000


def test_a_failed_record_leaves_no_file_behind(tmp_path: Path) -> None:
    """No prediction, and no stray staging file that a later glob could trip over."""
    _write_inputs(tmp_path, 1)
    with pytest.raises(ConnectionError):
        _run_again(tmp_path, _FailingWorkflow(fail_on="record-0.json"))

    output_dir = tmp_path / "output"
    assert not list(output_dir.glob("*.json"))
    assert not list((output_dir / "decisions").glob("*.json"))
    assert not list(output_dir.rglob("*.tmp"))


def test_run_experiment_refuses_resume_with_overwrite(tmp_path: Path) -> None:
    _write_inputs(tmp_path, 1)
    with pytest.raises(ValueError, match="not both"):
        _run_again(tmp_path, _StubWorkflow(), overwrite=True, resume=True)


class _ConfigRecordingWorkflow(_StubWorkflow):
    """Records the run config each record was invoked with."""

    def __init__(self) -> None:
        super().__init__()
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.configs.append(dict(config or {}))
        return await super().ainvoke(state, config)


def test_each_record_gets_the_agents_recursion_limit(tmp_path: Path) -> None:
    """The sweep and the CLI share one limit, so a record cannot stop in one and finish in the other."""
    from arms_agent.workflow import RECURSION_LIMIT

    _write_inputs(tmp_path, 2)
    workflow = _ConfigRecordingWorkflow()
    _run_again(tmp_path, workflow)
    assert [config["recursion_limit"] for config in workflow.configs] == [RECURSION_LIMIT, RECURSION_LIMIT]


class _RaisingWorkflow(_StubWorkflow):
    """Spends its tokens on every record, then raises *error* on the ones in *fail_on*."""

    def __init__(self, error: Exception, fail_on: set[str], delay: float = 0.0) -> None:
        super().__init__()
        self.error = error
        self.fail_on = fail_on
        self.delay = delay
        self.seen: list[str] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        name = (config or {})["metadata"]["input_file"]
        self.seen.append(name)
        # Yield once, as a request does, so the records sharing the semaphore are all under way.
        await asyncio.sleep(0)
        if name in self.fail_on:
            for handler in (config or {}).get("callbacks") or []:
                handler.on_llm_end(
                    LLMResult(
                        generations=[[]],
                        llm_output={"token_usage": {"prompt_tokens": 500}, "model_name": self.model},
                    )
                )
            raise self.error
        await asyncio.sleep(self.delay)
        return await super().ainvoke(state, config)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:4000/v1/chat/completions"))


def test_a_failed_record_does_not_stop_the_others(tmp_path: Path) -> None:
    """The case this exists for: one record's bad answer used to abort the whole job."""
    _write_inputs(tmp_path, 3)
    workflow = _RaisingWorkflow(ValueError("Agent produced no text response."), {"record-1.json"})

    result = _run_again(tmp_path, workflow)

    output_dir = tmp_path / "output"
    assert [path.name for path in result.written] == ["record-0.json", "record-2.json"]
    assert [failure.input_file.name for failure in result.failed] == ["record-1.json"]
    assert result.failed[0].error == "ValueError: Agent produced no text response."
    assert result.failed[0].log == output_dir / "failures" / "record-1.json"
    assert sorted(path.name for path in output_dir.glob("*.json")) == ["record-0.json", "record-2.json"]
    total = json.loads((output_dir / "usage" / "_sweep_total.json").read_text())
    assert total["files"] == 2, "the total counts the predictions"


def test_an_empty_agent_reply_is_retried_once_then_the_job_continues(tmp_path: Path) -> None:
    class _TransientEmptyResponse(_StubWorkflow):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: dict[str, int] = {}

        async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
            name = (config or {})["metadata"]["input_file"]
            self.attempts[name] = self.attempts.get(name, 0) + 1
            if name == "record-0.json" and self.attempts[name] == 1:
                raise ValueError("Agent produced no text response.")
            return await super().ainvoke(state, config)

    _write_inputs(tmp_path, 3)
    workflow = _TransientEmptyResponse()
    result = _run_again(tmp_path, workflow)

    assert [path.name for path in result.written] == [f"record-{index}.json" for index in range(3)]
    assert result.failed == []
    assert workflow.attempts == {"record-0.json": 2, "record-1.json": 1, "record-2.json": 1}


def test_a_failed_attempt_is_recorded_with_its_traceback_and_spend(tmp_path: Path) -> None:
    _write_inputs(tmp_path, 1)
    _run_again(tmp_path, _RaisingWorkflow(ValueError("no text"), {"record-0.json"}))

    (attempt,) = json.loads((tmp_path / "output" / "failures" / "record-0.json").read_text())
    assert attempt["input_file"] == "record-0.json"
    assert attempt["error"] == "ValueError"
    assert attempt["message"] == "no text"
    assert "Traceback" in attempt["traceback"]
    assert attempt["usage"]["prompt_tokens"] == 500
    assert attempt["failed_at"]


def test_resume_retries_a_failed_record_and_keeps_its_earlier_attempts(tmp_path: Path) -> None:
    """A record that failed once still shows it after a resumed run migrates it."""
    _write_inputs(tmp_path, 2)
    failing = _RaisingWorkflow(ValueError("no text"), {"record-1.json"})
    _run_again(tmp_path, failing)
    _run_again(tmp_path, failing, resume=True)
    assert failing.seen == ["record-0.json", "record-1.json", "record-1.json"]

    retry = _RaisingWorkflow(ValueError("unused"), set())
    result = _run_again(tmp_path, retry, resume=True)

    assert retry.seen == ["record-1.json"]
    assert [path.name for path in result.written] == ["record-1.json"]
    attempts = json.loads((tmp_path / "output" / "failures" / "record-1.json").read_text())
    assert len(attempts) == 2


def test_an_outage_starts_no_further_record_and_lets_the_running_ones_finish(tmp_path: Path) -> None:
    """Every record after a dropped connection would fail too, so the run stops, but spends nothing twice."""
    _write_inputs(tmp_path, 4)
    workflow = _RaisingWorkflow(_connection_error(), {"record-0.json"}, delay=0.05)

    with pytest.raises(openai.APIConnectionError) as raised:
        _run_again(tmp_path, workflow, max_concurrency=2)

    assert workflow.seen == ["record-0.json", "record-1.json"], "records 2 and 3 were never started"
    output_dir = tmp_path / "output"
    assert [path.name for path in output_dir.glob("*.json")] == ["record-1.json"], "the running record finished"
    assert not (output_dir / "failures").exists(), "an outage is not the record's failure"
    assert "3 record(s) left unmigrated" in "".join(raised.value.__notes__)
    assert "resume=True" in "".join(raised.value.__notes__)
    total = json.loads((output_dir / "usage" / "_sweep_total.json").read_text())
    assert total["files"] == 1


@pytest.mark.parametrize(
    "error",
    [
        openai.AuthenticationError(
            "bad key", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None
        ),
        openai.RateLimitError(
            "budget spent", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None
        ),
        openai.InternalServerError(
            "bad gateway", response=httpx.Response(502, request=httpx.Request("POST", "http://x")), body=None
        ),
    ],
    ids=["auth", "rate-limit", "server"],
)
def test_an_endpoint_refusing_every_request_is_an_outage(tmp_path: Path, error: Exception) -> None:
    _write_inputs(tmp_path, 2)
    workflow = _RaisingWorkflow(error, {"record-0.json"})

    with pytest.raises(type(error)):
        _run_again(tmp_path, workflow)

    assert workflow.seen == ["record-0.json"]


def test_a_request_the_endpoint_rejects_is_the_records_failure(tmp_path: Path) -> None:
    """A 400 -- a prompt too long for the context, say -- is about this record alone."""
    _write_inputs(tmp_path, 2)
    error = openai.BadRequestError(
        "context too long", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None
    )

    result = _run_again(tmp_path, _RaisingWorkflow(error, {"record-0.json"}))

    assert [failure.input_file.name for failure in result.failed] == ["record-0.json"]
    assert [path.name for path in result.written] == ["record-1.json"]
