"""Offline SDK adapter wiring, configuration gates, result coverage and CLI tests."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
from langfuse import Langfuse

from experiment import adapter, runner
from experiment.dataset import load_items
from experiment.evaluators import make_evaluators

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic credentials only; no authentication requests in mocked tests."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OPENAI_API_KEY", "CEDAR_API_KEY", "BIOPORTAL_API_KEY"):
        monkeypatch.setenv(name, "test-placeholder")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:9999")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setattr(runner, "version", lambda _: runner.SDK_VERSION)


def test_runner_wiring_and_flush(
    corpus: tuple[Path, Callable], monkeypatch: pytest.MonkeyPatch, configured: None
) -> None:
    """Pass local dictionaries and deterministic callbacks into the published SDK API."""
    root, pair = corpus
    pair("a.json")
    client = Mock()
    monkeypatch.setattr("langfuse.get_client", lambda: client)
    task = Mock()
    build = Mock(return_value=task)
    monkeypatch.setattr(adapter, "make_task", build)
    assert (
        runner.run_experiment(model="chosen-model", limit=1, data_dir=root, max_concurrency=2)
        is client.run_experiment.return_value
    )
    call = client.run_experiment.call_args.kwargs
    assert call["max_concurrency"] == 2
    assert call["task"] is task
    assert call["data"] == load_items(root)
    assert len(call["evaluators"]) == len(call["run_evaluators"]) == 1
    assert call["metadata"]["model"] == "chosen-model"
    assert call["metadata"]["langfuse_sdk"] == "4.18.0"
    assert client.method_calls[0][0] == "auth_check"
    client.flush.assert_called_once()
    client.create_dataset.assert_not_called()
    client.create_dataset_item.assert_not_called()
    client.run_experiment.side_effect = RuntimeError("synthetic SDK failure")
    with pytest.raises(RuntimeError, match="synthetic SDK"):
        runner.run_experiment(model="chosen-model", limit=1, data_dir=root)
    assert client.flush.call_count == 2


def test_configuration_gates(monkeypatch: pytest.MonkeyPatch, configured: None) -> None:
    """Fail before network/agent construction on missing keys, host or wrong SDK."""
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        runner.check_environment("chosen-model")
    runner.check_environment("ollama:test")
    monkeypatch.delenv("LANGFUSE_HOST")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="LANGFUSE_HOST"):
        runner.check_environment("ollama:test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:9999")
    runner.check_environment("ollama:test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    with pytest.raises(ValueError, match="must not be false"):
        runner.check_environment("ollama:test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setattr(runner, "version", lambda _: "5.0.4")
    with pytest.raises(ValueError, match="4.18.0"):
        runner.check_environment("ollama:test")


def test_auth_failure_before_agent_build(
    corpus: tuple[Path, Callable], monkeypatch: pytest.MonkeyPatch, configured: None
) -> None:
    """Do not spend on a model when Langfuse auth is rejected."""
    root, pair = corpus
    pair("a.json")
    client = Mock(auth_check=Mock(return_value=False))
    build = Mock()
    monkeypatch.setattr("langfuse.get_client", lambda: client)
    monkeypatch.setattr(adapter, "make_task", build)
    with pytest.raises(ValueError, match="authentication failed"):
        runner.run_experiment(model="chosen-model", limit=1, data_dir=root)
    build.assert_not_called()
    client.flush.assert_called_once()


def test_offline_validation_and_explicit_scope(
    corpus: tuple[Path, Callable], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validation needs no credentials/client/model, while execution requires scope."""
    root, pair = corpus
    pair("a.json")
    monkeypatch.setattr("langfuse.get_client", Mock(side_effect=AssertionError("must stay offline")))
    assert runner.main(["--validate-only", "--data-dir", str(root)]) == 0
    assert "selected 1" in capsys.readouterr().out
    for args in ([], ["--model", "chosen-model"], ["--limit", "5"], ["--validate-only", "--limit", "0"]):
        with pytest.raises(SystemExit) as error:
            runner.main(args)
        assert error.value.code == 2


def test_real_pinned_sdk_offline(corpus: tuple[Path, Callable], monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise 4.18.0 async task and evaluator contracts; forbid socket connections.

    Langfuse tracing is explicitly disabled ONLY in this synthetic SDK contract
    test, not in the live runner. One task failure must not masquerade as success.
    """
    assert runner.version("langfuse") == runner.SDK_VERSION
    root, pair = corpus
    pair("a.json")
    pair("z.json")
    forbidden = Mock(side_effect=AssertionError("Network forbidden in offline test"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    client = Langfuse(public_key="pk-lf-offline-contract-test", secret_key="unused", tracing_enabled=False)
    item_eval, run_eval = make_evaluators(root / "schemas/rnaseq.json", expected_records=2)

    async def task(*, item: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if item["metadata"]["record_id"] == "z":
            raise ValueError("synthetic task failure")
        return {"a": "X"}  # Deliberately not reading expected_output.

    try:
        result = client.run_experiment(
            name="offline-contract",
            data=load_items(root),
            task=task,
            evaluators=[item_eval],
            run_evaluators=[run_eval],
            max_concurrency=1,
        )
        summary = runner.summarize(result)
        assert summary["complete"] is False
        assert summary["scores"]["completed_records"] == 1
        assert summary["scores"]["failed_records"] == 1
        assert summary["scores"]["micro_f1"] == 1
        result = client.run_experiment(
            name="offline-success",
            data=load_items(root, limit=1),
            task=task,
            evaluators=[item_eval],
            run_evaluators=[make_evaluators(root / "schemas/rnaseq.json", expected_records=1)[1]],
            max_concurrency=1,
        )
        assert runner.summarize(result)["complete"] is True
    finally:
        client.shutdown()
    forbidden.assert_not_called()


def test_cli_env_file_and_full_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit dotenv loading preserves shell settings and forwards full selection."""
    import os

    env_file = tmp_path / "settings.env"
    env_file.write_text("OPENAI_API_KEY=file-placeholder\nCEDAR_API_KEY=cedar-placeholder\n")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-placeholder")
    call = Mock(return_value=object())
    monkeypatch.setattr(runner, "run_experiment", call)
    monkeypatch.setattr(runner, "summarize", lambda _: {"complete": True})
    with patch.dict(os.environ):  # Restore keys added directly by dotenv, too.
        assert (
            runner.main(
                [
                    "--env-file",
                    str(env_file),
                    "--model",
                    "chosen-model",
                    "--all",
                    "--max-concurrency",
                    "3",
                    "--run-name",
                    "test-full",
                ]
            )
            == 0
        )
        assert os.environ["OPENAI_API_KEY"] == "shell-placeholder"
        assert os.environ["CEDAR_API_KEY"] == "cedar-placeholder"
    assert call.call_args.kwargs["limit"] is None
    assert call.call_args.kwargs["max_concurrency"] == 3
    assert call.call_args.kwargs["run_name"] == "test-full"
    assert "placeholder" not in capsys.readouterr().out


def test_cli_incomplete_result_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK-swallowed errors must produce a nonzero exit and a safe summary."""
    result = SimpleNamespace(name="test", run_name="run", item_results=[], run_evaluations=[])
    monkeypatch.setattr(runner, "run_experiment", lambda **_: result)
    assert runner.main(["--model", "chosen-model", "--limit", "5"]) == 1
