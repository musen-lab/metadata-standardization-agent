"""Tests for the TokenUsageTracker callback handler."""

from __future__ import annotations

import pytest
from langchain_core.outputs import LLMResult

from metadata_migration_agent.token_tracker import TokenUsageTracker


def _make_llm_result(prompt_tokens: int, completion_tokens: int, model_name: str) -> LLMResult:
    """Create a minimal LLMResult with token_usage metadata."""
    return LLMResult(
        generations=[[]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": model_name,
        },
    )


class TestOnLlmEnd:
    """Tests for token accumulation via on_llm_end."""

    def test_single_call(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(100, 50, "gpt-4o"))

        assert tracker.prompt_tokens == 100
        assert tracker.completion_tokens == 50
        assert tracker.total_tokens == 150

    def test_multiple_calls_accumulate(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(100, 50, "gpt-4o"))
        tracker.on_llm_end(_make_llm_result(200, 80, "gpt-4o"))

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 130
        assert tracker.total_tokens == 430

    def test_no_token_usage_is_ignored(self) -> None:
        tracker = TokenUsageTracker()
        result = LLMResult(generations=[[]], llm_output={})
        tracker.on_llm_end(result)

        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0

    def test_none_llm_output_is_ignored(self) -> None:
        tracker = TokenUsageTracker()
        result = LLMResult(generations=[[]], llm_output=None)
        tracker.on_llm_end(result)

        assert tracker.total_tokens == 0


class TestCostCalculation:
    """Tests for cost estimation."""

    def test_known_model_cost(self) -> None:
        tracker = TokenUsageTracker()
        # gpt-4o: $2.50/1M input, $10.00/1M output
        tracker.on_llm_end(_make_llm_result(1_000_000, 1_000_000, "gpt-4o"))

        assert tracker.total_cost == 2.50 + 10.00

    def test_unknown_model_no_cost(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(1000, 500, "unknown-model"))

        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 1500

    def test_mixed_models_accumulate_cost(self) -> None:
        tracker = TokenUsageTracker()
        # gpt-4o: $2.50/1M input, $10.00/1M output
        tracker.on_llm_end(_make_llm_result(100_000, 50_000, "gpt-4o"))
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        tracker.on_llm_end(_make_llm_result(200_000, 100_000, "gpt-4o-mini"))

        expected_cost = (
            (100_000 / 1_000_000) * 2.50
            + (50_000 / 1_000_000) * 10.00  # gpt-4o
            + (200_000 / 1_000_000) * 0.15
            + (100_000 / 1_000_000) * 0.60  # gpt-4o-mini
        )
        assert tracker.total_cost == pytest.approx(expected_cost)


class TestFormatSummary:
    """Tests for format_summary output."""

    def test_zero_usage(self) -> None:
        tracker = TokenUsageTracker()
        summary = tracker.usage_summary()

        assert "User prompt tokens: 0" in summary
        assert "Completion tokens: 0" in summary
        assert "Total tokens: 0" in summary
        assert "$0.0000" in summary

    def test_with_usage(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(1500, 500, "gpt-4o"))
        summary = tracker.usage_summary()

        assert "User prompt tokens: 1,500" in summary
        assert "Completion tokens: 500" in summary
        assert "Total tokens: 2,000" in summary
        assert "$" in summary
