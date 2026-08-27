"""Tests for reading, pricing and accumulating token usage."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.outputs import LLMResult

from arms_agent.token_tracker import (
    MODEL_COSTS,
    BillingPolicy,
    TokenUsageTracker,
    lookup_rates,
)


def _make_llm_result(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> LLMResult:
    """Create a minimal LLMResult with token_usage metadata.

    ``cached_tokens`` and ``reasoning_tokens`` are omitted from the payload entirely
    when None, mirroring a provider that reports no such breakdown.
    """
    token_usage: dict[str, object] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        token_usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    if reasoning_tokens is not None:
        token_usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return LLMResult(
        generations=[[]],
        llm_output={"token_usage": token_usage, "model_name": model_name},
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

    @pytest.mark.parametrize(
        ("model", "input_cost", "output_cost"),
        [
            ("gpt-5.6-luna", 0.20, 1.20),
            ("gpt-5.6-terra", 2.00, 12.00),
            ("gpt-5.6-sol", 5.00, 30.00),
        ],
    )
    def test_current_model_choices_are_priced(self, model: str, input_cost: float, output_cost: float) -> None:
        """Every model the CLIs offer must price, or a run silently reports $0.00."""
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(1_000_000, 1_000_000, model))

        assert tracker.total_cost == pytest.approx(input_cost + output_cost)

    def test_cached_prompt_tokens_are_billed_at_the_cached_rate(self) -> None:
        """prompt_tokens includes cached_tokens, so only the remainder is billed at full rate."""
        tracker = TokenUsageTracker()
        # gpt-5.6-terra: $2.00/1M input, $0.20/1M cached, $12.00/1M output
        tracker.on_llm_end(_make_llm_result(1_000_000, 0, "gpt-5.6-terra", cached_tokens=750_000))

        expected = (250_000 / 1_000_000) * 2.00 + (750_000 / 1_000_000) * 0.20
        assert tracker.total_cost == pytest.approx(expected)
        assert tracker.cached_tokens == 750_000
        assert tracker.prompt_tokens == 1_000_000

    def test_absent_cache_breakdown_bills_the_whole_prompt(self) -> None:
        """A payload without prompt_tokens_details must not be billed as if fully cached."""
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(1_000_000, 0, "gpt-5.6-terra"))

        assert tracker.total_cost == pytest.approx(2.00)
        assert tracker.cached_tokens == 0

    def test_cached_tokens_exceeding_the_prompt_do_not_go_negative(self) -> None:
        """An inconsistent payload must not credit the caller with a negative cost."""
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(1_000, 0, "gpt-5.6-terra", cached_tokens=5_000))

        assert tracker.total_cost >= 0.0


class TestLookupCost:
    """Tests for model-name resolution, including dated variants."""

    def test_every_price_is_a_three_tuple(self) -> None:
        """Input, cached input and output, in that order."""
        for model, costs in MODEL_COSTS.items():
            assert len(costs) == 3, model
            input_cost, cached_cost, output_cost = costs
            assert cached_cost < input_cost, f"{model}: cached input must be cheaper than input"
            assert output_cost > 0, model

    @pytest.mark.parametrize(
        ("dated", "expected_key"),
        [
            ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"),
            ("gpt-5-mini-2025-08-07", "gpt-5-mini"),
            ("gpt-5.6-terra-2026-08-01", "gpt-5.6-terra"),
        ],
    )
    def test_dated_variant_resolves_to_the_most_specific_name(self, dated: str, expected_key: str) -> None:
        """A shorter name must not claim a dated variant of a longer one."""
        assert lookup_rates(dated) == MODEL_COSTS[expected_key]

    @pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"])
    def test_gpt5_does_not_claim_the_gpt56_family(self, model: str) -> None:
        """``"gpt-5.6-terra".startswith("gpt-5")`` is True, which must not decide the price."""
        assert lookup_rates(model) == MODEL_COSTS[model]
        assert lookup_rates(model) != MODEL_COSTS["gpt-5"]

    @pytest.mark.parametrize("model", ["gpt-5.6-vega", "gpt-6", "unknown-model", ""])
    def test_unrecognised_names_report_no_cost(self, model: str) -> None:
        """An unknown sibling must return None rather than a nearby family's price."""
        assert lookup_rates(model) is None


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

    def test_cached_tokens_are_reported(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(2000, 500, "gpt-5.6-terra", cached_tokens=1536))
        summary = tracker.usage_summary()

        assert "User prompt tokens: 2,000 (cached: 1,536)" in summary


def _make_responses_result(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
    cache_read: int | None = None,
    reasoning: int | None = None,
) -> LLMResult:
    """Create an LLMResult shaped as the Responses API path produces one.

    Deliberately carries no ``llm_output``: langchain builds the Responses result
    without one, putting the counts on the message as ``usage_metadata`` instead.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration

    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read is not None:
        usage["input_token_details"] = {"cache_read": cache_read}
    if reasoning is not None:
        usage["output_token_details"] = {"reasoning": reasoning}
    message = AIMessage(
        content="answer",
        usage_metadata=usage,
        response_metadata={"model_name": model_name, "model_provider": "openai"},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestResponsesApiUsage:
    """Usage from a reasoning run, which goes through the Responses API.

    Read only the chat-completions shape and every such run records zero tokens and
    zero cost, which reads as a free run rather than an unmeasured one.
    """

    def test_counts_are_accumulated(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1000, 500, "gpt-5.6-luna"))
        assert tracker.prompt_tokens == 1000
        assert tracker.completion_tokens == 500
        assert tracker.total_tokens == 1500

    def test_the_cost_is_estimated(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1_000_000, 1_000_000, "gpt-5.6-luna"))
        input_cost, _cached_cost, output_cost = MODEL_COSTS["gpt-5.6-luna"]
        assert tracker.total_cost == pytest.approx(input_cost + output_cost)

    def test_cached_tokens_are_billed_at_the_cached_rate(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna", cache_read=1_000_000))
        _input_cost, cached_cost, _output_cost = MODEL_COSTS["gpt-5.6-luna"]
        assert tracker.cached_tokens == 1_000_000
        assert tracker.total_cost == pytest.approx(cached_cost)

    def test_several_calls_accumulate(self) -> None:
        tracker = TokenUsageTracker()
        for _ in range(3):
            tracker.on_llm_end(_make_responses_result(100, 50, "gpt-5.6-luna"))
        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 150

    def test_a_result_with_no_usage_anywhere_is_ignored(self) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration

        tracker = TokenUsageTracker()
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hi"))]]))
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_an_unpriced_model_records_tokens_but_no_cost(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1000, 500, "some-unknown-model"))
        assert tracker.total_tokens == 1500
        assert tracker.total_cost == 0.0


class TestReasoningTokens:
    """What a run spent thinking, reported as a share of completion and never added to it.

    Both reply shapes carry the figure, under their own names: chat completions puts it
    in ``completion_tokens_details.reasoning_tokens``, the Responses API in
    ``output_token_details.reasoning``.
    """

    def test_read_from_a_chat_completions_reply(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_llm_result(100, 500, "gpt-5.6-luna", reasoning_tokens=300))

        assert tracker.reasoning_tokens == 300

    def test_read_from_a_responses_reply(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(100, 500, "gpt-5.6-luna", reasoning=300))

        assert tracker.reasoning_tokens == 300

    @pytest.mark.parametrize("make_result", [_make_llm_result, _make_responses_result])
    def test_an_absent_breakdown_reports_none_rather_than_all(self, make_result: Any) -> None:
        """A model that does not reason sends no breakdown, which must read as zero."""
        tracker = TokenUsageTracker()
        tracker.on_llm_end(make_result(100, 500, "gpt-4o"))

        assert tracker.reasoning_tokens == 0

    def test_calls_accumulate(self) -> None:
        """One migration makes many calls, so its thinking is the sum of theirs."""
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(100, 500, "gpt-5.6-luna", reasoning=300))
        tracker.on_llm_end(_make_responses_result(200, 800, "gpt-5.6-luna", reasoning=450))

        assert tracker.reasoning_tokens == 750
        assert tracker.completion_tokens == 1300

    def test_they_stay_out_of_the_totals(self) -> None:
        """The endpoint already counts them in output_tokens, so re-adding would double them."""
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1000, 500, "gpt-5.6-luna", reasoning=300))

        assert tracker.completion_tokens == 500
        assert tracker.total_tokens == 1500

    def test_they_do_not_change_the_cost(self) -> None:
        """Two runs differing only in how much they thought must be billed alike."""
        thinking = TokenUsageTracker()
        thinking.on_llm_end(_make_responses_result(1_000_000, 1_000_000, "gpt-5.6-luna", reasoning=900_000))
        answering = TokenUsageTracker()
        answering.on_llm_end(_make_responses_result(1_000_000, 1_000_000, "gpt-5.6-luna", reasoning=0))

        assert thinking.total_cost == pytest.approx(answering.total_cost)

    def test_the_summary_reports_them_beside_the_completion_count(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1000, 11_267, "gpt-5.6-luna", reasoning=8_004))

        assert "Completion tokens: 11,267 (reasoning: 8,004)" in tracker.usage_summary()


class TestCostMultiplier:
    """Pricing an endpoint that does not charge OpenAI's published rates."""

    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("OPENAI_COST_MULTIPLIER", "OPENAI_COST_CACHE_DISCOUNT"):
            monkeypatch.delenv(name, raising=False)

    def test_unset_means_list_price(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna"))
        assert tracker.total_cost == pytest.approx(MODEL_COSTS["gpt-5.6-luna"][0])

    def test_the_multiplier_scales_the_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_COST_MULTIPLIER", "0.5")
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna"))
        assert tracker.total_cost == pytest.approx(MODEL_COSTS["gpt-5.6-luna"][0] / 2)

    @pytest.mark.parametrize("bad", ["half", "", "  ", "-1"])
    def test_an_unusable_multiplier_falls_back_to_list_price(self, bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A mistyped variable must not lose a sweep, nor silently zero its cost."""
        monkeypatch.setenv("OPENAI_COST_MULTIPLIER", bad)
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna"))
        assert tracker.total_cost == pytest.approx(MODEL_COSTS["gpt-5.6-luna"][0])

    def test_the_multiplier_is_fixed_when_the_tracker_is_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Otherwise one file's recorded cost could differ from another's in one sweep."""
        monkeypatch.setenv("OPENAI_COST_MULTIPLIER", "0.5")
        tracker = TokenUsageTracker()
        monkeypatch.setenv("OPENAI_COST_MULTIPLIER", "0.1")
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna"))
        assert tracker.total_cost == pytest.approx(MODEL_COSTS["gpt-5.6-luna"][0] / 2)

    def test_cached_input_can_be_billed_at_the_full_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An endpoint reselling access need not pass the cache discount on."""
        monkeypatch.setenv("OPENAI_COST_CACHE_DISCOUNT", "false")
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna", cache_read=1_000_000))
        assert tracker.cached_tokens == 1_000_000
        assert tracker.total_cost == pytest.approx(MODEL_COSTS["gpt-5.6-luna"][0])

    def test_the_gateways_billing_is_reproduced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pinned against what the gateway's usage endpoint actually reported.

        gpt-5.6-luna, 68,679 input and 20,951 output tokens, billed at $0.0194385 --
        list price with no cache discount, halved.
        """
        monkeypatch.setenv("OPENAI_COST_MULTIPLIER", "0.5")
        monkeypatch.setenv("OPENAI_COST_CACHE_DISCOUNT", "false")
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_make_responses_result(68_679, 20_951, "gpt-5.6-luna", cache_read=40_000))
        assert tracker.total_cost == pytest.approx(0.0194385, abs=5e-7)


class TestBillingPolicyIsInjectable:
    """A policy can be given directly, so pricing is testable without the environment."""

    @pytest.fixture(autouse=True)
    def _clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("OPENAI_COST_MULTIPLIER", "OPENAI_COST_CACHE_DISCOUNT"):
            monkeypatch.delenv(name, raising=False)

    def test_the_default_policy_is_openais_own(self) -> None:
        assert BillingPolicy() == BillingPolicy(multiplier=1.0, discounts_cached_input=True)

    def test_a_given_policy_beats_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_COST_MULTIPLIER", "0.5")
        tracker = TokenUsageTracker(billing=BillingPolicy(multiplier=1.0))
        tracker.on_llm_end(_make_responses_result(1_000_000, 0, "gpt-5.6-luna"))
        assert tracker.total_cost == pytest.approx(MODEL_COSTS["gpt-5.6-luna"][0])

    def test_the_gateway_policy_reproduces_its_billing(self) -> None:
        """Measured: ten 90%-cached calls on the gateway cost $0.0016970."""
        gateway = BillingPolicy(multiplier=0.5, discounts_cached_input=False)
        tracker = TokenUsageTracker(billing=gateway)
        tracker.on_llm_end(_make_responses_result(16_670, 50, "gpt-5.6-luna", cache_read=14_976))
        assert tracker.total_cost == pytest.approx(0.0016970, abs=5e-8)

    def test_the_same_usage_costs_less_through_the_gateway(self) -> None:
        """The two policies must not silently agree, or neither is doing anything."""
        usage = dict(input_tokens=28_253, output_tokens=10_636, model_name="gpt-5.6-luna", cache_read=19_196)
        direct = TokenUsageTracker(billing=BillingPolicy())
        gateway = TokenUsageTracker(billing=BillingPolicy(multiplier=0.5, discounts_cached_input=False))
        for tracker in (direct, gateway):
            tracker.on_llm_end(_make_responses_result(**usage))
        assert direct.total_cost == pytest.approx(0.0149585, abs=5e-7)
        assert gateway.total_cost == pytest.approx(0.0092069, abs=5e-7)
