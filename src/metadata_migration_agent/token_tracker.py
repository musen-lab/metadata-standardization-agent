"""Callback handler for tracking LLM token usage and estimating costs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Pricing per 1M tokens: (input_cost, output_cost)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-mini": (1.00, 4.00),
}


class TokenUsageTracker(BaseCallbackHandler):
    """Accumulates token usage across multiple LLM calls and estimates cost."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Accumulate token counts and cost from an LLM response."""
        token_usage = (response.llm_output or {}).get("token_usage", {})
        if not token_usage:
            return

        prompt = token_usage.get("prompt_tokens", 0)
        completion = token_usage.get("completion_tokens", 0)
        total = token_usage.get("total_tokens", 0)

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

        model_name = (response.llm_output or {}).get("model_name", "")
        llm_output_keys = list(response.llm_output or {})
        logger.debug("LLM response model_name: %r, llm_output keys: %s", model_name, llm_output_keys)
        costs = _lookup_cost(model_name)
        if costs:
            input_cost, output_cost = costs
            self.total_cost += (prompt / 1_000_000) * input_cost + (completion / 1_000_000) * output_cost

    def usage_summary(self) -> str:
        """Return a human-readable summary of accumulated token usage and cost."""
        return (
            f"User prompt tokens: {self.prompt_tokens:,} | "
            f"Completion tokens: {self.completion_tokens:,} | "
            f"Total tokens: {self.total_tokens:,} | "
            f"Estimated cost: ${self.total_cost:.4f}"
        )


def _lookup_cost(model_name: str) -> tuple[float, float] | None:
    """Look up cost by model name, matching against known prefixes to handle dated variants."""
    if not model_name:
        return None
    if model_name in MODEL_COSTS:
        return MODEL_COSTS[model_name]
    for known in MODEL_COSTS:
        if model_name.startswith(known):
            return MODEL_COSTS[known]
    return None