"""Accumulate usage and cost across the many LLM calls one migration makes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from arms_agent.token_tracker.pricing import BillingPolicy
from arms_agent.token_tracker.usage import read_usage

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenUsageTracker(BaseCallbackHandler):
    """Accumulates token usage across multiple LLM calls and estimates cost.

    One tracker covers one migration: ``run_experiment`` builds a tracker per input
    file, so what it accumulates is that record's whole cost, tool calls included.
    """

    def __init__(self, billing: BillingPolicy | None = None) -> None:
        """Set up an empty tracker.

        Args:
            billing: How this endpoint charges.  Read from the environment when not
                given, and read once rather than per call, so every call a tracker
                accumulates is priced alike and a file's recorded cost cannot depend
                on when in the sweep it was written.
        """
        super().__init__()
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.billing = billing if billing is not None else BillingPolicy.from_env()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Accumulate token counts and cost from an LLM response."""
        usage = read_usage(response)
        if usage is None:
            logger.debug("LLM response carried no usage; nothing to accumulate")
            return

        self.prompt_tokens += usage.prompt_tokens
        self.cached_tokens += usage.cached_tokens
        self.completion_tokens += usage.completion_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += self.billing.cost_of(usage)

    def usage_summary(self) -> str:
        """Return a human-readable summary of accumulated token usage and cost."""
        return (
            f"User prompt tokens: {self.prompt_tokens:,} "
            f"(cached: {self.cached_tokens:,}) | "
            f"Completion tokens: {self.completion_tokens:,} "
            f"(reasoning: {self.reasoning_tokens:,}) | "
            f"Total tokens: {self.total_tokens:,} | "
            f"Estimated cost: ${self.total_cost:.4f}"
        )
