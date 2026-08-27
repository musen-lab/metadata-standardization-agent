"""Put a price on what a reply consumed.

No API returns a cost -- the Responses payload carries a ``cost`` field and leaves it
null -- so the figure recorded for a run is worked out here, from provider-reported
token counts against the published rates in :data:`MODEL_COSTS`.

Those rates are OpenAI's, and an endpoint that resells access need not charge them.
:class:`BillingPolicy` is the two ways one can differ, both measured against the
Stanford AI API Gateway's usage endpoint rather than assumed: it bills half of list
price, and gives no discount at all for cached input.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arms_agent.token_tracker.usage import Usage

logger = logging.getLogger(__name__)

# Pricing per 1M tokens: (input_cost, cached_input_cost, output_cost).
# Standard tier, from https://developers.openai.com/api/docs/pricing as of 2026-08-04.
# Cached input has its own lower rate, which matters here because every call resends
# the same long system prompt -- where the endpoint passes that discount on.
MODEL_COSTS: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
}

_MULTIPLIER_VAR = "OPENAI_COST_MULTIPLIER"
_CACHE_DISCOUNT_VAR = "OPENAI_COST_CACHE_DISCOUNT"
_FALSE = {"false", "0", "no", "off"}


def lookup_rates(model_name: str) -> tuple[float, float, float] | None:
    """Look up rates by model name, matching known prefixes to handle dated variants.

    A dated variant appends ``-<date>`` to a known name, so a prefix only counts
    as a match when the next character is the separator.  Requiring it keeps
    ``gpt-5`` from claiming ``gpt-5.6-terra``, whose next character is ``.``.
    Trying the longest prefix first keeps ``gpt-5`` from claiming
    ``gpt-5-mini-2025-08-07``, which ``gpt-5-mini`` should win.

    An unrecognised name returns ``None`` rather than a nearby price, so an
    unknown model reports no cost instead of a plausible wrong one.
    """
    if not model_name:
        return None
    if model_name in MODEL_COSTS:
        return MODEL_COSTS[model_name]
    for known in sorted(MODEL_COSTS, key=lambda name: (-len(name), name)):
        if model_name.startswith(f"{known}-"):
            return MODEL_COSTS[known]
    return None


@dataclass(frozen=True)
class BillingPolicy:
    """How an endpoint's charges differ from OpenAI's published prices.

    Two independent differences, because they apply at different points: whether cached
    input gets its own cheaper rate decides what a token costs, and the multiplier then
    scales whatever the call came to.  The Stanford gateway needs both -- either alone
    misses its billing in the opposite direction.

    Attributes:
        multiplier: The fraction of list price actually billed.  ``0.5`` for the
            Stanford gateway's 50% discount, ``1.0`` for OpenAI itself.
        discounts_cached_input: Whether cached input tokens are billed at their own
            lower rate.  True for OpenAI; false for the Stanford gateway, whose usage
            endpoint reconciles with every input token at the full rate.  Cached tokens
            are counted and reported either way; only the price changes.
    """

    multiplier: float = 1.0
    discounts_cached_input: bool = True

    @classmethod
    def from_env(cls) -> BillingPolicy:
        """Build the policy the environment describes, falling back to OpenAI's own.

        A multiplier that is not a non-negative number is ignored with a warning rather
        than failing the run: a mistyped variable should not lose a sweep, and list
        price is a wrong answer that is reported rather than silently believed.
        """
        multiplier = 1.0
        raw = os.environ.get(_MULTIPLIER_VAR, "").strip()
        if raw:
            try:
                parsed = float(raw)
            except ValueError:
                logger.warning("%s=%r is not a number; estimating at list price instead", _MULTIPLIER_VAR, raw)
            else:
                if parsed < 0:
                    logger.warning("%s=%r is negative; estimating at list price instead", _MULTIPLIER_VAR, raw)
                else:
                    multiplier = parsed
        discounts = os.environ.get(_CACHE_DISCOUNT_VAR, "").strip().lower() not in _FALSE
        return cls(multiplier=multiplier, discounts_cached_input=discounts)

    def cost_of(self, usage: Usage) -> float:
        """Return what *usage* costs under this policy, or ``0.0`` for an unpriced model."""
        rates = lookup_rates(usage.model_name)
        if rates is None:
            logger.debug("No published rates for %r; recording its tokens at no cost", usage.model_name)
            return 0.0
        input_cost, cached_cost, output_cost = rates
        if not self.discounts_cached_input:
            cached_cost = input_cost
        # prompt_tokens already includes cached_tokens, so bill the remainder at the
        # full rate.  The clamp guards an inconsistent usage payload.  reasoning_tokens
        # needs no such treatment: it is already inside completion_tokens and carries no
        # rate of its own, so the output line below charges for it exactly once.
        uncached = max(usage.prompt_tokens - usage.cached_tokens, 0)
        listed = (
            (uncached / 1_000_000) * input_cost
            + (usage.cached_tokens / 1_000_000) * cached_cost
            + (usage.completion_tokens / 1_000_000) * output_cost
        )
        return listed * self.multiplier
