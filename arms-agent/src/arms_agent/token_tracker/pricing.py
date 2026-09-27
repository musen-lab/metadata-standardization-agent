"""Put a price on what a reply consumed.

No API returns a cost -- the Responses payload carries a ``cost`` field and leaves it
null -- so the figure recorded for a run is worked out here, from provider-reported
token counts against the published rates in :data:`MODEL_COSTS`.

Those rates are OpenAI's, and an endpoint that resells access need not charge them.
:class:`BillingPolicy` is how one differs: a fraction of list price, and a separate
fraction for cached input.  Both are measured against the Stanford AI API Gateway's usage
endpoint rather than assumed: as of 2026-09-26 it bills half of list price, except cached
input, which it bills at OpenAI's full cached rate.
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
_CACHED_MULTIPLIER_VAR = "OPENAI_COST_CACHED_MULTIPLIER"
# Read until 2026-09-26, when the gateway started discounting cached input; still looked
# for so a stale .env is reported rather than silently priced by a rule that is gone.
_RETIRED_CACHE_DISCOUNT_VAR = "OPENAI_COST_CACHE_DISCOUNT"


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

    Cached input gets a fraction of its own, because an endpoint that discounts list
    price need not discount the cached rate alike.  The Stanford gateway does not: it
    halves every rate except cached input's, which it passes through at OpenAI's price.

    Attributes:
        multiplier: The fraction of list price billed for uncached input and output.
            ``0.5`` for the Stanford gateway, ``1.0`` for OpenAI itself.
        cached_multiplier: The fraction of OpenAI's cached-input rate billed for cached
            input tokens, or ``None`` to use *multiplier*.  ``1.0`` for the Stanford
            gateway.  Cached tokens are counted and reported either way; only the price
            changes.
    """

    multiplier: float = 1.0
    cached_multiplier: float | None = None

    @classmethod
    def from_env(cls) -> BillingPolicy:
        """Build the policy the environment describes, falling back to OpenAI's own.

        A multiplier that is not a non-negative number is ignored with a warning rather
        than failing the run: a mistyped variable should not lose a sweep, and list
        price is a wrong answer that is reported rather than silently believed.
        """
        if os.environ.get(_RETIRED_CACHE_DISCOUNT_VAR, "").strip():
            logger.warning(
                "%s is no longer read; set %s instead (1.0 for the Stanford gateway)",
                _RETIRED_CACHE_DISCOUNT_VAR,
                _CACHED_MULTIPLIER_VAR,
            )
        multiplier = _read_multiplier(_MULTIPLIER_VAR)
        return cls(
            multiplier=1.0 if multiplier is None else multiplier,
            cached_multiplier=_read_multiplier(_CACHED_MULTIPLIER_VAR),
        )

    def rates(self, model_name: str) -> tuple[float, float, float] | None:
        """Return what *model_name* is billed per 1M tokens: (input, cached input, output).

        ``None`` for a model with no published rates.  The one place a policy is applied
        to a price, so the token tracker and the Langfuse price definitions cannot drift.
        """
        listed = lookup_rates(model_name)
        if listed is None:
            return None
        input_cost, cached_cost, output_cost = listed
        cached_multiplier = self.multiplier if self.cached_multiplier is None else self.cached_multiplier
        return input_cost * self.multiplier, cached_cost * cached_multiplier, output_cost * self.multiplier

    def cost_of(self, usage: Usage) -> float:
        """Return what *usage* costs under this policy, or ``0.0`` for an unpriced model."""
        rates = self.rates(usage.model_name)
        if rates is None:
            logger.debug("No published rates for %r; recording its tokens at no cost", usage.model_name)
            return 0.0
        input_cost, cached_cost, output_cost = rates
        # prompt_tokens already includes cached_tokens, so bill the remainder at the
        # full rate.  The clamp guards an inconsistent usage payload.  reasoning_tokens
        # needs no such treatment: it is already inside completion_tokens and carries no
        # rate of its own, so the output line below charges for it exactly once.
        uncached = max(usage.prompt_tokens - usage.cached_tokens, 0)
        return (
            (uncached / 1_000_000) * input_cost
            + (usage.cached_tokens / 1_000_000) * cached_cost
            + (usage.completion_tokens / 1_000_000) * output_cost
        )


def _read_multiplier(name: str) -> float | None:
    """Return the non-negative number in environment variable *name*, or ``None`` if unset or unusable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; ignoring it", name, raw)
        return None
    if parsed < 0:
        logger.warning("%s=%r is negative; ignoring it", name, raw)
        return None
    return parsed
