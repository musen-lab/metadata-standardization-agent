"""Make Langfuse price a model's calls the way the endpoint actually bills them.

Langfuse works out a generation's cost itself, from the token counts the callback handler
reports and a price definition matched on the model name.  The definitions it ships are
OpenAI's list prices, so under the Stanford gateway every traced call is overstated twice
over: the gateway bills half of list price, and gives cached input no discount.

A price definition registered in the project takes precedence over the one Langfuse
ships, so this module registers one per model, built from the same
:data:`~arms_agent.token_tracker.pricing.MODEL_COSTS` and
:class:`~arms_agent.token_tracker.pricing.BillingPolicy` the token tracker prices with.
The cost in a trace then agrees with the cost the sweep reports.

Langfuse prices only what it ingests after the definition exists, so this has to run
before a sweep, not after -- which is why :func:`sweep.plan_sweep` calls it.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from arms_agent.token_tracker.pricing import BillingPolicy, lookup_rates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langfuse.api import LangfuseAPI, Model

# The usage keys a price is attached to, by which rate each is billed at.  They are the
# keys Langfuse's own OpenAI definitions price; the LangChain handler reports ``input``,
# ``input_cache_read``, ``output`` and ``output_reasoning``, with the cached and reasoning
# tokens already taken out of ``input`` and ``output``, so no token is charged twice.
_INPUT_KEYS = ("input",)
_CACHE_READ_KEYS = ("input_cache_read", "input_cached_tokens", "cache_read_input_tokens")
# OpenAI charges nothing extra to write the cache, so these are billed as plain input.
_CACHE_WRITE_KEYS = ("input_cache_creation", "cache_write_tokens")
_OUTPUT_KEYS = ("output", "output_reasoning", "output_reasoning_tokens", "reasoning_tokens")

_PAGE_SIZE = 100


def billed_prices(model: str, policy: BillingPolicy) -> dict[str, float] | None:
    """Return the per-token price of each usage key of *model* under *policy*, or ``None`` if unpriced."""
    rates = lookup_rates(model)
    if rates is None:
        return None
    input_cost, cached_cost, output_cost = rates
    if not policy.discounts_cached_input:
        cached_cost = input_cost
    per_million = (
        [(key, input_cost) for key in (*_INPUT_KEYS, *_CACHE_WRITE_KEYS)]
        + [(key, cached_cost) for key in _CACHE_READ_KEYS]
        + [(key, output_cost) for key in _OUTPUT_KEYS]
    )
    return {key: cost * policy.multiplier / 1_000_000 for key, cost in per_million}


def match_pattern(model: str) -> str:
    """The name pattern a definition for *model* claims: the name itself, or a dated variant of it."""
    return rf"(?i)^(openai/)?({re.escape(model)})(-\d{{4}}-\d{{2}}-\d{{2}})?$"


def register_model_prices(
    models: Sequence[str],
    *,
    policy: BillingPolicy | None = None,
    api: LangfuseAPI | None = None,
) -> None:
    """Register what each of *models* is billed at as a price definition in the Langfuse project.

    Idempotent: a definition already holding the right prices is left alone, and one
    holding stale prices -- the multiplier changed, say -- is replaced.  Which project is
    written to is decided by the Langfuse key pair, as it is for the traces.

    Args:
        models: The model names to price, as the gateway reports them.
        policy: How the endpoint bills (default: the one the environment describes).
        api: The Langfuse API client (default: the one the environment's keys open).

    A model with no published rates -- a local ``ollama:<tag>`` one, say -- is skipped
    with a note: it is left to whatever Langfuse makes of it, which is no cost at all.
    """
    policy = policy or BillingPolicy.from_env()
    if api is None:
        from langfuse import get_client

        api = get_client().api

    wanted: dict[str, dict[str, float]] = {}
    for model in models:
        prices = billed_prices(model, policy)
        if prices is None:
            print(f"Langfuse has no price registered for {model}: it has no published rates in MODEL_COSTS.")
            continue
        wanted[model] = prices
    if not wanted:
        return

    registered = _project_definitions(api)
    for model, prices in wanted.items():
        current = registered.get(model, [])
        rates = f"input ${prices['input'] * 1e6:g}, cached input ${prices['input_cache_read'] * 1e6:g}, " + (
            f"output ${prices['output'] * 1e6:g} per 1M tokens"
        )
        if len(current) == 1 and _holds(current[0], model, prices):
            print(f"Langfuse prices {model} at {rates}: already registered.")
            continue
        for stale in current:
            api.models.delete(stale.id)
        _create(api, model, prices)
        print(f"Langfuse prices {model} at {rates}: {'updated' if current else 'registered'}.")


def _project_definitions(api: LangfuseAPI) -> dict[str, list[Model]]:
    """The project's own price definitions, by model name, leaving out the ones Langfuse ships."""
    definitions: dict[str, list[Model]] = {}
    page = 1
    while True:
        response = api.models.list(page=page, limit=_PAGE_SIZE)
        for definition in response.data:
            if not definition.is_langfuse_managed:
                definitions.setdefault(definition.model_name, []).append(definition)
        if page >= response.meta.total_pages:
            return definitions
        page += 1


def _holds(definition: Model, model: str, prices: dict[str, float]) -> bool:
    """Whether *definition* already prices *model* at exactly *prices*, and nothing else."""
    if definition.match_pattern != match_pattern(model):
        return False
    tiers = definition.pricing_tiers or []
    if len(tiers) != 1 or tiers[0].conditions:
        return False
    held = tiers[0].prices
    return held.keys() == prices.keys() and all(math.isclose(held[key], prices[key]) for key in prices)


def _create(api: LangfuseAPI, model: str, prices: dict[str, float]) -> None:
    """Register one definition pricing *model* at *prices*, in a single unconditional tier."""
    from langfuse.api import PricingTierInput

    api.models.create(
        model_name=model,
        match_pattern=match_pattern(model),
        unit="TOKENS",
        pricing_tiers=[PricingTierInput(name="Standard", is_default=True, priority=0, conditions=[], prices=prices)],
    )
