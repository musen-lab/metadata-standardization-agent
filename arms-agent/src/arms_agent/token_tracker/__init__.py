"""Track what a run consumed and what it cost.

Three steps, one module each, because the endpoint matters at two of them:

:mod:`~arms_agent.token_tracker.usage`
    Reads the counts out of a reply.  Every API reports them its own way, so there is
    one reader per shape: chat completions puts them in ``llm_output`` under OpenAI's
    field names, the Responses API puts them on the message under langchain's.

:mod:`~arms_agent.token_tracker.pricing`
    Turns counts into a cost, since no API returns one.  ``MODEL_COSTS`` holds OpenAI's
    published rates and ``BillingPolicy`` says how a given endpoint departs from them --
    a fraction of list price, and whether cached input is discounted at all.

:mod:`~arms_agent.token_tracker.tracker`
    Adds it all up across the many calls one migration makes.

Only :class:`TokenUsageTracker` is needed from outside; the rest is reachable for tests
and for anyone changing how an endpoint is priced.
"""

from arms_agent.token_tracker.pricing import MODEL_COSTS, BillingPolicy, lookup_rates
from arms_agent.token_tracker.tracker import TokenUsageTracker
from arms_agent.token_tracker.usage import Usage, read_usage

__all__ = [
    "MODEL_COSTS",
    "BillingPolicy",
    "TokenUsageTracker",
    "Usage",
    "lookup_rates",
    "read_usage",
]
