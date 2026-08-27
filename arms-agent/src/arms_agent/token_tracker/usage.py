"""Read what a reply says it consumed.

Each OpenAI API reports usage its own way, so each gets its own reader and
:func:`read_usage` tries them in turn.  The difference is not cosmetic: langchain
builds a chat-completions result with the counts in ``llm_output`` under OpenAI's field
names, and a Responses result with no ``llm_output`` at all, the counts instead sitting
on the message as ``usage_metadata`` under langchain's own names.  The agent uses the
Responses API whenever it reasons, so a reader that knows only the older shape records
nothing for a reasoning run -- which reads as a free run rather than an unmeasured one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.outputs import LLMResult


@dataclass(frozen=True)
class Usage:
    """What one reply consumed, in the terms every endpoint agrees on.

    Two of the five counts are breakdowns of another rather than additions to it, and
    both are reported by the provider rather than inferred here.  ``cached_tokens`` is
    part of ``prompt_tokens``: a request whose leading tokens match an earlier one comes
    back marked as a cache read.  ``reasoning_tokens`` is part of ``completion_tokens``:
    what the model spent thinking before it answered, billed at the same output rate as
    the answer itself.  Adding either to the total it belongs to would double-count it.
    """

    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    model_name: str


def _from_chat_completions(response: LLMResult) -> Usage | None:
    """Read a chat-completions reply, which reports usage in ``llm_output``."""
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    if not token_usage:
        return None
    return Usage(
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        # Absent for providers or models that do not report a cache breakdown, in which
        # case the whole prompt is billed at the full input rate.
        cached_tokens=(token_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        completion_tokens=token_usage.get("completion_tokens", 0),
        # Absent for a non-reasoning model, which spends none.
        reasoning_tokens=(token_usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
        total_tokens=token_usage.get("total_tokens", 0),
        model_name=(response.llm_output or {}).get("model_name", ""),
    )


def _from_responses(response: LLMResult) -> Usage | None:
    """Read a Responses-API reply, which reports usage on the message itself.

    Reasoning tokens are recorded on their own so a run's thinking can be told apart
    from its answer, but they stay out of ``completion_tokens`` and ``total_tokens``:
    the endpoint already counts them there, and adding them again would overstate both
    the volume and the cost of every reasoning run.
    """
    prompt = cached = completion = reasoning = total = 0
    model_name = ""
    found = False
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if not usage:
                continue
            found = True
            prompt += usage.get("input_tokens", 0)
            completion += usage.get("output_tokens", 0)
            total += usage.get("total_tokens", 0)
            cached += (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
            reasoning += (usage.get("output_token_details") or {}).get("reasoning", 0) or 0
            model_name = model_name or (getattr(message, "response_metadata", None) or {}).get("model_name", "")
    if not found:
        return None
    return Usage(
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        total_tokens=total,
        model_name=model_name,
    )


# One reader per API shape, tried in turn.  Add an API by adding a reader.
READERS: tuple[Callable[[LLMResult], Usage | None], ...] = (_from_chat_completions, _from_responses)


def read_usage(response: LLMResult) -> Usage | None:
    """Return what *response* reports consuming, or ``None`` if it reports nothing."""
    for reader in READERS:
        usage = reader(response)
        if usage is not None:
            return usage
    return None
