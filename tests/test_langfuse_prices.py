"""Tests for :mod:`langfuse_prices`: what a model is billed at, and registering it in Langfuse.

Nothing here reaches Langfuse.  The API client is a fake holding a list of definitions,
which is all the module reads and writes through.
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any

import pytest

import sweep
from arms_agent.token_tracker.pricing import BillingPolicy
from langfuse_prices import billed_prices, match_pattern, register_model_prices

_GATEWAY = BillingPolicy(multiplier=0.5, cached_multiplier=1.0)


class _FakeModels:
    """The ``api.models`` endpoints, over an in-memory list of definitions."""

    def __init__(self, definitions: list[SimpleNamespace], page_size: int = 2) -> None:
        self.definitions = definitions
        self.page_size = page_size
        self.deleted: list[str] = []

    def list(self, *, page: int, limit: int) -> SimpleNamespace:
        size = min(limit, self.page_size)
        pages = max(math.ceil(len(self.definitions) / size), 1)
        chunk = self.definitions[(page - 1) * size : page * size]
        return SimpleNamespace(data=chunk, meta=SimpleNamespace(total_pages=pages))

    def create(self, *, model_name: str, match_pattern: str, unit: str, pricing_tiers: list[Any]) -> None:
        tiers = [SimpleNamespace(conditions=tier.conditions, prices=tier.prices) for tier in pricing_tiers]
        self.definitions.append(_definition(model_name, match_pattern, tiers, managed=False))

    def delete(self, model_id: str) -> None:
        self.deleted.append(model_id)
        self.definitions = [d for d in self.definitions if d.id != model_id]


def _definition(name: str, pattern: str, tiers: list[Any], *, managed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"{name}-{len(tiers)}-{managed}-{id(tiers)}",
        model_name=name,
        match_pattern=pattern,
        is_langfuse_managed=managed,
        pricing_tiers=tiers,
    )


def _api(definitions: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(models=_FakeModels(definitions or []))


def _custom(api: SimpleNamespace, name: str) -> list[SimpleNamespace]:
    return [d for d in api.models.definitions if d.model_name == name and not d.is_langfuse_managed]


def test_gateway_prices_are_half_list_except_cached_input() -> None:
    """gpt-5.6-terra lists at $2 / $0.20 cached / $12 per 1M; the gateway bills $1 / $0.20 / $6."""
    prices = billed_prices("gpt-5.6-terra", _GATEWAY)

    assert prices is not None
    assert prices["input"] == pytest.approx(1.00e-6)
    assert prices["input_cache_read"] == pytest.approx(0.20e-6)
    assert prices["output"] == pytest.approx(6.00e-6)
    assert prices["output_reasoning"] == pytest.approx(6.00e-6)


def test_list_prices_keep_the_cache_discount() -> None:
    """Under OpenAI's own billing, cached input keeps its lower rate."""
    prices = billed_prices("gpt-5.6-luna", BillingPolicy())

    assert prices is not None
    assert prices["input"] == pytest.approx(0.20e-6)
    assert prices["input_cache_read"] == pytest.approx(0.02e-6)
    assert prices["output"] == pytest.approx(1.20e-6)


def test_a_traced_run_costs_what_the_gateway_billed() -> None:
    """One arms-agent record's usage as Langfuse recorded it, against the $0.062795 the gateway billed."""
    usage = {"input": 14_487, "input_cache_read": 16_210, "output": 4_521, "output_reasoning": 2_990}
    prices = billed_prices("gpt-5.6-terra", _GATEWAY)

    assert prices is not None
    total = sum(count * prices[key] for key, count in usage.items())
    assert total == pytest.approx(0.062795)


def test_unpriced_model_has_no_prices() -> None:
    assert billed_prices("ollama:qwen3.6:27b", _GATEWAY) is None


@pytest.mark.parametrize(
    ("name", "matches"),
    [
        ("gpt-5.6-terra", True),
        ("openai/gpt-5.6-terra", True),
        ("GPT-5.6-Terra", True),
        ("gpt-5.6-terra-2026-07-01", True),
        ("gpt-5.6-terra-mini", False),
        ("gpt-5.6xterra", False),
    ],
)
def test_match_pattern(name: str, matches: bool) -> None:
    """Claims the model and its dated variants, and nothing that only starts like it."""
    pattern = match_pattern("gpt-5.6-terra").removeprefix("(?i)")
    assert bool(re.match(pattern, name, re.IGNORECASE)) is matches


def test_registers_each_model(capsys: pytest.CaptureFixture[str]) -> None:
    api = _api()

    register_model_prices(["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"], policy=_GATEWAY, api=api)

    for name in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"):
        (definition,) = _custom(api, name)
        assert definition.match_pattern == match_pattern(name)
        assert definition.pricing_tiers[0].prices == billed_prices(name, _GATEWAY)
    expected = "gpt-5.6-terra at input $1, cached input $0.2, output $6 per 1M tokens: registered"
    assert expected in capsys.readouterr().out


def test_leaves_a_current_definition_alone() -> None:
    api = _api()
    register_model_prices(["gpt-5.6-terra"], policy=_GATEWAY, api=api)

    register_model_prices(["gpt-5.6-terra"], policy=_GATEWAY, api=api)

    assert len(_custom(api, "gpt-5.6-terra")) == 1
    assert api.models.deleted == []


def test_replaces_a_stale_definition(capsys: pytest.CaptureFixture[str]) -> None:
    """A changed multiplier replaces the definition rather than adding a second one."""
    api = _api()
    register_model_prices(["gpt-5.6-terra"], policy=BillingPolicy(), api=api)

    register_model_prices(["gpt-5.6-terra"], policy=_GATEWAY, api=api)

    (definition,) = _custom(api, "gpt-5.6-terra")
    assert definition.pricing_tiers[0].prices == billed_prices("gpt-5.6-terra", _GATEWAY)
    assert len(api.models.deleted) == 1
    assert "updated" in capsys.readouterr().out


def test_never_touches_the_definitions_langfuse_ships() -> None:
    """Langfuse's own definition stays; the project's takes precedence over it."""
    shipped = _definition("gpt-5.6-terra", match_pattern("gpt-5.6-terra"), [], managed=True)
    api = _api([shipped])

    register_model_prices(["gpt-5.6-terra"], policy=_GATEWAY, api=api)

    assert shipped in api.models.definitions
    assert api.models.deleted == []
    assert len(_custom(api, "gpt-5.6-terra")) == 1


def test_finds_definitions_on_later_pages() -> None:
    api = _api()
    register_model_prices(["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"], policy=_GATEWAY, api=api)

    register_model_prices(["gpt-5.6-sol"], policy=_GATEWAY, api=api)

    assert len(_custom(api, "gpt-5.6-sol")) == 1
    assert api.models.deleted == []


def test_skips_an_unpriced_model(capsys: pytest.CaptureFixture[str]) -> None:
    api = _api()

    register_model_prices(["ollama:qwen3.6:27b"], policy=_GATEWAY, api=api)

    assert api.models.definitions == []
    assert "no published rates" in capsys.readouterr().out


def test_policy_defaults_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COST_MULTIPLIER", "0.5")
    monkeypatch.setenv("OPENAI_COST_CACHED_MULTIPLIER", "1.0")
    api = _api()

    register_model_prices(["gpt-5.6-terra"], api=api)

    (definition,) = _custom(api, "gpt-5.6-terra")
    assert definition.pricing_tiers[0].prices == billed_prices("gpt-5.6-terra", _GATEWAY)


def test_plan_sweep_registers_its_model_only_when_tracing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The sweep prices the model it is about to trace, and leaves Langfuse alone otherwise."""
    monkeypatch.setattr(sweep, "load_dotenv", lambda *args, **kwargs: False)
    for key in ("OPENAI_API_KEY", "CEDAR_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    input_dir = tmp_path / "atacseq" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "atacseq-0.json").write_text("{}")
    registered: list[list[str]] = []
    monkeypatch.setattr(sweep, "register_model_prices", registered.append)

    monkeypatch.setattr(sweep, "tracing_enabled", lambda: False)
    sweep.plan_sweep(tmp_path, "gpt-5.6-terra", assays=["atacseq"], conditions=["baseline"])
    assert registered == []

    monkeypatch.setattr(sweep, "tracing_enabled", lambda: True)
    sweep.plan_sweep(tmp_path, "gpt-5.6-terra", assays=["atacseq"], conditions=["baseline"])
    assert registered == [["gpt-5.6-terra"]]
