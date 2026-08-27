"""Orchestration functions for running experiments and computing metrics."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from arms_agent.token_tracker import TokenUsageTracker
from arms_agent.tracing import flush_tracing, instrument, traced_run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def _usage_record(tracker: TokenUsageTracker) -> dict[str, Any]:
    """Return a tracker's accumulated usage as a JSON-serialisable dict."""
    return {
        "prompt_tokens": tracker.prompt_tokens,
        "cached_tokens": tracker.cached_tokens,
        "completion_tokens": tracker.completion_tokens,
        "reasoning_tokens": tracker.reasoning_tokens,
        "total_tokens": tracker.total_tokens,
        "estimated_cost_usd": round(tracker.total_cost, 6),
    }


def _sum_usage(trackers: Iterable[TokenUsageTracker]) -> TokenUsageTracker:
    """Total several per-file trackers into one, to reuse its summary formatting."""
    combined = TokenUsageTracker()
    for tracker in trackers:
        combined.prompt_tokens += tracker.prompt_tokens
        combined.cached_tokens += tracker.cached_tokens
        combined.completion_tokens += tracker.completion_tokens
        combined.reasoning_tokens += tracker.reasoning_tokens
        combined.total_tokens += tracker.total_tokens
        combined.total_cost += tracker.total_cost
    return combined


def run_experiment(
    template_iri: str,
    input_dir: Path,
    output_dir: Path,
    workflow_factory: Callable[[], CompiledStateGraph],
    user_prompt_builder: Callable[[dict[str, Any], str], str],
    *,
    config: dict[str, Any] | None = None,
    max_concurrency: int = 5,
) -> list[Path]:
    """Run the migration workflow on all JSON files in *input_dir*.

    The workflow is built once via *workflow_factory* and reused for every
    input file.  Up to *max_concurrency* files are processed in parallel.
    Each result is written to *output_dir* with the same filename as the
    input.  The user message is constructed by *user_prompt_builder*.

    Token usage and estimated cost are recorded per file under
    ``<output_dir>/usage/``, with the sweep total in ``usage/_sweep_total.json``.

    Returns the list of output file paths that were written.
    """
    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        logger.warning("No *.json files found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = workflow_factory()

    async def _run_all() -> list[tuple[Path, TokenUsageTracker]]:
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            _process_file(workflow, input_file, output_dir, template_iri, user_prompt_builder, config, semaphore)
            for input_file in input_files
        ]
        return list(await asyncio.gather(*tasks))

    try:
        results = asyncio.run(_run_all())
    finally:
        # One flush for the whole sweep; per-file flushing would serialise the
        # exporter against the concurrent runs.
        flush_tracing()

    total = _sum_usage(tracker for _, tracker in results)
    total_path = output_dir / "usage" / "_sweep_total.json"
    total_path.parent.mkdir(parents=True, exist_ok=True)
    with open(total_path, "w") as f:
        json.dump({"files": len(results), **_usage_record(total)}, f, indent=2)
    logger.info("Sweep usage over %d file(s): %s", len(results), total.usage_summary())

    return [output_path for output_path, _ in results]


async def _process_file(
    workflow: CompiledStateGraph,
    input_file: Path,
    output_dir: Path,
    template_iri: str,
    user_prompt_builder: Callable[[dict[str, Any], str], str],
    config: dict[str, Any] | None,
    semaphore: asyncio.Semaphore,
) -> tuple[Path, TokenUsageTracker]:
    """Process a single input file through the migration workflow.

    Acquires *semaphore* before invoking the workflow so that at most
    *max_concurrency* files are processed in parallel.

    Returns the output path and this file's token usage.
    """
    from langchain_core.messages import HumanMessage

    async with semaphore:
        task_name = asyncio.current_task().get_name()
        logger.info("[%s] Processing %s", task_name, input_file.name)
        with open(input_file) as f:
            legacy_metadata = json.load(f)

        user_message = user_prompt_builder(legacy_metadata, template_iri)

        tracker = TokenUsageTracker()
        run_config = dict(config) if config else {}
        run_config.setdefault("recursion_limit", 30)
        run_config["run_name"] = f"evaluate-{input_file.stem}"
        run_config.setdefault("tags", [])
        run_config["tags"] = [*run_config["tags"], input_file.stem]
        run_config.setdefault("metadata", {})
        run_config["metadata"] = {**run_config["metadata"], "input_file": input_file.name}
        # A fresh list, so counting this file's tokens never touches the shared config.
        run_config["callbacks"] = [*(run_config.get("callbacks") or []), tracker]
        # Each file gets its own handler so concurrent runs keep separate traces.
        run_config = instrument(run_config)

        # Entered inside this file's own task, so the tracing context each file
        # attaches stays private to it while files are processed concurrently.
        with traced_run(run_config["run_name"], {"input_file": input_file.name, "template_iri": template_iri}):
            result = await workflow.ainvoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                    "cedar_template_iri": template_iri,
                },
                config=run_config,
            )

        output_path = output_dir / input_file.name
        with open(output_path, "w") as f:
            json.dump(result["metadata"], f, indent=2)
        logger.info("[%s] Wrote %s", task_name, output_path)

        # The processing log goes in a sibling directory so that *output_dir* keeps
        # one file per input, matching the gold standard for evaluation.
        decisions = result.get("decisions") or []
        decisions_path = output_dir / "decisions" / input_file.name
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decisions_path, "w") as f:
            json.dump(decisions, f, indent=2)
        if not decisions:
            logger.warning("[%s] No processing-log entries for %s", task_name, input_file.name)

        # Usage goes beside the processing log, for the same reason: *output_dir*
        # keeps one file per input so it lines up with the gold standard.
        usage_path = output_dir / "usage" / input_file.name
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(usage_path, "w") as f:
            json.dump({"input_file": input_file.name, **_usage_record(tracker)}, f, indent=2)
        logger.info("[%s] %s: %s", task_name, input_file.name, tracker.usage_summary())

        return output_path, tracker
