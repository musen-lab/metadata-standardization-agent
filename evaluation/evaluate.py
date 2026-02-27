"""Orchestration functions for running experiments and computing metrics."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langgraph.graph.state import CompiledStateGraph

from evaluation.metrics import compute_accuracy, compute_completeness, compute_correctness

logger = logging.getLogger(__name__)


def run_experiment(
    template_iri: str,
    input_dir: Path,
    output_dir: Path,
    gold_dir: Path,
    report_path: Path,
    workflow_factory: Callable[[], CompiledStateGraph],
    user_prompt_builder: Callable[[dict[str, Any], str], str],
    *,
    max_concurrency: int = 5,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the full experiment: workflow execution, metric computation, and reporting.

    Returns the per-file metrics list.
    """
    execute_workflow(
        template_iri,
        input_dir,
        output_dir,
        workflow_factory,
        user_prompt_builder,
        config=config,
        max_concurrency=max_concurrency,
    )
    metrics = apply_metrics(output_dir, gold_dir)
    _write_report(metrics, report_path)
    logger.info("Report written to %s", report_path)
    return metrics


def execute_workflow(
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

    Returns the list of output file paths that were written.
    """
    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        logger.warning("No *.json files found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = workflow_factory()

    async def _run_all() -> list[Path]:
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            _process_file(workflow, input_file, output_dir, template_iri, user_prompt_builder, config, semaphore)
            for input_file in input_files
        ]
        return list(await asyncio.gather(*tasks))

    return asyncio.run(_run_all())


def apply_metrics(input_dir: Path, gold_dir: Path) -> list[dict[str, Any]]:
    """Compare predicted outputs in *input_dir* against gold standards in *gold_dir*.

    Files are matched by exact filename. Files without a gold-standard
    counterpart are skipped with a warning.

    Returns a list of per-file metric dicts.
    """
    input_files = sorted(input_dir.glob("*.json"))
    results: list[dict[str, Any]] = []

    for input_file in input_files:
        gold_file = gold_dir / input_file.name
        if not gold_file.exists():
            logger.warning("No gold standard for %s — skipping", input_file.name)
            continue

        with open(input_file) as f:
            predicted = json.load(f)
        with open(gold_file) as f:
            gold = json.load(f)

        results.append(
            {
                "input_file": input_file.name,
                "correctness": compute_correctness(predicted, gold),
                "completeness": compute_completeness(predicted, gold),
                "accuracy": compute_accuracy(predicted, gold),
            }
        )

    return results


async def _process_file(
    workflow: CompiledStateGraph,
    input_file: Path,
    output_dir: Path,
    template_iri: str,
    user_prompt_builder: Callable[[dict[str, Any], str], str],
    config: dict[str, Any] | None,
    semaphore: asyncio.Semaphore,
) -> Path:
    """Process a single input file through the migration workflow.

    Acquires *semaphore* before invoking the workflow so that at most
    *max_concurrency* files are processed in parallel.
    """
    from langchain_core.messages import HumanMessage

    async with semaphore:
        task_name = asyncio.current_task().get_name()
        logger.info("[%s] Processing %s", task_name, input_file.name)
        with open(input_file) as f:
            legacy_metadata = json.load(f)

        user_message = user_prompt_builder(legacy_metadata, template_iri)

        run_config = dict(config) if config else {}
        run_config.setdefault("recursion_limit", 30)
        run_config["run_name"] = f"evaluate-{input_file.stem}"
        run_config.setdefault("tags", [])
        run_config["tags"] = [*run_config["tags"], input_file.stem]
        run_config.setdefault("metadata", {})
        run_config["metadata"] = {**run_config["metadata"], "input_file": input_file.name}

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
        return output_path


def _write_report(metrics: list[dict[str, Any]], report_path: Path) -> None:
    """Write a CSV report with per-file metrics and an AVERAGE summary row."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_file", "correctness", "completeness", "accuracy"])
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
