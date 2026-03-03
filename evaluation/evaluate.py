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

from evaluation.metrics import compute_accuracy

logger = logging.getLogger(__name__)


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
