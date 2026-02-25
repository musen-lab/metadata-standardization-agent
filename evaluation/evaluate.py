"""Orchestration functions for running experiments and computing metrics."""

from __future__ import annotations

import csv
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from evaluation.metrics import compute_accuracy, compute_completeness

logger = logging.getLogger(__name__)


def run_experiment(
    input_dir: Path,
    template_iri: str,
    output_dir: Path,
    gold_dir: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    """Run the full experiment: workflow execution, metric computation, and reporting.

    Returns the per-file metrics list.
    """
    run_experiment_workflow(input_dir, template_iri, output_dir)
    metrics = apply_metrics(output_dir, gold_dir)
    _write_report(metrics, report_path)
    logger.info("Report written to %s", report_path)
    return metrics


def run_experiment_workflow(
    input_dir: Path,
    template_iri: str,
    output_dir: Path,
) -> list[Path]:
    """Run the migration workflow on all JSON files in *input_dir*.

    The workflow is built once and reused for every input file. Each result is
    written to *output_dir* with the same filename as the input.

    Returns the list of output file paths that were written.
    """
    from langchain_core.messages import HumanMessage

    from metadata_migration_agent.workflow import build_workflow

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        logger.warning("No *.json files found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = build_workflow()
    output_paths: list[Path] = []

    for input_file in input_files:
        logger.info("Processing %s", input_file.name)
        with open(input_file) as f:
            legacy_metadata = json.load(f)

        user_message = (
            f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
            f"CEDAR Template IRI: {template_iri}\n\n"
            f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
        )

        result = workflow.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "cedar_template_iri": template_iri,
            }
        )

        output_path = output_dir / input_file.name
        with open(output_path, "w") as f:
            json.dump(result["metadata"], f, indent=2)
        output_paths.append(output_path)
        logger.info("Wrote %s", output_path)

    return output_paths


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
                "accuracy": compute_accuracy(predicted, gold),
                "completeness": compute_completeness(predicted, gold),
            }
        )

    return results


def _write_report(metrics: list[dict[str, Any]], report_path: Path) -> None:
    """Write a CSV report with per-file metrics and an AVERAGE summary row."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_file", "accuracy", "completeness"])
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
