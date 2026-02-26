"""CLI entry point for the evaluation framework.

Usage::

    evaluate --input <dir> --target-schema <iri> --output <dir> --gold <dir> --report <path> \
        (--baseline | --experiment) [--debug]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env (project root)
_project_root = Path(__file__).resolve().parents[1]
load_dotenv(_project_root / ".env", override=True)


def main() -> None:
    """Parse arguments and run the evaluation experiment."""
    parser = argparse.ArgumentParser(
        description="Batch-run the migration workflow and evaluate against gold standards.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing input JSON files.")
    parser.add_argument("--target-schema", required=True, help="IRI of the CEDAR template to migrate to.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write migrated output files.")
    parser.add_argument("--gold", required=True, type=Path, help="Directory containing gold standard JSON files.")
    parser.add_argument("--report", required=True, type=Path, help="Path for the CSV report file.")
    workflow_group = parser.add_mutually_exclusive_group(required=True)
    workflow_group.add_argument("--baseline", action="store_true", help="Use the baseline workflow (single LLM call).")
    workflow_group.add_argument("--experiment", action="store_true", help="Use the experiment workflow (ReAct agent).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.baseline:
        from evaluation.baseline import build_baseline_workflow, build_user_prompt

        workflow_factory = build_baseline_workflow
        prompt_builder = build_user_prompt
    else:
        from evaluation.experiment import build_experiment_workflow, build_user_prompt

        workflow_factory = build_experiment_workflow
        prompt_builder = build_user_prompt

    from evaluation.evaluate import run_experiment

    workflow_type = "baseline" if args.baseline else "experiment"
    metrics = run_experiment(
        template_iri=args.target_schema,
        input_dir=args.input,
        output_dir=args.output,
        gold_dir=args.gold,
        report_path=args.report,
        workflow_factory=workflow_factory,
        user_prompt_builder=prompt_builder,
        config={
            "tags": ["evaluation", workflow_type],
            "metadata": {
                "template_iri": args.target_schema,
                "workflow_type": workflow_type,
            },
        },
    )

    if metrics:
        avg_a = sum(m["accuracy"] for m in metrics) / len(metrics)
        avg_c = sum(m["completeness"] for m in metrics) / len(metrics)
        print(f"Evaluated {len(metrics)} file(s) — avg accuracy: {avg_a:.3f}, avg completeness: {avg_c:.3f}")
    else:
        print("No files were evaluated.")


if __name__ == "__main__":
    main()
