"""CLI entry point for the evaluation framework.

Usage::

    evaluate --input <dir> --target-schema <iri> --output <dir> --gold <dir> --report <path> \
        [--model MODEL] [--concurrent N] [--langsmith-project NAME] \
        (--baseline | --experiment) \
        [--evaluation-only] \
        [--debug]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from functools import partial
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
    workflow_group = parser.add_mutually_exclusive_group(required=False)
    workflow_group.add_argument("--baseline", action="store_true", help="Use the baseline workflow (single LLM call).")
    workflow_group.add_argument("--experiment", action="store_true", help="Use the experiment workflow (ReAct agent).")
    gpt_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
    parser.add_argument(
        "--model",
        choices=gpt_models,
        default="gpt-4o-mini",
        help="GPT model variant (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Max number of concurrent file evaluations (default: 5).",
    )
    parser.add_argument("--langsmith-project", type=str, default=None, help="LangSmith project name (overrides .env).")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Skip workflow execution and only compute metrics against existing output files.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    if not args.evaluation_only and not args.baseline and not args.experiment:
        parser.error("one of --baseline or --experiment is required (unless --evaluation-only is set)")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = args.langsmith_project

    if args.evaluation_only:
        from evaluation.evaluate import _write_report, apply_metrics

        metrics = apply_metrics(args.output, args.gold)
        _write_report(metrics, args.report)
    else:
        if args.baseline:
            from evaluation.baseline import build_baseline_workflow, build_user_prompt_v2

            workflow_factory = partial(build_baseline_workflow, model=args.model)
            prompt_builder = build_user_prompt_v2
        else:
            from evaluation.experiment import build_experiment_workflow, build_user_prompt

            workflow_factory = partial(build_experiment_workflow, model=args.model)
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
            max_concurrency=args.concurrent,
            config={
                "tags": ["evaluation", workflow_type],
                "metadata": {
                    "template_iri": args.target_schema,
                    "workflow_type": workflow_type,
                },
            },
        )

    if metrics:
        avg_a = sum(m["correctness"] for m in metrics) / len(metrics)
        avg_c = sum(m["completeness"] for m in metrics) / len(metrics)
        avg_d = sum(m["accuracy"] for m in metrics) / len(metrics)
        print(
            f"Evaluated {len(metrics)} file(s) —"
            f" avg value correctness: {avg_a:.3f}, avg field completeness: {avg_c:.3f},"
            f" avg record accuracy: {avg_d:.3f}"
        )
    else:
        print("No files were evaluated.")


if __name__ == "__main__":
    main()
