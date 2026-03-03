"""CLI entry point for the evaluation framework.

Usage::

    evaluate --input <dir> --target-schema <iri> --output <dir> \
        [--model MODEL] [--concurrent N] [--langsmith-project NAME] \
        (--baseline | --experiment) \
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
    workflow_group = parser.add_mutually_exclusive_group(required=True)
    workflow_group.add_argument("--baseline", action="store_true", help="Use the baseline workflow (single LLM call).")
    workflow_group.add_argument("--experiment", action="store_true", help="Use the experiment workflow (ReAct agent).")
    gpt_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
    parser.add_argument(
        "--model",
        choices=gpt_models,
        default="gpt-4.1-mini",
        help="GPT model variant (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Max number of concurrent file evaluations (default: 5).",
    )
    parser.add_argument("--langsmith-project", type=str, default=None, help="LangSmith project name (overrides .env).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = args.langsmith_project

    if args.baseline:
        from evaluation.baseline import build_baseline_workflow, build_user_prompt
        workflow_factory = partial(build_baseline_workflow, model=args.model)
        prompt_builder = build_user_prompt
    elif args.experiment:
        from evaluation.experiment import build_experiment_workflow, build_user_prompt
        workflow_factory = partial(build_experiment_workflow, model=args.model)
        prompt_builder = build_user_prompt

    from evaluation.evaluate import run_experiment

    workflow_type = "baseline" if args.baseline else "experiment"
    run_experiment(
        template_iri=args.target_schema,
        input_dir=args.input,
        output_dir=args.output,
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


if __name__ == "__main__":
    main()
