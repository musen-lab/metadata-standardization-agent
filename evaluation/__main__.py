"""CLI entry point for the evaluation framework.

Usage::

    evaluate --input <dir> --target-schema <iri> --output <parent-dir> \
        --condition CONDITION [--run-name NAME] \
        [--model MODEL] [--concurrent N] [--langfuse-environment NAME] \
        [--debug]

``--condition`` takes any condition declared under ``conditions/``; the list is read
from there rather than written down here, so a module dropped in is offered without
this file changing.  The run is named after it, unless ``--run-name`` says otherwise:
that name tags the trace and is the subdirectory of ``--output`` the predictions are
written to.
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
    # Read before the parser is built: the conditions are the flag's choices, and its
    # help text lists them.
    from conditions import condition_names, get_condition

    known = condition_names()

    parser = argparse.ArgumentParser(
        description="Batch-run the migration workflow and evaluate against gold standards.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing input JSON files.")
    parser.add_argument("--target-schema", required=True, help="IRI of the CEDAR template to migrate to.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Parent directory for migrated output files.  The run writes to the subdirectory of it named by the run.",
    )
    # The choices are the declared conditions, so a name outside them is refused here --
    # before the input is read and before anything is spent.
    parser.add_argument(
        "--condition",
        required=True,
        choices=known,
        help="The condition to run.  Each is a module under conditions/ that declares itself; "
        f"the module says which family it belongs to and what keys it needs.  One of: {', '.join(known)}.",
    )
    parser.add_argument(
        "--run-name",
        metavar="NAME",
        help="What to call this run: the subdirectory of --output it writes to, and the tag its "
        "trace carries (default: the condition's own name).  Name a run to hold a repeat of one "
        "condition beside the first rather than over it.",
    )
    gpt_models = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    parser.add_argument(
        "--model",
        choices=gpt_models,
        default="gpt-5.6-terra",
        help="GPT model variant (default: gpt-5.6-terra).",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Max number of concurrent file evaluations (default: 5).",
    )
    parser.add_argument(
        "--langfuse-environment",
        type=str,
        default=None,
        help="Langfuse tracing environment to file this run under, e.g. 'histology-gpt5mini'",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.langfuse_environment:
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = args.langfuse_environment

    condition = get_condition(args.condition)
    # The template is passed at build time so the answer can be validated against it.
    workflow_factory = partial(condition.build_workflow, model=args.model, template_iri=args.target_schema)

    from evaluate import run_experiment

    # Names the on-disk output directory (data/<assay>/output/<model>/<run_name>/) and is
    # matched verbatim by the modules under analysis/.  It is the condition's name unless
    # --run-name overrode it, which is how a repeat run is kept beside the first.
    run_name = args.run_name or condition.name
    output_dir = args.output / run_name
    logging.getLogger(__name__).info("Running condition %s as %s", condition.name, run_name)
    logging.getLogger(__name__).info("Writing output to %s", output_dir)
    run_experiment(
        template_iri=args.target_schema,
        input_dir=args.input,
        output_dir=output_dir,
        workflow_factory=workflow_factory,
        user_prompt_builder=condition.build_user_prompt,
        max_concurrency=args.concurrent,
        config={
            "tags": ["evaluation", run_name],
            "metadata": {
                "template_iri": args.target_schema,
                "workflow_type": run_name,
                "condition": condition.name,
            },
        },
    )


if __name__ == "__main__":
    main()
