"""CLI entry point for the evaluation framework.

Usage::

    evaluate --input <dir> --target-schema <iri> --output <parent-dir> \
        [--model MODEL] [--concurrent N] [--langfuse-environment NAME] \
        (--prompt-only [CONDITION] | --agent-tool [AGENT_NAME]) \
        [--debug]

The value given to ``--prompt-only`` / ``--agent-tool`` names the run: it tags the trace
and is the subdirectory of ``--output`` the predictions are written to.
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
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Parent directory for migrated output files.  The run writes to the subdirectory of it "
        "named by whichever of --prompt-only / --agent-tool was given.",
    )
    # One flag picks the workflow *and* names the run, so the two can never disagree and
    # neither mode carries a naming option the other silently ignores.  Both default to
    # None when absent, which is what distinguishes "not given" from "given bare".
    workflow_group = parser.add_mutually_exclusive_group(required=True)
    workflow_group.add_argument(
        "--prompt-only",
        nargs="?",
        const="baseline",
        choices=["baseline"],
        help="Use the prompt-only call workflow, under the named condition: what the prompt "
        "spells out of the template -- 'baseline' its field and vocabulary names, and no tools "
        "(default: baseline).",
    )
    workflow_group.add_argument(
        "--agent-tool",
        nargs="?",
        const="arms-agent",
        metavar="AGENT_NAME",
        help="Use the agent tool call workflow, under the named agent, e.g. 'arms-agent' (default: arms-agent).",
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

    from conditions import build_condition

    # --agent-tool names the run rather than choosing one, since ARMS is the only agent:
    # any name given to it runs ARMS and names the directory it writes to.
    condition = args.prompt_only if args.prompt_only is not None else "arms-agent"
    build_workflow, prompt_builder = build_condition(condition)
    # The template is passed at build time so the answer can be validated against it.
    workflow_factory = partial(build_workflow, model=args.model, template_iri=args.target_schema)
    logging.getLogger(__name__).info("Running condition: %s", condition)

    from evaluate import run_experiment

    # Names the on-disk output directory (data/<assay>/output/<model>/<workflow_type>/)
    # and is matched verbatim by the modules under analysis/: the value given to whichever
    # of --prompt-only / --agent-tool selected the workflow above.
    workflow_type = args.prompt_only if args.prompt_only is not None else args.agent_tool
    output_dir = args.output / workflow_type
    logging.getLogger(__name__).info("Writing output to %s", output_dir)
    run_experiment(
        template_iri=args.target_schema,
        input_dir=args.input,
        output_dir=output_dir,
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
