"""Entry point for running the metadata standardization agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

from arms_agent.agent import build_migration_agent, build_response_format
from arms_agent.logging_config import configure_logging
from arms_agent.prompts import SYSTEM_PROMPT
from arms_agent.token_tracker import TokenUsageTracker
from arms_agent.tools import all_tools
from arms_agent.tracing import flush_tracing, instrument, traced_run
from arms_agent.workflow import build_workflow

# Load environment variables from the nearest .env, searching upward from the working
# directory.  Counting parent directories from this file would only work inside a source
# checkout; installed from a wheel, the package sits in site-packages and has no project
# root above it.  find_dotenv returns "" when there is no .env, and load_dotenv("") is a
# no-op, so a run without one is fine.
load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger("arms_agent.__main__")


def main() -> None:
    """Run the migration agent with a legacy metadata record and CEDAR template IRI."""
    parser = argparse.ArgumentParser(
        description="Migrate a legacy metadata record to a CEDAR template format.",
    )
    parser.add_argument("--input", required=True, help="Path to the legacy metadata JSON file.")
    parser.add_argument("--target-schema", required=True, help="IRI of the CEDAR template to migrate to.")
    parser.add_argument(
        "--output",
        help="Output file path or directory. If a directory, the output filename is derived from the input. "
        f"(default: {Path(tempfile.gettempdir()) / 'migrated-metadata.json'})",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.6-terra",
        help="LLM model identifier (default: gpt-5.6-terra).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    configure_logging(args.debug)

    logger.debug("Legacy metadata file: %s", args.input)
    logger.debug("CEDAR template IRI: %s", args.target_schema)

    with open(args.input) as f:
        legacy_metadata = json.load(f)

    user_message = (
        f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
        f"CEDAR Template IRI: {args.target_schema}\n\n"
        f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
    )

    agent = build_migration_agent(
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
        response_format=build_response_format(args.target_schema),
        tools=all_tools,
        reasoning_effort="high",
        reasoning_mode="standard",
    )
    workflow = build_workflow(agent)
    tracker = TokenUsageTracker()
    input_stem = Path(args.input).stem
    run_metadata = {
        "input_file": Path(args.input).name,
        "template_iri": args.target_schema,
        "model": args.model,
    }
    run_config = instrument(
        {
            "recursion_limit": 30,
            "callbacks": [tracker],
            "run_name": f"migrate-{input_stem}",
            "tags": ["cli", "migrate"],
            "metadata": run_metadata,
        }
    )
    start = time.perf_counter()
    try:
        with traced_run(f"migrate-{input_stem}", run_metadata):
            result = asyncio.run(
                workflow.ainvoke(
                    {
                        "messages": [HumanMessage(content=user_message)],
                        "cedar_template_iri": args.target_schema,
                    },
                    config=run_config,
                )
            )
    finally:
        # Traces of a failed run are the ones worth keeping, so flush either way.
        flush_tracing()
    elapsed = time.perf_counter() - start
    if args.output is None:
        output_path = Path(tempfile.gettempdir()) / "migrated-metadata.json"
    elif Path(args.output).is_dir():
        output_path = Path(args.output) / f"{input_stem}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result["metadata"], indent=2) + "\n")
    print(f"Output written to: {output_path}")

    decisions_path = output_path.with_suffix(".decisions.json")
    decisions_path.write_text(json.dumps(result.get("decisions") or [], indent=2) + "\n")
    print(f"Processing log written to: {decisions_path}")
    print(f"Execution time: {elapsed:.2f}s")
    print(tracker.usage_summary())


if __name__ == "__main__":
    main()
