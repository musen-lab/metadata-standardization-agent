"""Entry point for running the metadata migration agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from metadata_migration_agent.logging_config import configure_logging
from metadata_migration_agent.token_tracker import TokenUsageTracker
from metadata_migration_agent.workflow import build_workflow

# Load environment variables from .env (project root)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env", override=True)

logger = logging.getLogger("metadata_migration_agent.__main__")


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

    workflow = build_workflow()
    tracker = TokenUsageTracker()
    input_stem = Path(args.input).stem
    start = time.perf_counter()
    result = asyncio.run(
        workflow.ainvoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "cedar_template_iri": args.target_schema,
            },
            config={
                "recursion_limit": 30,
                "callbacks": [tracker],
                "run_name": f"migrate-{input_stem}",
                "tags": ["cli", "migrate"],
                "metadata": {
                    "input_file": Path(args.input).name,
                    "template_iri": args.target_schema,
                },
            },
        )
    )
    elapsed = time.perf_counter() - start
    if args.output is None:
        output_path = Path(tempfile.gettempdir()) / "migrated-metadata.json"
    elif Path(args.output).is_dir():
        output_path = Path(args.output) / f"{input_stem}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result["metadata"], indent=2) + "\n")
    logger.info("Output written to: %s", output_path)
    logger.info("Execution time: %.2fs", elapsed)
    logger.info(tracker.usage_summary())


if __name__ == "__main__":
    main()
