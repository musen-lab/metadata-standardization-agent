"""Entry point for running the metadata migration agent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from metadata_migration_agent.agent import build_agent
from metadata_migration_agent.logging_config import configure_logging

# Load environment variables from .env (project root)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env", override=True)

logger = logging.getLogger("metadata_migration_agent.__main__")


def main() -> None:
    """Run the migration agent with a legacy metadata record and CEDAR template IRI."""
    parser = argparse.ArgumentParser(
        description="Migrate a legacy metadata record to a CEDAR template format.",
    )
    parser.add_argument("legacy_metadata", help="Path to the legacy metadata JSON file.")
    parser.add_argument("cedar_template_iri", help="IRI of the CEDAR template to migrate to.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    args = parser.parse_args()

    configure_logging(args.debug)

    logger.debug("Legacy metadata file: %s", args.legacy_metadata)
    logger.debug("CEDAR template IRI: %s", args.cedar_template_iri)

    with open(args.legacy_metadata) as f:
        legacy_metadata = json.load(f)

    user_message = (
        f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
        f"CEDAR Template IRI: {args.cedar_template_iri}\n\n"
        f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
    )

    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content=user_message)],
        "cedar_template_iri": args.cedar_template_iri,
    })
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
