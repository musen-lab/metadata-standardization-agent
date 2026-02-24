"""Entry point for running the metadata migration agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from metadata_migration_agent.agent import build_agent

# Load environment variables from .env (project root)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env", override=True)


def main() -> None:
    """Run the migration agent with a legacy metadata record and CEDAR template IRI."""
    if len(sys.argv) < 3:
        print("Usage: python -m metadata_migration_agent <legacy_metadata.json> <cedar_template_iri>")
        sys.exit(1)

    metadata_path = sys.argv[1]
    template_iri = sys.argv[2]

    with open(metadata_path) as f:
        legacy_metadata = json.load(f)

    user_message = (
        f"Migrate the following legacy metadata record to the CEDAR template.\n\n"
        f"CEDAR Template IRI: {template_iri}\n\n"
        f"Legacy metadata:\n```json\n{json.dumps(legacy_metadata, indent=2)}\n```"
    )

    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    # Print the final assistant response
    for message in reversed(result["messages"]):
        if message.type == "ai" and message.content:
            print(message.content)
            break


if __name__ == "__main__":
    main()
