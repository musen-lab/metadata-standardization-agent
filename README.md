# Metadata Migration Agent

A LangGraph-based agent for migrating legacy metadata records to
[CEDAR](https://metadatacenter.org/) metadata template format.

## Overview

This project provides an AI agent that:
- Accepts legacy metadata records (JSON / JSON-LD) and a target CEDAR template (JSON-LD)
- Uses an LLM (GPT-4o) to analyze, map, and transform metadata fields
- Produces migrated metadata that conforms to the CEDAR template structure

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone the repository
git clone <repo-url>
cd metadata-migration-agent

# Install dependencies
uv sync --all-extras

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Usage

```python
from metadata_migration_agent.agent import app
from metadata_migration_agent.state import AgentState
from metadata_migration_agent.utils import load_json

state = AgentState(
    legacy_metadata=load_json("data/input/my_record.json"),
    cedar_template=load_json("data/templates/my_template.jsonld"),
)
result = app.invoke(state)
print(result["migrated_metadata"])
```

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/ evaluation/
uv run ruff format src/ tests/ evaluation/
```

## Evaluation

The `evaluation/` directory contains tooling for comparing agent output against
gold standard reference data. Two metrics are computed:

- **Precision** -- fraction of output fields that match the gold standard
- **Stability** -- consistency of output across repeated runs with the same input

## Project Structure

```
src/metadata_migration_agent/   Core agent source code
tests/                          Unit and integration tests
data/templates/                 CEDAR template files
data/input/                     Legacy metadata input files
data/output/                    Agent-generated output
evaluation/                     Evaluation scripts and gold standard data
```

## License

MIT
