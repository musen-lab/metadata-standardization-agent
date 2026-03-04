# Metadata Standardization Agent

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
cd metadata-standardization-agent

# Install dependencies
uv sync --all-extras

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Usage

### CLI

```bash
uv run python -m metadata_standardization_agent \
  --input data/input/my_record.json \
  --target-schema https://repo.metadatacenter.org/templates/TEMPLATE_ID \
  --output data/output/migrated.json \
  --debug  # optional: enable debug logging
```

### Python API

```python
from metadata_standardization_agent.agent import app
from metadata_standardization_agent.state import AgentState
from metadata_standardization_agent.utils import load_json

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

## License

MIT
