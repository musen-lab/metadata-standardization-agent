# Metadata Standardization Agent

A LangGraph-based agent for standardizing legacy metadata records into
[CEDAR](https://metadatacenter.org/) metadata template format.

## Overview

This project provides an AI agent that:
- Accepts legacy metadata records (JSON / JSON-LD) and a target CEDAR template IRI
- Uses a ReAct agent (LangGraph + OpenAI) with tools for CEDAR API access and BioPortal ontology search
- Produces standardized metadata that conforms to the CEDAR template structure

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone the repository
git clone <repo-url>
cd metadata-standardization-agent

# Install dependencies
uv sync --all-extras
```

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=sk-...
CEDAR_API_KEY=...
BIOPORTAL_API_KEY=...
```

## Usage

### CLI

```bash
uv run python -m metadata_standardization_agent \
  --input data/input/my_record.json \
  --target-schema https://repo.metadatacenter.org/templates/TEMPLATE_ID \
  --output data/output/standardized.json \
  --debug  # optional: enable debug logging
```

## Evaluation

An evaluation framework for measuring precision and stability of the agent output
against gold standard reference data is available in the `evaluation/` directory.

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/ evaluation/
uv run ruff format src/ tests/ evaluation/
```

## License

BSD 2-Clause License
