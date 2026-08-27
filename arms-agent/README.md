# arms-agent

An LLM agent that standardizes legacy biomedical metadata records into the
[CEDAR](https://metadatacenter.org/) template format.

Most tools put ontology constraints in the prompt as static text. This agent looks them
up while it runs. It fetches the live CEDAR template and queries BioPortal for canonical
terms through [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
tools, so the constraints it applies are the ones the template holds right now.

This is the agent described in *Automated Standardization of Legacy Biomedical Metadata
Using an Ontology-Constrained LLM Agent* ([arXiv:2604.08552](https://arxiv.org/abs/2604.08552)).
The evaluation harness, the 839-record dataset, and the code for every figure in the paper
live in the [project repository](https://github.com/musen-lab/metadata-standardization-agent).

## Install

```bash
pip install arms-agent
```

## Configure

Three keys are required. Put them in the environment, or in a `.env` file in the
directory you run from:

```
OPENAI_API_KEY=...       # LLM calls
CEDAR_API_KEY=...        # fetching CEDAR templates
BIOPORTAL_API_KEY=...    # ontology term lookups
```

Optional: set `OPENAI_BASE_URL` to route LLM calls through an OpenAI-compatible gateway.
Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to trace each LLM call, tool call, and
agent step to [Langfuse](https://langfuse.com/); tracing stays off until both are set.

## Command line

```bash
arms-migrate \
  --input legacy-record.json \
  --target-schema https://repo.metadatacenter.org/templates/dd5e8653-81cf-470b-b71b-15cab421bb84 \
  --output migrated.json \
  --model gpt-5-mini
```

`--output` takes a file or a directory. Given a directory, the filename comes from the
input. Add `--debug` for step-by-step logging on stderr.

## Python

```python
import asyncio, json

from langchain_core.messages import HumanMessage

from arms_agent.agent import build_migration_agent, build_response_format
from arms_agent.prompts import SYSTEM_PROMPT
from arms_agent.tools import all_tools
from arms_agent.workflow import build_workflow

template_iri = "https://repo.metadatacenter.org/templates/dd5e8653-81cf-470b-b71b-15cab421bb84"
legacy = json.load(open("legacy-record.json"))

agent = build_migration_agent(
    model="gpt-5-mini",
    system_prompt=SYSTEM_PROMPT,
    response_format=build_response_format(template_iri),
    tools=all_tools,
    reasoning_effort="high",
)

result = asyncio.run(
    build_workflow(agent).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Migrate the following legacy metadata record to the CEDAR template.\n\n"
                        f"CEDAR Template IRI: {template_iri}\n\n"
                        f"Legacy metadata:\n```json\n{json.dumps(legacy, indent=2)}\n```"
                    )
                )
            ],
            "cedar_template_iri": template_iri,
        },
        config={"recursion_limit": 30},
    )
)
print(json.dumps(result["metadata"], indent=2))
```

The agent answers against a JSON schema built from the template, so the result conforms to
the template's field structure. When a model answers without a validated object, a fixed
extraction step parses the text into one.

## Caching

CEDAR template and BioPortal term responses are cached in SQLite for 24 hours, to keep
repeated runs fast and off the rate limits. Override with `ARMS_CACHE_DIR` and
`ARMS_CACHE_TTL_SECONDS`.

## License

BSD 2-Clause.
