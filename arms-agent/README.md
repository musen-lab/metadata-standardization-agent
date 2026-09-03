# ARMS Agent

An LLM agent that standardizes legacy biomedical metadata records to adhere to the [CEDAR](https://metadatacenter.org/) template.

It fetches the live CEDAR template and queries BioPortal for canonical terms through [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) tools, so the constraints it applies are the ones the template holds right now.

This is the agent described in *Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent* ([arXiv:2604.08552](https://arxiv.org/abs/2604.08552)). The evaluation harness, the experiment dataset, and the code for the data analysis between baseline vs ARMS agent in the [project repository](https://github.com/musen-lab/metadata-standardization-agent).

## Install

```bash
pip install arms-agent
```

## Configure

Three keys are required. Put them in the environment, or in a `.env` file in the directory you run from:

```
OPENAI_API_KEY=...       # LLM calls
CEDAR_API_KEY=...        # fetching CEDAR templates
BIOPORTAL_API_KEY=...    # ontology term lookups
```

Optional: set `OPENAI_BASE_URL` to route LLM calls through an OpenAI-compatible gateway.

To trace each LLM call, tool call, and agent step to [Langfuse](https://langfuse.com/), install the extra and
set both keys:

```bash
pip install 'arms-agent[tracing]'
```

```
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...       # optional, defaults to Langfuse Cloud
```

Tracing stays off until both keys are set, and `LANGFUSE_TRACING_ENABLED=false` switches it off while leaving
the keys in place.

## Command line

```bash
arms-migrate \
  --input legacy-metadata.json \
  --target-schema https://repo.metadatacenter.org/templates/[CEDAR-TEMPLATE-UUID] \
  --output standardized-metadata.json \
  --model gpt-5-mini
```

`--output` takes a file or a directory. Given a directory, the filename comes from the input (default: the
system's temp directory. `--model` defaults to `gpt-5.6-terra`. Add `--debug` for step-by-step
logging on stderr.

## Integration in Python

```python
import asyncio, json

from langchain_core.messages import HumanMessage

from arms_agent.agent import build_migration_agent, build_response_format
from arms_agent.prompts import SYSTEM_PROMPT
from arms_agent.tools import all_tools
from arms_agent.workflow import build_workflow

template_iri = "https://repo.metadatacenter.org/templates/[CEDAR-TEMPLATE-UUID]"
legacy = json.load(open("legacy-metadata.json"))

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
                        "Standarize the legacy metadata record to adhere to the CEDAR template.\n\n"
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

The agent answers against a JSON schema built from the template, so the result conforms to the template's field structure. When a model answers without a validated object, a fixed extraction step parses the text into one.

## Caching

CEDAR template and BioPortal term responses are cached in SQLite for 24 hours, to keep repeated runs fast and off the rate limits. Override with `ARMS_CACHE_DIR` and `ARMS_CACHE_TTL_SECONDS` in the .env file. 

## Other settings

| Variable | Default | What it does |
| --- | --- | --- |
| `OPENAI_EXTRACTION_MODEL` | `gpt-4.1-mini` | The model that parses a reply into an object when the main model answers without one. |
| `OPENAI_COST_MULTIPLIER` | `1.0` | Scales the reported cost when your endpoint charges a fraction of OpenAI's list prices. |
| `OPENAI_COST_CACHE_DISCOUNT` | `true` | Whether the endpoint discounts cached input tokens. |

Costs are local estimates from provider-reported token counts, not billed amounts.

## License

BSD 2-Clause.
