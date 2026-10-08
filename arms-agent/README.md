# ARMS Agent

An LLM agent that standardizes legacy biomedical metadata records to conform to a [CEDAR](https://metadatacenter.org/) template.

It fetches the live CEDAR template and queries BioPortal for canonical terms through [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) tools, so the constraints it applies are the ones the template holds right now.

This is the agent described in *"Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent"* article ([arXiv:2604.08552](https://arxiv.org/abs/2604.08552)). The evaluation harness, experiment dataset, and analysis comparing baseline and ARMS live in the [project repository](https://github.com/musen-lab/metadata-standardization-agent).

## Install

```bash
pip install arms-agent
```

## Configure

Python 3.12 or later is required. Put these three keys in the environment, or in a `.env` file in the directory you run from:

```
OPENAI_API_KEY=...       # LLM calls
CEDAR_API_KEY=...        # fetching CEDAR templates
BIOPORTAL_API_KEY=...    # ontology term lookups
```

The CLI loads the `.env` and its values override existing environment variables.

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
  --target-schema 'https://repo.metadatacenter.org/templates/[CEDAR-TEMPLATE-UUID]' \
  --output standardized-metadata.json \
  --model gpt-5-mini
```

Replace `[CEDAR-TEMPLATE-UUID]` with the template's UUID. `--output` takes a file path or an existing directory. Given a directory, the filename comes from the input; if omitted, it writes `migrated-metadata.json` in the system's temp directory. The CLI also writes a sibling `<output>.decisions.json` processing log and reports elapsed time, token usage, and estimated cost.

`--model` defaults to `gpt-5.6-terra`. Add `--debug` for step-by-step logging on stderr. The workflow uses `RECURSION_LIMIT=100` to allow models that search terms one at a time to finish.

For a model on another OpenAI-compatible server, set how it reasons and samples:

```bash
arms-migrate ... --model qwen3.8-flash-next-fast \
  --reasoning-effort medium \
  --sampling '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}'
```

`--reasoning-effort` defaults to `high`; the server decides which levels it accepts.
`--sampling` always sends a temperature (default: 0) and sends other settings only when given.
`top_k`, `min_p` and `repetition_penalty` are not OpenAI parameters, so they go in the request body for a server such as vLLM or SGLang to read.

## Integration in Python

```python
import asyncio
import json

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

from arms_agent.agent import build_migration_agent, build_response_format
from arms_agent.prompts import SYSTEM_PROMPT
from arms_agent.tools import all_tools
from arms_agent.workflow import RECURSION_LIMIT, build_workflow

load_dotenv(find_dotenv(usecwd=True), override=True)

template_iri = "https://repo.metadatacenter.org/templates/[CEDAR-TEMPLATE-UUID]"
with open("legacy-metadata.json") as source:
    legacy = json.load(source)

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
                        "Standardize the legacy metadata record to adhere to the CEDAR template.\n\n"
                        f"CEDAR Template IRI: {template_iri}\n\n"
                        f"Legacy metadata:\n```json\n{json.dumps(legacy, indent=2)}\n```"
                    )
                )
            ],
            "cedar_template_iri": template_iri,
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
)
print(json.dumps(result["metadata"], indent=2))
```

The agent answers against a JSON schema built from the template. If it writes the answer as text, the workflow first tries to parse and validate it locally. If needed, it calls an extraction model with a 120-second request timeout and no automatic retries, then validates the extracted record. Invalid records raise an error. The result contains `metadata` and a `decisions` processing log.

## Caching

CEDAR template and BioPortal term responses are cached in SQLite for 24 hours, to keep repeated runs fast and off the rate limits. Override with `ARMS_CACHE_DIR` and `ARMS_CACHE_TTL_SECONDS` in the environment or `.env` file.

## Other settings

| Variable | Default | What it does |
| --- | --- | --- |
| `OPENAI_EXTRACTION_MODEL` | `gpt-4.1-mini` | The model that parses a reply into an object when the main model answers without one. |
| `OPENAI_COST_MULTIPLIER` | `1.0` | Scales the reported cost when your endpoint charges a fraction of OpenAI's list prices. |
| `OPENAI_COST_CACHED_MULTIPLIER` | `OPENAI_COST_MULTIPLIER` | Scales OpenAI's cached-input rate, for an endpoint that discounts cached input differently from the rest. |
| `OPENAI_STREAMING` | `false` | Set to `true` to stream replies, for an endpoint behind a proxy that cuts off slow ones, such as Cloudflare's 100-second limit. |
| `OPENAI_STRUCTURED_OUTPUT` | `provider` | How the template's schema reaches the model: `provider` sends it as the request's `response_format`; `tool` offers it as a tool the model calls to answer, for an endpoint that enforces `response_format` on every reply and so never lets the model call a tool. |

Costs are local estimates from provider-reported token counts, not billed amounts.

`OPENAI_COST_CACHE_DISCOUNT` is no longer used. Replace it with `OPENAI_COST_CACHED_MULTIPLIER`: set it to `1.0` if cached input is charged at OpenAI's full cached-input rate while `OPENAI_COST_MULTIPLIER` discounts other tokens, or leave it unset to apply the same multiplier to all token rates.

## License

BSD 2-Clause.
