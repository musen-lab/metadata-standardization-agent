# Evaluation Framework

Measures the quality of agent-predicted metadata against gold-standard references.

## Getting Started

The recommended way to run evaluations and explore results is the **`experiment.ipynb`** notebook in the repository root. It provides an interactive workflow for:

- Running the method evaluations (prompt-only and agent-tool) across all assay types
- Computing per-assay and overall accuracy summaries
- Plotting grouped bar charts comparing baseline vs agent-tool
- Generating error analysis reports

Open the notebook and follow the configuration cells to set your `DATA_ROOT`, `MODEL`, `ASSAYS` and `RUN_TYPES`. Run it from the repository root: its setup cell puts this directory on the import path so the modules below can be imported by name.

Running the experiments is two calls, both from [`sweep.py`](sweep.py):

```python
from sweep import plan_sweep, run_sweep

plan = plan_sweep(DATA_ROOT, MODEL, assays=ASSAYS, run_types=RUN_TYPES)
run_sweep(plan, dry_run=False)
```

`plan_sweep` loads the API keys from `.env` and prints what the sweep covers. It raises on an unknown assay, an unknown condition, an assay with no input records, or a missing key — before anything is spent. `run_sweep` then makes the runs, one at a time, every condition of one assay before the next assay starts. It spends nothing while `dry_run` stands, which is its default.

## Directory Conventions

The evaluation functions expect the following directory structure underneath `DATA_ROOT`:

```
DATA_ROOT/
├── schemas/
│   ├── atacseq.json              # JSON Schema for each assay type
│   ├── lcms.json
│   └── ...
├── atacseq/                      # One directory per assay type
│   ├── input/
│   │   ├── atacseq-<hash>.json   # Legacy metadata records (input)
│   │   └── ...
│   ├── gold/
│   │   ├── atacseq-<hash>.json   # Gold-standard reference outputs
│   │   └── ...
│   └── output/
│       └── <MODEL>/              # e.g., "gpt5mini"
│           ├── baseline/                 # Prompt-only: field and vocabulary names
│           │   ├── atacseq-<hash>.json
│           │   └── ...
│           └── arms-agent/               # Agent tool: named by --agent-name
│               ├── atacseq-<hash>.json
│               └── ...
├── lcms/
│   ├── input/ ...
│   ├── gold/ ...
│   └── output/ ...
└── ...
```

Gold-standard and output files share the same filenames so that each output can be matched to its reference for evaluation.

## Metrics

Three accuracy metrics are computed by `analysis/metrics/`:

### Ontology-Constrained Field Accuracy (`ontology_constrained_field_accuracy`)

Accuracy restricted to fields whose values must come from a controlled ontology or branch-based permissible-value list (as defined in the schema). Only those fields are evaluated; all others are ignored.

### Non-Ontology-Constrained Field Accuracy (`non_ontology_constrained_field_accuracy`)

Accuracy restricted to free-text and other fields that are **not** ontology-constrained. This is the complement of the ontology-constrained subset.

### All-Field Accuracy (`all_field_accuracy`)

Record-level agreement across all fields in the gold standard. Two fields agree when both values are missing (`null`), or both are non-missing and match. The denominator is all keys present in gold.

```
accuracy = |{k ∈ gold : agree(predicted[k], gold[k])}| / |gold|
```

### Match Parameters

All three metrics accept two optional parameters that relax string matching:

| Parameter | Default | Effect |
|---|---|---|
| `match_case` | `True` | When `False`, string values are lowercased before comparison. Non-strings are unaffected. |
| `match_whole_word` | `True` | When `False`, the gold value only needs to be a **substring of** the predicted value (for strings). Non-strings are unaffected. |

Both parameters can be combined (e.g. case-insensitive substring matching). With defaults `(True, True)`, behaviour is identical to strict exact match.

## Adding a Condition

A condition is a module under `conditions/prompt_only/` or `conditions/agent_tool/` that declares itself. Drop the file in and the CLI, the sweep and the notebook all see it; no list anywhere needs editing, because no list exists.

```python
# conditions/prompt_only/schema_vocab.py
from conditions.registry import Condition


def build_schema_vocab_workflow(model: str, template_iri: str | None = None) -> CompiledStateGraph: ...
def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str: ...


CONDITION = Condition(
    name="schema+vocab",                    # what the CLI takes, and the output directory
    build_workflow=build_schema_vocab_workflow,
    build_user_prompt=build_user_prompt,
    requires_keys=("BIOPORTAL_API_KEY",),   # optional: checked before a sweep spends anything
    order=20,                               # optional: where it sits in the reported order
)
```

The name is declared rather than read off the filename because the two need not agree — `schema+vocab` is not a legal module name. `requires_keys` is declared because only the condition knows what it calls out to; `plan_sweep` collects it from every condition in the sweep and stops on a missing key before any run starts. A module that declares no `CONDITION` is not one: `prompt_only/template_spec.py` is the material the prompts are built from, and the registry passes over it.

A directory added beside `prompt_only/` and `agent_tool/` becomes a third family the same way, with no code change.

## CLI

You can also run standardizations from the command line:

```bash
python -m evaluation --input <dir> --target-schema <iri> --output <parent-dir> \
    (--prompt-only [CONDITION] | --agent-tool [AGENT_NAME]) \
    [--model MODEL] [--concurrent N] [--langfuse-environment NAME] \
    [--debug]
```

| Flag | Description |
|------|-------------|
| `--input DIR` | Directory containing input JSON files |
| `--target-schema IRI` | IRI of the CEDAR template to standardize to |
| `--output DIR` | Parent directory for the migrated output JSON files. The run writes to `DIR/<run name>/`, where the run name is the value given to the workflow flag |
| `--prompt-only [CONDITION]` | Use the prompt-only workflow (single LLM call) under the named condition: `baseline` (default: `baseline`) |
| `--agent-tool [AGENT_NAME]` | Use the agent tool workflow (ReAct agent) under the named agent, e.g. `arms-agent` (default: `arms-agent`) |
| `--model MODEL` | GPT model variant: `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol` (default: `gpt-5.6-terra`) |
| `--concurrent N` | Max number of concurrent file evaluations (default: `5`) |
| `--langfuse-environment NAME` | Langfuse tracing environment to file this run under (overrides `.env` setting) |
| `--debug` | Enable debug logging to stderr |

One of `--prompt-only` or `--agent-tool` is required, and the value it is given names the run: it tags the Langfuse trace and is the subdirectory of `--output` the predictions land in. This will run the standardization workflow on each JSON file in the input directory.
