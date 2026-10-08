# Evaluation Framework

Measures the quality of agent-predicted metadata against reference-standard records.

## Getting Started

The recommended way to run evaluations and explore results is the `experiment.ipynb` notebook in the repository root. It provides an interactive workflow for:

- Running the method evaluations (prompt-only and ARMS) across all assay types
- Computing per-assay and overall accuracy summaries
- Plotting grouped bar charts comparing baseline vs ARMS
- Generating error analysis reports

Open the notebook and follow the configuration cells to set your `DATA_ROOT`, `MODEL`, `ASSAYS` and `CONDITIONS`. Run it from the repository root: its setup cell puts this directory on the import path so the modules below can be imported by name.

```python
from sweep import plan_sweep, run_sweep

plan = plan_sweep(DATA_ROOT, MODEL, assays=ASSAYS, conditions=CONDITIONS)
run_sweep(plan, dry_run=False)             # one run, into each <condition>/
run_sweep(plan, n_repeat=5, dry_run=False) # the whole plan five times, into <condition>/run-1/ .. run-5/
run_sweep(plan, n_repeat=2, first_run=2, dry_run=False)  # two more beside an existing run-1: run-2/, run-3/
```

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
│           │   ├── atacseq-<hash>.json   # Run once: the predictions sit here
│           │   └── ...
│           ├── template-tool/            # Ablation: template fetch, no term search
│           │   └── ...
│           ├── term-tool/                # Ablation: term search, no template fetch
│           │   └── ...
│           └── arms-agent/               # Agent tool: both
│               ├── run-1/                # Run several times: one directory per run
│               │   ├── atacseq-<hash>.json
│               │   └── ...
│               └── run-2/ ...
├── lcms/
│   ├── input/ ...
│   ├── gold/ ...
│   └── output/ ...
└── ...
```

Reference-standard and output files share the same filenames so that each output can be matched to its reference for evaluation.

## Metrics

`analysis/metrics/` computes two families of metrics over the fields of each reference record. Both judge a field the same way: `null`, `""` and whitespace-only strings count as blank, and a field missing from the prediction counts as blank.

### Accuracy

Accuracy is the fraction of reference fields where the prediction agrees with reference records.

| Function | Fields scored |
|---|---|
| `compute_all_field_accuracy(predicted, gold)` | Every field in the gold record. |
| `compute_ontology_constrained_field_accuracy(predicted, gold, schema_path)` | Fields whose values must come from a controlled ontology or branch-based permissible-value list, as defined in the schema. |
| `compute_non_ontology_constrained_field_accuracy(predicted, gold, schema_path)` | Free-text and other fields that are **not** ontology-constrained, the complement of the subset above. |

### Precision and Recall

`compute_field_confusion(predicted, gold, schema_path)` sorts every gold field into confusion-matrix counters, where the positive action is asserting a value.

| Gold | Prediction | Counted as |
|---|---|---|
| blank | blank | `TN` |
| blank | a value | `FP` (insertion) |
| a value | blank | `FN` (deletion) |
| a value | the matching value | `TP` |
| a value | a different value | `FP` and `FN` (substitution) |

`precision_recall_f1(counts)` turns one category's counts into scores:

- **Precision** is `TP / (TP + FP)`, the fraction of asserted values that were right.
- **Recall** is `TP / (TP + FN)`, the fraction of gold values that were produced.

### Match Parameters

All three accuracy functions and `compute_field_confusion` accept two optional keyword parameters that relax string matching:

| Parameter | Default | Effect |
|---|---|---|
| `match_case` | `True` | When `False`, string values are lowercased before comparison. Non-strings are unaffected. |
| `match_whole_word` | `True` | When `False`, the gold value only needs to be a **substring of** the predicted value (for strings). Non-strings are unaffected. |

Both parameters can be combined (e.g. case-insensitive substring matching). With defaults `(True, True)`, behaviour is identical to strict exact match.

## Adding a Condition

A condition is a Python module that declares a `CONDITION`. Put it in any folder under `conditions/`. The CLI, the sweep and the notebook find it automatically; there is no list to edit.

This is `conditions/agent_tool/arms.py`, shortened:

```python
def build_agent_tool_workflow(model: str, template_iri: str | None = None) -> CompiledStateGraph:
    return build_workflow(
        build_migration_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            response_format=build_response_format(template_iri) if template_iri else None,
            tools=all_tools,
            reasoning_effort="high",
            reasoning_mode="standard",
        )
    )

def build_user_prompt(legacy_metadata: dict[str, Any], template_iri: str) -> str:
    return f"Metadata template IRI: {template_iri}\n\nLegacy metadata record:\n{json.dumps(legacy_metadata)}"

CONDITION = Condition(
    name="arms-agent",
    build_workflow=build_agent_tool_workflow,
    build_user_prompt=build_user_prompt,
    requires_keys=("BIOPORTAL_API_KEY",),
    order=100,
)
```

The parameters of `Condition`:

- `name`: what `--condition` and `CONDITIONS` take, and the name of the output directory. It must be unique across all conditions.
- `build_workflow`: a function taking `model` and `template_iri` that returns the compiled graph to run.
- `build_user_prompt`: a function taking the legacy record and the template IRI that returns the user message. It may be imported from another condition.
- `requires_keys` (optional): environment variables the condition needs beyond `OPENAI_API_KEY` and `CEDAR_API_KEY`. `plan_sweep` stops before any run if one is missing.
- `order` (optional): where the condition appears in tables and plots. The shipped conditions use 0, 40, 60 and 100.

Put a new system prompt in the family's `prompts/` directory, and add it to `PROMPTS` in `tests/test_condition_prompts.py` so the tests check it keeps the shared policy.

In a notebook kernel that is already running, call `discover(refresh=True)` from `conditions` after adding a module.

## CLI

You can also run standardizations from the command line:

```bash
python -m evaluation --input <dir> --target-schema <iri> --output <parent-dir> \
    --condition CONDITION \
    [--model MODEL] [--concurrent N] [--langfuse-environment NAME] \
    [--overwrite [--yes]] [--debug]
```

| Flag | Description |
|------|-------------|
| `--input DIR` | Directory containing input JSON files |
| `--target-schema IRI` | IRI of the CEDAR template to standardize to |
| `--output DIR` | Parent directory for the migrated output JSON files. The run writes to `DIR/<condition>/` |
| `--condition CONDITION` | Which condition to run. The choices are the modules declared under `conditions/`, so a module dropped in is offered here without this flag changing |
| `--model MODEL` | GPT model variant: `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol` (default: `gpt-5.6-terra`) |
| `--concurrent N` | Max number of concurrent file evaluations (default: `5`) |
| `--langfuse-environment NAME` | Langfuse tracing environment to file this run under (overrides `.env` setting) |
| `--overwrite` | Replace predictions already in `DIR/<condition>/`. It first says how many it would replace and asks to confirm; anything but `y` stops the run. Without it the run is refused before it starts; a folder holding a sweep's `run-<n>` folders is refused either way |
| `--yes` | Answer yes to `--overwrite`'s confirmation, for a script with no one to ask. Without it, a script's closed input counts as no |
| `--debug` | Enable debug logging to stderr |
