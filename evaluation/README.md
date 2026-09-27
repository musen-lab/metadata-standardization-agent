# Evaluation Framework

Measures the quality of agent-predicted metadata against gold-standard references.

## Getting Started

The recommended way to run evaluations and explore results is the `experiment.ipynb` notebook in the repository root. It provides an interactive workflow for:

- Running the method evaluations (prompt-only and ARMS) across all assay types
- Computing per-assay and overall accuracy summaries
- Plotting grouped bar charts comparing baseline vs ARMS
- Generating error analysis reports

Open the notebook and follow the configuration cells to set your `DATA_ROOT`, `MODEL`, `ASSAYS` and `CONDITIONS`. Run it from the repository root: its setup cell puts this directory on the import path so the modules below can be imported by name.

Running the experiments is two calls, both from `[sweep.py](sweep.py)`:

```python
from sweep import plan_sweep, run_sweep

plan = plan_sweep(DATA_ROOT, MODEL, assays=ASSAYS, conditions=CONDITIONS)
run_sweep(plan, dry_run=False)             # one run, into each <condition>/
run_sweep(plan, n_repeat=5, dry_run=False) # the whole plan five times, into <condition>/run-1/ .. run-5/
run_sweep(plan, n_repeat=2, first_run=2, dry_run=False)  # two more beside an existing run-1: run-2/, run-3/
```

`plan_sweep` loads the API keys from `.env` and prints what the sweep covers. It raises on an unknown assay, an unknown condition, an assay with no input records, or a missing key — before anything is spent. `run_sweep` then runs the jobs (one job is one assay under one condition), one at a time, every condition of one assay before the next assay starts. It spends nothing while `dry_run` stands, which is its default.
A single run writes each job straight into its condition's directory, as the CLI does.
With `n_repeat=N` above 1 it makes N runs of the whole plan instead, finishing each run before starting the next, and writes run *n* to `<condition>/run-<n>/`.
`first_run` numbers new runs after ones already on disk.
Nothing already written is overwritten unless you pass `overwrite=True`: a sweep that would write where predictions already are is refused before it spends anything, and names the `first_run` that follows the last run there.
The CLI follows the same rule, with `--overwrite`.
A condition directory holds one layout or the other: `run_sweep` refuses, before spending anything, to write one run where `run-<n>` directories already are, or several where a single run already is.
Every analysis function reads run 1 unless given `run=<n>`, finding it in whichever layout the condition has, for example `create_overall_precision_recall_summary(DATA_ROOT, MODEL, "arms-agent", run=2)`.
Two views read several runs at once: `create_run_spread_summary(DATA_ROOT, MODEL, "arms-agent", runs=(1, 2, 3))` gives precision and recall per assay as the mean with the lowest and highest run (`notebook_utils.show_run_spread` prints it for several conditions), and `plot_field_stability(DATA_ROOT, MODEL, runs=(1, 2, 3))` shows, per assay, how often each condition gives the same answer to a field in every run.

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

Gold-standard and output files share the same filenames so that each output can be matched to its reference for evaluation.

## Metrics

Three accuracy metrics are computed by `analysis/metrics/`:

### Ontology-Constrained Field Accuracy (`ontology_constrained_field_accuracy`)

Accuracy restricted to fields whose values must come from a controlled ontology or branch-based permissible-value list (as defined in the schema). Only those fields are evaluated; all others are ignored.

### Non-Ontology-Constrained Field Accuracy (`non_ontology_constrained_field_accuracy`)

Accuracy restricted to free-text and other fields that are **not** ontology-constrained. This is the complement of the ontology-constrained subset.

### Match Parameters

All three metrics accept two optional parameters that relax string matching:

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
    [--overwrite] [--debug]
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
| `--overwrite` | Replace predictions already in `DIR/<condition>/`. Without it the run is refused before it starts; a folder holding a sweep's `run-<n>` folders is refused either way |
| `--debug` | Enable debug logging to stderr |
