# Evaluation Framework

Measures the quality of agent-predicted metadata against gold-standard references.

## Getting Started

The recommended way to run evaluations and explore results is the **`demo.ipynb`** notebook in this directory. It provides an interactive workflow for:

- Running the method evaluations (baseline and experiment) across all assay types
- Computing per-assay and overall accuracy summaries
- Plotting grouped bar charts comparing baseline vs experiment
- Generating error analysis reports

Open the notebook and follow the configuration cells to set your `DATA_ROOT`, `MODEL`, and `RUN_TYPE`.

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
│           ├── baseline/
│           │   ├── atacseq-<hash>.json   # Prompt-only LLM outputs
│           │   └── ...
│           └── experiment/
│               ├── atacseq-<hash>.json   # Tool-augmented agent outputs
│               └── ...
├── lcms/
│   ├── input/ ...
│   ├── gold/ ...
│   └── output/ ...
└── ...
```

Gold-standard and output files share the same filenames so that each output can be matched to its reference for evaluation.

## Metrics

Three accuracy metrics are computed by `metrics.py`:

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

## CLI

You can also run standardizations from the command line:

```bash
python -m evaluation --input <dir> --target-schema <iri> --output <dir> \
    (--baseline | --experiment) \
    [--model MODEL] [--concurrent N] [--langsmith-project NAME] \
    [--debug]
```

| Flag | Description |
|------|-------------|
| `--input DIR` | Directory containing input JSON files |
| `--target-schema IRI` | IRI of the CEDAR template to standardize to |
| `--output DIR` | Directory to write migrated output JSON files |
| `--baseline` | Use the baseline workflow (single LLM call) |
| `--experiment` | Use the experiment workflow (ReAct agent) |
| `--model MODEL` | GPT model variant: `gpt-4.1`, `gpt-4.1-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (default: `gpt-4.1-mini`) |
| `--concurrent N` | Max number of concurrent file evaluations (default: `5`) |
| `--langsmith-project NAME` | LangSmith project name (overrides `.env` setting) |
| `--debug` | Enable debug logging to stderr |

One of `--baseline` or `--experiment` is required. This will run the standardization workflow on each JSON file in the input directory and write outputs to the output directory.
