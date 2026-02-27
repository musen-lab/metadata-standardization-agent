# Evaluation Framework

Measures the quality of agent-migrated metadata against gold-standard references.

## Running

```bash
python -m evaluation <template_iri> <input_dir> <output_dir> <gold_dir> <report_path> \
    [--model MODEL] [--concurrent N] [--langsmith-project NAME] \
    (--baseline | --experiment) \
    [--debug]
```

| Argument | Description |
|------|-------------|
| `template_iri` | IRI of the CEDAR template to migrate to |
| `input_dir` | Directory containing input JSON files |
| `output_dir` | Directory to write migrated output JSON files |
| `gold_dir` | Directory containing gold standard JSON files |
| `report_path` | Path for the evaluation report file |

| Flag | Description |
|------|-------------|
| `--baseline` | Using a single LLM call without tool access |
| `--experiment` | Using an agent with tool access |
| `--model MODEL` | GPT model variant: `gpt-4.1`, `gpt-4.1-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (default: `gpt-4.1-mini`) |
| `--concurrent N` | Max number of concurrent file evaluations (default: `5`) |
| `--langsmith-project NAME` | LangSmith project name (overrides `.env` setting) |
| `--debug` | Enable debug logging to stderr |

One of `--baseline` or `--experiment` is required.

This will:
1. Run the migration workflow on each JSON file in `<input_dir>`.
2. Write outputs to `<output_dir>`.
3. Compare outputs against gold-standard files in `<gold_dir>`.
4. Write a CSV report to `<report_path>`.

## Metrics

### Field Completeness

Proportion of gold-standard non-missing fields that are also present (non-missing)
in the predicted output. Only field presence is checked; values are not compared.

```
completeness = |non_missing(gold) ∩ non_missing(predicted)| / |non_missing(gold)|
```

A field is "missing" if its value is `None`. Empty strings and empty lists count as
present.

### Field-Value Correctness

Among fields that are non-missing in **both** predicted and gold, the fraction with
matching values.

```
comparable   = {k : k ∈ gold, gold[k] ≠ None, predicted[k] ≠ None}
correctness  = |{k ∈ comparable : matches(predicted[k], gold[k])}| / |comparable|
```

By default, `matches` is exact equality. Two optional parameters relax matching:

| Parameter | Default | Effect |
|---|---|---|
| `match_case` | `True` | When `False`, string values are lowercased before comparison. Non-strings are unaffected. |
| `match_whole_word` | `True` | When `False`, the gold value only needs to be a **substring of** the predicted value (for strings). Non-strings are unaffected. |

Both parameters can be combined (e.g. case-insensitive substring matching). With
defaults `(True, True)`, behaviour is identical to strict exact match.

### Record Accuracy

Overall record-level agreement across all fields in the gold standard. Unlike
correctness, which only considers fields that are non-missing in both records,
accuracy evaluates every field. Two fields agree when both values are missing
(`None`), or both are non-missing and match. Any difference in value or presence
counts as a mismatch.

```
accuracy = |{k ∈ gold : agree(predicted[k], gold[k])}| / |gold|
```

where `agree(p, g)` is true when both are `None`, or both are non-`None` and
`matches(p, g)`. The same `match_case` and `match_whole_word` parameters apply.

## Directory Conventions

Input, output, and gold-standard directories use matching filenames:

```
data/input/record1.json       ← legacy metadata
data/output/record1.json      ← agent output
evaluation/gold_standard/record1.json  ← gold reference
```

Files without a gold-standard counterpart are skipped during evaluation.

## CSV Report Format

| Column        | Description                              |
|---------------|------------------------------------------|
| input_file    | Filename of the evaluated record         |
| correctness   | Field-value correctness for that record  |
| completeness  | Field completeness for that record       |
| accuracy      | Record accuracy for that record          |
