# Evaluation Framework

Measures the quality of agent-migrated metadata against gold-standard references.

## Running

```bash
python -m evaluation <input_dir> <template_iri> <output_dir> <gold_dir> <report_path> \
    (--baseline | --experiment) [--debug]
```

| Flag | Description |
|------|-------------|
| `--baseline` | Using a single LLM call without tool access |
| `--experiment` | Using an agent with tool access |
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

### Field-Value Accuracy

Among fields that are non-missing in **both** predicted and gold, the fraction with
matching values.

```
comparable  = {k : k ∈ gold, gold[k] ≠ None, predicted[k] ≠ None}
accuracy    = |{k ∈ comparable : matches(predicted[k], gold[k])}| / |comparable|
```

By default, `matches` is exact equality. Two optional parameters relax matching:

| Parameter | Default | Effect |
|---|---|---|
| `match_case` | `True` | When `False`, string values are lowercased before comparison. Non-strings are unaffected. |
| `match_whole_word` | `True` | When `False`, the gold value only needs to be a **substring of** the predicted value (for strings). Non-strings are unaffected. |

Both parameters can be combined (e.g. case-insensitive substring matching). With
defaults `(True, True)`, behaviour is identical to strict exact match.

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
| accuracy      | Field-value accuracy for that record     |
| completeness  | Field completeness for that record       |
