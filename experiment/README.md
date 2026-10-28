# ARMS RNA-seq experiments via the Langfuse SDK

This unpublished harness uses **local SDK items**, not a hosted Langfuse dataset.
All implementation, tests and dependency configuration live here. It reuses
`evaluation/conditions/agent_tool/arms.py` (the ARMS workflow and user prompt) and
`evaluation/analysis/metrics/` without changing agent behavior or matching rules.

## Prepare

No relabeling, conversion or manual dataset upload is needed for the current
corpus: `data/rnaseq/input/` and `data/rnaseq/gold/` contain 100 matching JSON
filenames. Keep the existing gold records unchanged. The loader checks every pair
and nonempty JSON object **before** applying a limit, so a five-record smoke cannot
hide a broken pair elsewhere. Selection is deterministic, sorted by filename;
it is a wiring smoke, not a representative quality estimate.

Run commands below from the repository root (Python 3.12 and `uv` required).
The `--with-requirements` overlay pins **langfuse==4.18.0**, the newest published
4.x release verified during implementation. It is compatible with the published
agent's `langfuse>=4.14,<5.0` tracing extra. The root lock stays unchanged at 4.14.2;
no published dependency, root configuration or lock migration is necessary.
Always use the overlay for this runner; live execution checks its SDK version.

```bash
# Offline preparation check: validates the entire corpus, makes no network calls.
uv run --frozen --with-requirements experiment/requirements.txt \
  python -m experiment.runner --validate-only

# Offline tests, including an actual 4.18.0 SDK contract test with synthetic tasks,
# disabled tracing and socket connections forbidden. No model or service calls.
uv run --frozen --extra dev --with-requirements experiment/requirements.txt \
  python -m pytest experiment/tests
uv run --frozen --extra dev ruff check experiment/
uv run --frozen --extra dev ruff format --check experiment/
```

Package installation may access the package index; the offline validation and
tests themselves do not access Langfuse, the LLM, CEDAR or BioPortal. The new tests
must be selected explicitly: the repository's default pytest `testpaths` do not
include `experiment/tests`.

## Configure and run the approved five-record smoke

The remaining intervention is operational: choose the model available at your
endpoint and ensure valid credentials are configured in your shell or a local
`.env` file. Never paste secrets into chat or commit them. Authentication has not
been verified by the offline checks. Required variable names:

- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and explicitly configured
  `LANGFUSE_HOST` (or `LANGFUSE_BASE_URL`, normalized to `LANGFUSE_HOST`).
- `CEDAR_API_KEY`, `BIOPORTAL_API_KEY` for ARMS' existing template/vocabulary tools.
- `OPENAI_API_KEY` for non-Ollama models. Set `OPENAI_BASE_URL` (or the agent's
  existing `OPENAI_API_BASE` alias) when using a gateway; otherwise ARMS uses OpenAI.
- For `ollama:<tag>`, use a running, appropriately configured Ollama server and
  optional `OLLAMA_HOST`, per the agent docs; no OpenAI key is required for that path.

Do not set `LANGFUSE_TRACING_ENABLED=false` for a live experiment. The runner
checks configuration and Langfuse authentication before constructing ARMS, but
cannot pre-authenticate every model/tool endpoint. Existing shell variables take
precedence over `--env-file`; no `.env` is silently loaded by this runner.

```bash
# Replace the model with the exact identifier supported by your endpoint.
uv run --frozen --with-requirements experiment/requirements.txt \
  python -m experiment.runner --env-file .env \
  --model '<your-configured-model>' --limit 5 --max-concurrency 1 \
  --run-name rnaseq-smoke
```

**Privacy/cost:** “local data” means the corpus originates on disk, not that the
experiment is private/offline. The SDK sends inputs, expected outputs, predictions,
metadata, evaluator scores and traces to the configured Langfuse project. ARMS
also sends legacy metadata to the selected model and uses CEDAR/BioPortal tools.
Gold is available to deterministic evaluators and SDK tracing, **never** to the
agent's prompt, workflow state or model input. Confirm project/data permissions
before executing. No hosted dataset create/upsert/upload operation is used.

The runner prints JSON containing scores/coverage only (not raw corpus records).
A failed task or missing item/run evaluator result makes the CLI exit nonzero,
even though the SDK isolates and logs individual failures. Keep the terminal logs
for diagnostics; review them before sharing because SDK/provider errors can
contain request details. `--max-concurrency` defaults to one and must be positive.
ARMS retains its existing reasoning/sampling settings, schema binding and graph
recursion limit; the adapter does not introduce retries or change prompts.

Review the smoke's coverage, traces, model/tool access, latency and cost first.
**Do not run the full corpus until separately approved.** The explicit command
for that later step is:

```bash
uv run --frozen --with-requirements experiment/requirements.txt \
  python -m experiment.runner --env-file .env \
  --model '<your-configured-model>' --all --max-concurrency 1 \
  --run-name rnaseq-full
```

Use `--data-dir /absolute/path/to/data` for another copy with the same layout.
There is no resumption/cache of predictions in this runner: rerunning repeats
model work and may incur new cost. Neither smoke nor full-corpus model execution
was performed during implementation.

## Item contract and scores

Each local item has:

- `input = {legacy_metadata, template_iri}` with the existing RNA-seq IRI
  `https://repo.metadatacenter.org/templates/944e5fa0-f68b-4bdd-8664-74a3909429a9`.
- `expected_output` = the unmodified parsed gold JSON object.
- `metadata` = assay, filename-derived record ID, corpus-relative input/gold paths
  and SHA-256 content hashes. Run metadata also records model, template, schema
  hash, selection, SDK version and strict matching configuration.

Item evaluators return `precision`, `recall`, `f1`, `all_field_accuracy`,
`ontology_constrained_accuracy`, `non_ontology_constrained_accuracy`.
They call the existing functions with their strict defaults (`match_case=True`,
`match_whole_word=True`). This retains existing DOI URL normalization and blank
semantics; there is no LLM judge, substring matching or added normalization.

The run evaluator sums existing raw TP/FP/FN counts before calling the existing
`precision_recall_f1` function to produce `micro_precision`, `micro_recall`,
`micro_f1`. A substitution contributes both FP and FN. Both-blank fields earn
accuracy credit but not precision/recall credit. Accuracy divides summed correct
field counts by summed gold field counts, overall and by ontology/non-ontology
category (`field_weighted_*_accuracy`). These are **not** record-level means.
Only fields in gold are evaluated, as in the existing harness; extra prediction
keys are not a new penalty. Empty denominators score 0.0 under existing definitions.

`attempted_records` means the selected item count; `completed_records` counts
returned, fully scored items; `failed_records` includes task failures and incomplete
item evaluations. Ratios cover completed items only. An incomplete run is not a
valid quality comparison: inspect coverage and rerun after fixing the cause.

### Version-specific API verification and reporting caveat

Current [experiment docs](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)
now describe Python 5.x. Rather than migrate the published package, this runner
was checked against the authoritative [4.18.0 distribution](https://pypi.org/project/langfuse/4.18.0/):
`langfuse/_client/client.py:Langfuse.run_experiment` accepts local dictionaries,
async tasks, `evaluators`, `run_evaluators`, `max_concurrency`, and run metadata.
`langfuse/experiment.py` defines `Evaluation` and `ExperimentItemResult` with
`item`, `output`, and `evaluations`. Tests exercise this exact installed SDK.

In **4.18.0**, item scores are sent with traces, but the SDK's run-score upload
branch requires a hosted `dataset_run_id`. For local items, aggregates are
computed and returned in `ExperimentResult.run_evaluations` and printed by this
CLI; do **not** assume those aggregate scores are persisted as hosted dataset-run
scores in the UI. Save stdout if needed (for example, redirect to an approved
output location outside the corpus). Langfuse flushes are performed after SDK
execution and again in `finally`; success means complete local computation, not
independent verification that every trace was ingested remotely.
