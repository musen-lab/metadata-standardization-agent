# Agentic Real-Time Metadata Standardization (ARMS)

This repository is the **code and supplementary material** for the paper:

> **Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent.**
> Josef Hardi, Martin J. O'Connor, Marcos Martínez-Romero, Jean G. Rosario, Stephen A. Fisher, Mark A. Musen.
> arXiv: https://arxiv.org/abs/2604.08552

ARMS is an LLM agent that standardizes legacy biomedical metadata records into the [CEDAR](https://metadatacenter.org/) template format. Instead of treating ontology constraints as static text in a prompt, the agent calls external services at inference time — fetching the live CEDAR template and querying BioPortal for canonical ontology terms — through [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) tools. This repository contains the agent, the evaluation framework, and the data used to produce every number and figure in the paper.

## Experiment Code and Data Analysis

### The Agentic Real-Time Metadata Standardization (ARMS) agent

The agent is a standalone package under `arms-agent/`, published to PyPI as
[`arms-agent`](https://pypi.org/project/arms-agent/) and installable on its own. The
repository root is the evaluation harness that measures it. See
[arms-agent/README.md](arms-agent/README.md) to use the agent outside this repository.

| Component | Location |
|---|---|
| Agent graph (ReAct, LangGraph) | `arms-agent/src/arms_agent/agent.py` |
| The three MCP tools (`get_cedar_template`, `term_search_from_ontology`, `term_search_from_branch`) | `arms-agent/src/arms_agent/tools.py` |
| ARMS system prompt | `arms-agent/src/arms_agent/prompts.py` |
| Prompt-only condition (`baseline`) | `evaluation/conditions/prompt_only/`, one module each |
| Prompt-only system prompts | `evaluation/conditions/prompt_only/prompts/`, one module each |
| Agent prompt builder (`agent-tool`) | `evaluation/conditions/agent_tool/arms.py` |

Both conditions use temperature 0; the agent's output is normalized by a fixed GPT-4.1-mini step with strict JSON-schema decoding (`arms-agent/src/arms_agent/utils.py`).

### The experiment dataset

| Component | Location |
|---|---|
| Expert-curated gold standard | `data/<assay>/gold/` |
| Legacy input records | `data/<assay>/input/`|
| Prompt-only predictions output (one directory per condition) | `data/<assay>/output/<model>/baseline/` |
| ARMS predictions output (directory named by `--agent-name`) | `data/<assay>/output/<model>/arms-agent/` |
| CEDAR template specifications (one per assay) | `data/schemas/<assay>.json` |
| Sampling function (stratified, per-assay random sample) | `data/sampling.py` |

The evaluation set is 839 records across 12 assay types, sampled independently within each assay (up to 100 per assay; assays with fewer curated records included in full). See `data/sampling.py` for the exact procedure.

### The evaluation metrics and analysis

| What it produces | Location |
|---|---|
| Exact-match accuracy metrics; per-field results | `evaluation/analysis/metrics/` |
| Per-assay and overall accuracy tables (Table 2) | `evaluation/analysis/data_analysis/` |
| Confidence intervals and statistical tests (Wilcoxon, McNemar) | `evaluation/analysis/significance/` |
| Grouped bar charts with bootstrap 95% CI error bars (Figures 2–4) | `evaluation/plots/` |
| End-to-end analysis notebook | `experiment.ipynb` |

## Reproducing the Paper's Results

All analysis runs on the prediction files already in `data/.../output/` — **no LLM API calls are needed** to reproduce the accuracy numbers, confidence intervals, significance tests, or error breakdowns.

```bash
uv sync --all-extras
```

### Notebook (recommended)

`experiment.ipynb`, in the repository root, walks through, for a chosen model (`gpt5mini` or `gpt41mini`):

1. Per-assay and overall accuracy (Table 2).
2. **Confidence intervals and significance tests** — bootstrap 95% CIs, paired Wilcoxon (per record), and paired McNemar (per field), per category and pooled.
3. Grouped bar charts (Figures 2–4) with 95% CI error bars.
4. **Surface error-type breakdown** for the baseline and ARMS (`data_analysis.create_error_report`).

### Command line

```bash
# Confidence intervals + significance tests (overall and per assay):
uv run python -m evaluation.analysis.significance --data-root data --model gpt5mini
```

### Running the ARMS agent experiment (requires API keys)

To regenerate predictions (this calls the OpenAI, CEDAR, and BioPortal APIs), create a `.env` file with `OPENAI_API_KEY`, `CEDAR_API_KEY`, and `BIOPORTAL_API_KEY`, then:

```bash
uv run python -m evaluation \
  --input data/atacseq/input \
  --target-schema https://repo.metadatacenter.org/templates/dd5e8653-81cf-470b-b71b-15cab421bb84 \
  --output data/atacseq/output/gpt5mini \
  --model gpt-5-mini --concurrent 8 --agent-tool arms-agent
```

### Tracing agent runs (optional)

Runs can be traced to [Langfuse](https://langfuse.com/) to inspect each LLM call, MCP tool call, and agent step. Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` to `.env`; tracing activates only when both keys are set and is otherwise a no-op. Set `LANGFUSE_TRACING_ENVIRONMENT` (or pass `--langfuse-environment` to the evaluation CLI) to separate sweeps from each other within a project. See [.env.example](.env.example).

## Models Evaluated

The primary model is **GPT-5-mini**. The **GPT-4.1-mini** is reported as a secondary analysis. Predictions for both are under `data/<assay>/output/{gpt5mini,gpt41mini}/`.

## Development

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): the
root holds the evaluation harness, and `arms-agent/` holds the published package. One
`uv sync --all-extras` sets up both, and the agent is installed in editable mode, so edits
under `arms-agent/src/` take effect at once.

```bash
uv run python -m pytest                                   # tests
uv run ruff check arms-agent/ tests/ evaluation/          # lint
uv run ruff format arms-agent/ tests/ evaluation/         # format
```

## License

BSD 2-Clause License.
