# Agentic Real-Time Metadata Standardization (ARMS)

This repository is the **code and supplementary material** for the paper:

> **Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent.**
> Josef Hardi, Martin J. O'Connor, Marcos Martínez-Romero, Jean G. Rosario, Stephen A. Fisher, Mark A. Musen.
> arXiv: https://arxiv.org/abs/2604.08552

ARMS is an LLM agent that standardizes legacy biomedical metadata records into the [CEDAR](https://metadatacenter.org/) template format. Instead of treating ontology constraints as static text in a prompt, the agent calls external services at inference time — fetching the live CEDAR template and querying BioPortal for canonical ontology terms — through [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) tools. This repository contains the agent, the evaluation framework, and the data used to produce every number and figure in the paper.

## Component Source Code

### The Agentic Real-Time Metadata Standardization (ARMS) agent

| Component | Location |
|---|---|
| Agent graph (ReAct, LangGraph) | `src/metadata_standardization_agent/agent.py` |
| The three MCP tools (`get_cedar_template`, `term_search_from_ontology`, `term_search_from_branch`) | `src/metadata_standardization_agent/tools.py` |
| ARMS system prompt | `src/metadata_standardization_agent/prompts.py` |
| Baseline user prompt | `evaluation/baseline.py` |
| Agent prompt builder | `evaluation/experiment.py` |

Both conditions use temperature 0; the agent's output is normalized by a fixed GPT-4.1-mini step with strict JSON-schema decoding (`src/metadata_standardization_agent/utils.py`).

### The experiment dataset

| Component | Location |
|---|---|
| Expert-curated gold standard | `data/<assay>/gold/` |
| Legacy input records | `data/<assay>/input/`|
| Baseline predictions output | `data/<assay>/output/<model>/baseline/` |
| ARMS predictions output | `data/<assay>/output/<model>/experiment/` |
| CEDAR template specifications (one per assay) | `data/schemas/<assay>.json` |
| Sampling function (stratified, per-assay random sample) | `data/sampling.py` |

The evaluation set is 839 records across 12 assay types, sampled independently within each assay (up to 100 per assay; assays with fewer curated records included in full). See `data/sampling.py` for the exact procedure.

### The evaluation metrics and analysis

| What it produces | Location |
|---|---|
| Exact-match accuracy metrics; per-field results | `evaluation/metrics.py` |
| Per-assay and overall accuracy tables (Table 2) | `evaluation/data_analysis.py` |
| Confidence intervals dan statistical tests (Wilcoxon, McNemar) | `evaluation/significance.py` |
| Error-cause and error-type quantification | `evaluation/error_causes.py` |
| Grouped bar charts with bootstrap 95% CI error bars (Figures 2–4) | `evaluation/plots.py` |
| End-to-end analysis notebook | `evaluation/demo.ipynb` |

## Reproducing the Paper's Results

All analysis runs on the prediction files already in `data/.../output/` — **no LLM API calls are needed** to reproduce the accuracy numbers, confidence intervals, significance tests, or error breakdowns.

```bash
uv sync --all-extras
```

### Notebook (recommended)

`evaluation/demo.ipynb` walks through, for a chosen model (`gpt5mini` or `gpt41mini`):

1. Per-assay and overall accuracy (Table 2).
2. **Confidence intervals and significance tests** — bootstrap 95% CIs, paired Wilcoxon (per record), and paired McNemar (per field), per category and pooled.
3. Grouped bar charts (Figures 2–4) with 95% CI error bars.
4. **Error-cause and error-type quantification** for the baseline and ARMS.

### Command line

```bash
cd evaluation

# Confidence intervals + significance tests (overall and per assay):
uv run python significance.py --data-root ../data --model gpt5mini
```

### Running the ARMS agent experiment (requires API keys)

To regenerate predictions (this calls the OpenAI, CEDAR, and BioPortal APIs), create a `.env` file with `OPENAI_API_KEY`, `CEDAR_API_KEY`, and `BIOPORTAL_API_KEY`, then:

```bash
uv run python -m metadata_standardization_agent \
  --input data/atacseq/input \
  --target-schema https://repo.metadatacenter.org/templates/dd5e8653-81cf-470b-b71b-15cab421bb84 \
  --output data/atacseq/output/gpt5mini/experiment \
  --model gpt-5-mini --concurrent 8 --experiment
```

## Models Evaluated

The primary model is **GPT-5-mini**. The **GPT-4.1-mini** is reported as a secondary analysis. Predictions for both are under `data/<assay>/output/{gpt5mini,gpt41mini}/`.

## Development

```bash
uv run python -m pytest                                   # tests
uv run ruff check src/ tests/ evaluation/                 # lint
uv run ruff format src/ tests/ evaluation/                # format
```

## License

BSD 2-Clause License.
