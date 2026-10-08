# Agentic Real-Time Metadata Standardization (ARMS)

This repository is the **code and supplementary material** for the paper:

> **Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent.**
> Josef Hardi, Martin J. O'Connor, Marcos Martínez-Romero, Jean G. Rosario, Stephen A. Fisher, Mark A. Musen.
> arXiv: https://arxiv.org/abs/2604.08552

ARMS is an LLM agent that standardizes legacy biomedical metadata records to conform to a [CEDAR](https://metadatacenter.org/) template.  Instead of treating ontology constraints as static text in a prompt, the agent calls external services at inference time—fetching the live CEDAR template and querying BioPortal for standardized ontology terms—through [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) tools. This repository contains the agent, evaluation framework, and data used to produce every number and figure reported in the paper.

## Experiment Code and Data Analysis

### The Agentic Real-Time Metadata Standardization (ARMS) agent

The agent is a standalone package under `arms-agent/`, published to PyPI as [arms-agent](https://pypi.org/project/arms-agent/) and installable on its own. The repository root is the evaluation harness that measures it. See its [README.md](arms-agent/README.md) to use the agent outside this repository.

| Component | Location |
|---|---|
| Agent graph (ReAct, LangGraph) | `arms-agent/src/arms_agent/agent.py` |
| The three MCP tools (`get_cedar_template`, `term_search_from_ontology`, `term_search_from_branch`) | `arms-agent/src/arms_agent/tools.py` |
| ARMS system prompt | `arms-agent/src/arms_agent/prompts.py` |

### ARMS Evaluation

| Component | Location |
|---|---|
| Expert-curated reference standard | `data/<assay>/gold/` |
| Legacy input records | `data/<assay>/input/`|
| Baseline output | `data/<assay>/output/<model>/baseline/` |
| ARMS output | `data/<assay>/output/<model>/arms-agent/` |
| Ablation outputs | `data/<assay>/output/<model>/{template-tool,term-tool}/` |

Repeated conditions store predictions under `run-<n>/` inside their condition directory.

The evaluation set is 839 records across 12 assay types, sampled independently within each assay (up to 100 per assay; assays with fewer curated records included in full). `data/sampling.py` provides a utility for sampling paired input and gold records.

### The evaluation metrics and analysis

| What it produces | Location |
|---|---|
| Exact-match accuracy, precision, recall, and per-field results | `evaluation/analysis/metrics/` |
| Per-assay and pooled tables, deduplicated scoring, error analysis, and run spread | `evaluation/analysis/data_analysis/` |
| Bootstrap confidence intervals, paired permutation tests, and deduplicated comparisons | `evaluation/analysis/significance/` |
| Precision/recall comparisons, run spread, field stability, and model comparison plots | `evaluation/plots/` |
| End-to-end analysis notebook | `experiment.ipynb` |

## Reproducing the Paper's Results

All analysis runs on the prediction files already in the `data` folder. **No API keys or LLM API calls are needed** to reproduce the precision and recall, confidence intervals, significance tests, error breakdowns, or run stability results.

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then set up the workspace:

```bash
uv sync --all-extras
```

Open `experiment.ipynb` from the repository root using the `.venv` Python kernel. Run the setup and configuration cells in **A. Configurations**, then the cells from **Data Analysis** onward. Skip **B. Run Experiments** for offline analysis: its `plan_sweep` calls require API keys even when `RUN_EXPERIMENT=False`, and can register model prices in Langfuse when tracing is enabled.

Before running the model comparison plots, define their assay selection in a separate configuration cell (the notebook normally defines it in **B. Run Experiments**):

```python
MODEL_COMP_ASSAYS = ["codex", "histology", "imc-2d", "lightsheet"]
```

The notebook compares baseline and ARMS, the two ablations, repeated runs, and other LLM models. It also compares scores after deduplicating repeated values. See [evaluation/README.md](evaluation/README.md) for the evaluation interfaces and directory conventions.

### Running the ARMS agent experiment (requires API keys)

To regenerate predictions (this calls the OpenAI, CEDAR, and BioPortal APIs), create a `.env` file with `OPENAI_API_KEY`, `CEDAR_API_KEY`, and `BIOPORTAL_API_KEY`, then run a single condition into a fresh output directory:

```bash
uv run python -m evaluation \
  --input data/atacseq/input \
  --target-schema https://repo.metadatacenter.org/templates/dd5e8653-81cf-470b-b71b-15cab421bb84 \
  --output runs/atacseq/output/gpt-5.6-terra \
  --model gpt-5.6-terra --concurrent 8 --condition arms-agent
```

This writes predictions under `runs/atacseq/output/gpt-5.6-terra/arms-agent/`. The CLI refuses to run if a prediction directory already exists unless `--overwrite` is given.

### Tracing agent runs (optional)

Runs can be traced to [Langfuse](https://langfuse.com/) to inspect each LLM call, tool calls, and agent steps. Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` to `.env`; tracing activates only when both keys are set and is otherwise a no-op. Set `LANGFUSE_TRACING_ENVIRONMENT` (or pass `--langfuse-environment` to the evaluation CLI) to separate sweeps from each other within a project. See [.env.example](.env.example).

## Development

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): the root holds the evaluation harness, and `arms-agent/` holds the published package. One `uv sync --all-extras` sets up both, and the agent is installed in editable mode, so edits under `arms-agent/src/` take effect at once.

```bash
uv run python -m pytest                                   # tests
uv run ruff check arms-agent/ tests/ evaluation/          # lint
uv run ruff format arms-agent/ tests/ evaluation/         # format
```

## License

BSD 2-Clause License.
