"""Evaluation framework for the metadata migration agent."""

from __future__ import annotations

from evaluation.evaluate import apply_metrics, run_experiment, run_experiment_workflow
from evaluation.metrics import compute_completeness, compute_precision

__all__ = [
    "apply_metrics",
    "compute_completeness",
    "compute_precision",
    "run_experiment",
    "run_experiment_workflow",
]
