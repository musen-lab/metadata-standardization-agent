"""Evaluation framework for the metadata migration agent."""

from __future__ import annotations

from evaluation.evaluate import apply_metrics, execute_workflow, run_experiment
from evaluation.metrics import compute_accuracy, compute_completeness

__all__ = [
    "apply_metrics",
    "compute_accuracy",
    "compute_completeness",
    "run_experiment",
    "execute_workflow",
]
