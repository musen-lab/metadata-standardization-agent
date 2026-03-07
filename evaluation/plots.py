from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assays import ASSAY_ORDER
from data_analysis import apply_metrics


def plot_grouped_bar_chart(
    data_root: str,
    model: str,
    metric: str,
    title: str,
    *,
    show_error_bars: bool = True,
) -> None:
    """Grouped bar chart (baseline vs experiment) with optional min/max error bars."""
    root = Path(data_root)

    # Collect per-assay stats for each condition
    conditions = ["baseline", "experiment"]
    # assay_key -> condition -> {mean, min, max}
    stats: dict[str, dict[str, dict[str, float]]] = {}

    for assay_key, _ in ASSAY_ORDER:
        gold_dir = root / assay_key / "gold"
        schema_path = root / "schemas" / f"{assay_key}.json"
        if not gold_dir.exists() or not schema_path.exists():
            continue

        for condition in conditions:
            input_dir = root / assay_key / "output" / model / condition
            if not input_dir.exists():
                continue

            df = apply_metrics(input_dir, gold_dir, schema_path)
            if df.empty or metric not in df.columns:
                continue

            values = df[metric]
            stats.setdefault(assay_key, {})[condition] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

    # Filter to assays that have data for at least one condition
    ordered = [(k, lbl) for k, lbl in ASSAY_ORDER if k in stats]
    assays = [k for k, _ in ordered]
    assay_labels = [lbl for _, lbl in ordered]
    x = np.arange(len(assays))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (condition, color) in enumerate([("baseline", "#4472C4"), ("experiment", "#ED7D31")]):
        means = np.array([stats[a].get(condition, {}).get("mean", 0.0) for a in assays])
        label = "Baseline" if condition == "baseline" else "Our Method"

        bar_kwargs: dict[str, object] = {
            "width": width,
            "color": color,
            "label": label,
        }
        if show_error_bars:
            mins = np.array([stats[a].get(condition, {}).get("min", 0.0) for a in assays])
            maxs = np.array([stats[a].get(condition, {}).get("max", 0.0) for a in assays])
            err_low = np.maximum(means - mins, 0)
            err_high = np.maximum(maxs - means, 0)
            bar_kwargs["yerr"] = [err_low, err_high]
            bar_kwargs["capsize"] = 3

        ax.bar(x + (i - 0.5) * width, means, **bar_kwargs)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Prediction accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(assay_labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    fig.tight_layout()
    plt.show()
