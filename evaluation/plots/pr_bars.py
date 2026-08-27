"""The same precision and recall as :mod:`plots.pr_space`, read as heights rather than position.

A bar answers "how much" at a glance where a scatter answers "which way the trade-off
went", so this is the figure for a reading that has to carry a single number per run.  It
shares its rows and its argument checks with the scatter, in :mod:`plots.pr_scores`, so
the two cannot come to disagree about which assays there are.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plots.marks import CONDITION_COLOURS, FIELD_TYPE_LABELS, LADDER_BLUES, LADDER_ORANGES, _run_colours
from plots.pr_scores import _check_pr_arguments, _pr_scores
from plots.theme import AXIS_COLOUR, GRID_COLOUR, LEGEND_LINE_INCHES, _finish


def plot_pr_bar_chart(
    data_root: str,
    model: str,
    *,
    assays: tuple[str, ...] = (),
    baseline_runs: tuple[str, ...] = ("baseline",),
    system_runs: tuple[str, ...] = ("arms-agent",),
    field_type: str = "all",
    save_path: str | None = None,
) -> None:
    """Precision above, recall below, one bar per run within each assay.

    The two metrics get a panel each, stacked and sharing both the x axis and the y
    span, so an assay's precision sits directly above its recall and the two are read
    against one scale.  A metric is a panel rather than a hatch, which leaves colour
    free to carry the run, exactly as it does in :func:`~plots.pr_space.plot_pr_space`.

    *baseline_runs* and *system_runs* may each hold several runs -- every repetition of
    one condition, say -- and each run gets its own bar in both panels, drawn in the order
    given.  The gap inside an assay's group falls between the two groups, so each reads
    as a block.  Colour is the run: one hue per group, in monotone lightness steps when
    a group has several runs.

    *assays* names the assays to draw, by the keys ``ASSAY_ORDER`` uses; **left empty,
    the corpus is pooled into a single group of bars**.  *field_type* takes exactly one
    of ``"ontology"``, ``"non_ontology"`` or ``"all"``, since a bar's height is one
    number and the panels are already spent on the metric.  When *save_path* is given
    the figure is written there (PNG/PDF inferred from the extension) instead of shown.
    """
    _check_pr_arguments(baseline_runs, system_runs, (field_type,))
    runs = (*baseline_runs, *system_runs)
    rows, scores = _pr_scores(data_root, model, runs, field_type, assays)

    colours = [
        *_run_colours(len(baseline_runs), LADDER_BLUES, CONDITION_COLOURS[0]),
        *_run_colours(len(system_runs), LADDER_ORANGES, CONDITION_COLOURS[1]),
    ]

    # Slots inside one assay: the runs of a group sit shoulder to shoulder, with half a
    # slot of air between the groups so each group reads as a block.
    group_gap = 0.5
    slots = [float(index) for index in range(len(baseline_runs))]
    slots += [len(baseline_runs) + group_gap + index for index in range(len(system_runs))]
    span = slots[-1] + 1.0
    width = min(0.8 / span, 0.32)
    offsets = [(slot - (span - 1.0) / 2.0) * width for slot in slots]

    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(6.0, 1.5 * len(rows) + 2.0), 6.4), sharex=True, sharey=True)

    for ax, metric, index in ((axes[0], "precision", 1), (axes[1], "recall", 0)):
        for run, colour, offset in zip(runs, colours, offsets, strict=True):
            values = [scores[(row, run)][index] for row in rows]
            # 0.94 of the slot leaves a hairline of page between adjacent bars.
            ax.bar(x + offset, values, width * 0.94, color=colour, label=run)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", color=GRID_COLOUR, linewidth=0.6)
        ax.set_axisbelow(True)  # grid behind the bars rather than across them
        ax.tick_params(color=AXIS_COLOUR)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS_COLOUR)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rows, rotation=45 if len(rows) > 1 else 0, ha="right" if len(rows) > 1 else "center")
    fig.suptitle(f"Score, {FIELD_TYPE_LABELS[field_type]}", fontsize=11)

    keys = [Patch(facecolor=colour, label=run) for run, colour in zip(runs, colours, strict=True)]
    line = LEGEND_LINE_INCHES / fig.get_size_inches()[1]
    fig.legend(handles=keys, loc="lower center", ncol=len(keys), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.0, 1.2 * line, 1.0, 1.0))
    _finish(fig, save_path)
