"""The paper's figures, drawn from the tables :mod:`analysis.data_analysis` returns.

Nothing here reads a prediction file directly except
:func:`~plots.accuracy_bars.plot_grouped_bar_chart`, the oldest of them.  The rest take a
finished table and decide only how it is shown, which is what keeps a figure and the
number it draws from disagreeing.

The package is layered.  Underneath are three modules that no figure owns:

* :mod:`plots.theme` -- the page: the greys, the sizes of type, and how a figure is
  finished.
* :mod:`plots.marks` -- how a run and a field type become a mark, so a run is the same
  colour and a field type the same shape in every figure either appears in.
* :mod:`plots.segments` -- how a share becomes a segment of a stacked bar, and how a
  segment too narrow to hold its own number gets one anyway.

On top of those sits one module per figure, each holding the public function, the
constants that place it, and nothing else:

* :mod:`plots.pr_space` -- operating points in precision/recall space, the main figure.
* :mod:`plots.pr_bars` -- the same measurement as heights rather than as position.
* :mod:`plots.availability` -- where migration is transcription and where it is
  interpretation.
* :mod:`plots.error_composition` -- what the errors are made of, for the corpus and per
  assay.
* :mod:`plots.accuracy_bars` -- per-record accuracy per assay, with error bars.

The two precision/recall figures share :mod:`plots.pr_scores`, which decides what the rows
are and what the pooled row is called, so they cannot come to disagree about it.

This module re-exports the figures, so ``from plots import ...`` reaches every one of them
regardless of which module draws it.
"""

from __future__ import annotations

from plots.accuracy_bars import plot_grouped_bar_chart
from plots.availability import plot_availability_difficulty
from plots.error_composition import plot_corpus_error_composition, plot_error_composition
from plots.pr_bars import plot_pr_bar_chart
from plots.pr_space import plot_pr_space

__all__ = [
    "plot_availability_difficulty",
    "plot_corpus_error_composition",
    "plot_error_composition",
    "plot_grouped_bar_chart",
    "plot_pr_bar_chart",
    "plot_pr_space",
]
