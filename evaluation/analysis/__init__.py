"""Offline analysis of prediction files: scoring, statistics, and figures.

Nothing here calls an LLM.  Everything reads the prediction files already under
``data/<assay>/output/`` and the gold standard beside them, so the paper's tables and
figures can be reproduced without API keys.  :mod:`analysis.metrics` scores individual
records, :mod:`analysis.data_analysis` aggregates into tables, and
:mod:`analysis.significance` supplies confidence intervals and tests.

Drawing lives outside this package, in :mod:`plots`, which reads the tables the modules
here return.
"""
