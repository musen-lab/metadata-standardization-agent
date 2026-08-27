"""System prompts, one module per prompt-only condition.

The condition keeps its own copy on purpose: the shared policy has to read identically
everywhere, and only ``tests/test_condition_prompts.py`` stops the copies drifting
apart.  ARMS's own prompt is not one of these; it lives in :mod:`arms_agent.prompts`.
"""
