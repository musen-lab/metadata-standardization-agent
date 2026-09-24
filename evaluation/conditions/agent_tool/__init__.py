"""The ``agent-tool`` condition: ARMS itself.

:mod:`conditions.agent_tool.arms` builds it from the shipped agent, with all three tools
and the shipped system prompt, in :mod:`arms_agent.prompts`.

A module dropped in here is another tool-using condition as soon as it declares a
``CONDITION``; see :mod:`conditions.registry` for what it declares.
"""
