"""The ``agent-tool`` conditions.

:mod:`conditions.agent_tool.arms` is ARMS itself, built from the shipped agent with all
three tools and the shipped system prompt, in :mod:`arms_agent.prompts`.  The other two
each hold one half of its tools.

* :mod:`conditions.agent_tool.template_tool` fetches the template but cannot search.
* :mod:`conditions.agent_tool.term_tool` searches but is told only what the baseline is.

Their system prompts are copies of a parent's with only the information-access sections
changed, and live in :mod:`conditions.agent_tool.prompts`.

A module dropped in here is another tool-using condition as soon as it declares a
``CONDITION``; see :mod:`conditions.registry` for what it declares.
"""
