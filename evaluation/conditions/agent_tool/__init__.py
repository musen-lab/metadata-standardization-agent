"""The ``agent-tool`` condition: ARMS itself.

:mod:`conditions.agent_tool.arms` builds it from the shipped agent, which fetches the
template and looks up terms through tools instead of being handed them in the prompt.
Its system prompt is not kept here: it is the shipped one, in
:mod:`arms_agent.prompts`.
"""
