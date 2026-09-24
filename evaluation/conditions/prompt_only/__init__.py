"""The condition that reaches the template through its prompt rather than tools.

:mod:`conditions.prompt_only.baseline` is handed the field names and the vocabulary
names its user prompt supplies, and pairs with the system prompt of the same name under
:mod:`conditions.prompt_only.prompts`.

It is built by the shipped :func:`arms_agent.agent.build_migration_agent`, the same
function that builds ARMS -- handed an empty tool list and its own system prompt.  With
no tools it yields a model node with no tool-calling loop, so one call answers, but the
answer is requested and validated exactly as the tool arm's is.

:mod:`conditions.prompt_only.template_spec` builds the template material that prompt is
assembled from.  It lives here rather than beside the other condition because the tool
arm has no use for it: it fetches the template through its tools instead.  It declares no
``CONDITION``, which is what tells the registry it is material rather than an arm.

A module dropped in here is another prompt-only condition as soon as it declares one; see
:mod:`conditions.registry` for what it declares.
"""
