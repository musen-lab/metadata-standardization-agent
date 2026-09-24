"""The ablations of ARMS: each holds one half of its tools.

* :mod:`conditions.ablation.template_tool` fetches the template but cannot search.
* :mod:`conditions.ablation.term_tool` searches but is told only what the baseline is.

Between them they separate what each half of ARMS's tools adds.  Their system prompts are
copies of a parent's with only the information-access sections changed, and live in
``prompts/`` here.

A module dropped in here is another ablation as soon as it declares a ``CONDITION``; see
:mod:`conditions.registry` for what it declares.
"""
