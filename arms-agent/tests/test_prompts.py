"""Tests for the system prompt's structure and versioning."""

from __future__ import annotations

import re

from arms_agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT

_HEADING_RE = re.compile(r"^(\d+\.\d+) ", re.MULTILINE)
_REFERENCE_RE = re.compile(r"\b(?:in|to) (\d+\.\d+)\b")


def test_prompt_version_is_set() -> None:
    assert isinstance(PROMPT_VERSION, str)
    assert re.fullmatch(r"\d+\.\d+", PROMPT_VERSION), PROMPT_VERSION


def test_every_subsection_cross_reference_resolves() -> None:
    """A renumbering must not leave a reference pointing at a section that is gone."""
    headings = set(_HEADING_RE.findall(SYSTEM_PROMPT))
    referenced = set(_REFERENCE_RE.findall(SYSTEM_PROMPT))
    assert headings, "expected numbered subsections in the prompt"
    assert referenced <= headings, f"dangling reference(s): {sorted(referenced - headings)}"


def test_step_three_subsections_are_contiguous() -> None:
    numbers = sorted(int(h.split(".")[1]) for h in _HEADING_RE.findall(SYSTEM_PROMPT))
    assert numbers == list(range(1, len(numbers) + 1)), numbers


def test_no_flag_vocabulary_remains() -> None:
    """The processing log replaced the flag scheme; stale flag names confuse it."""
    for flag in ("NO_ONTOLOGY_MATCH", "AMBIGUOUS_TERM", "AMBIGUOUS_MAPPING", "INFERRED"):
        assert flag not in SYSTEM_PROMPT, flag


def test_both_output_keys_are_specified() -> None:
    """The answer is one schema-enforced object, so the prompt names its two keys."""
    assert "`record`" in SYSTEM_PROMPT
    assert "`log`" in SYSTEM_PROMPT


def test_no_fenced_output_blocks_remain() -> None:
    """The fenced-block scheme is gone; asking for it again would fight the schema."""
    assert "```json record" not in SYSTEM_PROMPT
    assert "```json log" not in SYSTEM_PROMPT
