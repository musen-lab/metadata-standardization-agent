"""Tests that the conditions differ only in information access.

Each condition keeps its own system prompt in its own module, so nothing but these tests
stops the shared parts drifting apart.  A difference in anything other than information
access would make the comparison measure that instead.
"""

from __future__ import annotations

import re

import pytest

from arms_agent.prompts import SYSTEM_PROMPT as ARMS
from conditions.agent_tool.prompts.template_tool import SYSTEM_PROMPT as TEMPLATE_TOOL
from conditions.agent_tool.prompts.term_tool import SYSTEM_PROMPT as TERM_TOOL
from conditions.prompt_only.prompts.baseline import SYSTEM_PROMPT as BASELINE

PROMPTS = {
    "baseline": BASELINE,
    "template-tool": TEMPLATE_TOOL,
    "term-tool": TERM_TOOL,
    "agent-tool": ARMS,
}
PROMPT_ONLY = ("baseline",)
# The arms that fetch the template, and the arms that search a vocabulary.  ARMS does both,
# the baseline neither, and each ablation one.
FETCHES_TEMPLATE = ("template-tool", "agent-tool")
SEARCHES = ("term-tool", "agent-tool")

# Policy that must read identically in every condition.  Abstention matters most: a
# record of nothing but nulls already scores 0.375 against a baseline of 0.559, so a
# condition that gives up more readily than another scores differently for that reason
# alone, and the comparison would measure wording rather than information access.
SHARED_POLICY = [
    "For each template field in turn, decide which legacy field or fields, if any, provide its value.",
    "Do not start from the legacy fields and map forward.",
    "Map only what the record supports.",
    "Where nothing in the record supports the field, leave it null.",
    "Work out what the field is about before you fill it.",
    "Do not trust the names:",
    "Then check the other direction once.",
    "Report whatever is still unplaced in the processing log",
    "Choose the label the legacy value denotes and output it verbatim. Judge meaning, not string similarity:",
    "- Differences of case, spacing, punctuation, or singular versus plural never rule a label out.",
    "- Leniency about form applies between the legacy value and a label, never between two labels.",
    "- An abbreviation, code, symbol, or superseded name denotes the label it stands for.",
    "Never fall back on the legacy value; outside the vocabulary it is not valid for the field.",
    "Never choose a label you cannot justify from the record.",
    "Preserve the legacy value as-is",
    "Never supply a value from any other source.",
    "- Not from your own knowledge of the domain, its products, or its conventions",
    "- Not a conventional path, code, or identifier of your own",
    "- Not a permissible value or option chosen because the field would otherwise be empty",
    "Your knowledge of the domain is for reading the record, not for filling it",
    "Never use it to supply a fact the record does not carry.",
    "Required-ness raises the effort, never the licence: look again before giving up.",
    "A required field with no corresponding value in the record is still null.",
    "Fix what fails. A non-conforming value is worse than null.",
    "Answer with one object holding two keys, `record` and `log`.",
    "`log` is an array of entries, one per decision:",
    "`key` and `value` must match `record` exactly.",
    "`resolution` is where the value came from, exactly one of:",
]


class TestSharedPolicy:
    @pytest.mark.parametrize("fragment", SHARED_POLICY)
    @pytest.mark.parametrize("condition", sorted(PROMPTS))
    def test_fragment_reads_identically_everywhere(self, condition: str, fragment: str) -> None:
        assert fragment in PROMPTS[condition], f"{condition} has drifted"

    @pytest.mark.parametrize("condition", sorted(PROMPTS))
    def test_abstention_is_stated_exactly_once(self, condition: str) -> None:
        assert PROMPTS[condition].count("Never fall back on the legacy value") == 1

    @pytest.mark.parametrize("condition", sorted(PROMPTS))
    def test_no_condition_asks_for_a_fenced_block(self, condition: str) -> None:
        """Every arm's answer is schema-validated now, so asking for fences would fight it."""
        for marker in ("```json record", "```json log"):
            assert marker not in PROMPTS[condition], f"{condition} still asks for {marker}"

    @pytest.mark.parametrize("condition", sorted(PROMPTS))
    def test_processing_log_schema_is_identical(self, condition: str) -> None:
        for key in ("key", "value", "legacy_fields", "legacy_values", "resolution", "candidates", "reasoning"):
            assert f'"{key}"' in PROMPTS[condition], f"{condition} is missing log key {key}"
        for value in ("copied", "harmonized", "derived", "defaulted", "no_value", "unmapped"):
            assert f"`{value}`" in PROMPTS[condition], f"{condition} is missing resolution {value}"


class TestInformationAccessDiffers:
    def test_no_condition_is_handed_the_permissible_values(self) -> None:
        """Both arms have to reach the values themselves -- one by search, one by recall."""
        for condition in PROMPTS:
            assert "complete list of permissible labels" not in PROMPTS[condition], condition

    def test_only_the_arms_that_fetch_the_template_have_the_specification(self) -> None:
        for condition in PROMPTS:
            lacks = "You do not have the template specification." in PROMPTS[condition]
            assert lacks is (condition not in FETCHES_TEMPLATE), condition

    def test_only_the_arms_that_fetch_the_template_are_told_to(self) -> None:
        for condition in PROMPTS:
            fetches = "Call get_cedar_template tool." in PROMPTS[condition]
            assert fetches is (condition in FETCHES_TEMPLATE), condition

    def test_only_the_arms_that_search_are_told_to(self) -> None:
        for condition in PROMPTS:
            searches = "Never skip the call, even when you believe you know the term." in PROMPTS[condition]
            assert searches is (condition in SEARCHES), condition
            batches = "## Tool Call Strategy" in PROMPTS[condition]
            assert batches is (condition in SEARCHES), f"{condition}: batching advice follows the searches"

    def test_the_arms_that_cannot_search_say_so(self) -> None:
        """Otherwise a model left without the tools may reach for a call it cannot make."""
        for condition in PROMPTS:
            if condition not in SEARCHES:
                assert "no way to query it" in PROMPTS[condition], condition
                assert "term_search_from" not in PROMPTS[condition], condition

    def test_term_tool_is_told_which_search_its_information_allows(self) -> None:
        """Its message names a vocabulary and no branch, so the branch search needs an IRI it lacks."""
        prompt = PROMPTS["term-tool"]
        assert "`term_search_from_ontology` needs only the vocabulary's acronym" in prompt
        assert "call it only when you were given that IRI, and never invent one" in prompt
        assert "get_cedar_template" not in prompt

    def test_only_arms_has_tools(self) -> None:
        assert "term_search_from_branch" in PROMPTS["agent-tool"]
        for condition in PROMPT_ONLY:
            for tool in ("term_search_from_branch", "term_search_from_ontology", "get_cedar_template"):
                assert tool not in PROMPTS[condition], f"{condition} mentions {tool}"
            assert "Tool Call Strategy" not in PROMPTS[condition]

    def test_only_the_arms_that_search_cite_a_search_in_their_provenance_clause(self) -> None:
        """An arm that cannot search resolves from what it was given, so its clause says that."""
        assert "a label the template or a search tool returned" in PROMPTS["agent-tool"]
        assert "the legacy record, or a label a search tool returned." in PROMPTS["term-tool"]
        assert "a label from the template or the vocabulary it names" in PROMPTS["template-tool"]
        for condition in PROMPT_ONLY:
            assert "a label from the given vocabulary" in PROMPTS[condition]
        for condition in PROMPTS:
            if condition not in SEARCHES:
                assert "a search tool" not in PROMPTS[condition], f"{condition} cites a search it cannot run"

    def test_only_the_arms_handed_a_candidate_list_must_weigh_all_of_it(self) -> None:
        """The searching arms are handed candidates; the others recall labels instead."""
        for condition in PROMPTS:
            weighs = "not only those resembling the legacy value" in PROMPTS[condition]
            assert weighs is (condition in SEARCHES), condition

    def test_the_baseline_never_mentions_what_it_cannot_see(self) -> None:
        for absent in ("permissible_values", "`options`", "3.2 Value-Constrained", "3.4 Datatype"):
            assert absent not in PROMPTS["baseline"], absent

    def test_a_failed_resolution_falls_to_the_default_wherever_there_is_one(self) -> None:
        """3.1 and 3.2 must not answer null where 3.5 answers with the template's default.

        Only an arm that fetches the template sees its defaults; the others have none to use.
        """
        for condition in PROMPTS:
            prompt = PROMPTS[condition]
            if condition in FETCHES_TEMPLATE:
                assert "→ the field's `default_value` if it has one, else null" in prompt, condition
                assert "if none denotes it, output null" not in prompt, f"{condition} still overrides 3.5"
            else:
                assert "`default_value`" not in prompt, f"{condition} has no defaults to fall back on"


class TestStructure:
    @pytest.mark.parametrize("condition", sorted(PROMPTS))
    def test_no_condition_cites_a_subsection_it_lacks(self, condition: str) -> None:
        prompt = PROMPTS[condition]
        headings = set(re.findall(r"^(\d+\.\d+) ", prompt, re.MULTILINE))
        cited = set(re.findall(r"\b(?:in|to|Subsections?) (\d+\.\d+)\b", prompt))
        cited |= set(re.findall(r"(\d+\.\d+) (?:and|to) \d+\.\d+", prompt))
        cited |= set(re.findall(r"\d+\.\d+ (?:and|to) (\d+\.\d+)", prompt))
        assert cited <= headings, f"{condition} cites missing: {sorted(cited - headings)}"

    @pytest.mark.parametrize("condition", sorted(PROMPTS))
    def test_sections_are_in_document_order(self, condition: str) -> None:
        prompt = PROMPTS[condition]
        markers = ["Input:", "Output:", "## Workflow", "### Step 1.", "### Step 2.", "### Step 3."]
        markers += ["3.1 ", "3.3 ", "3.5 ", "### Step 4.", "### Step 5.", "### Step 6.", "## Error Handling"]
        positions = []
        for marker in markers:
            found = re.search(rf"^{re.escape(marker)}", prompt, re.MULTILINE)
            assert found, f"{condition} is missing {marker!r}"
            positions.append(found.start())
        assert positions == sorted(positions), condition

    def test_arms_is_untouched_by_the_comparison(self) -> None:
        """ARMS is the contribution; the baseline is apparatus and must not edit it."""
        assert "### Step 1. Fetch Template" in PROMPTS["agent-tool"]
        assert "Call get_cedar_template tool." in PROMPTS["agent-tool"]
