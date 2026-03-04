"""Tests for the metadata standardization agent."""

from metadata_standardization_agent.state import AgentState
from metadata_standardization_agent.tools import all_tools


def test_state_has_messages():
    """Verify AgentState supports a messages list."""
    state = AgentState(messages=[])
    assert state["messages"] == []


def test_tools_registered():
    """Verify all expected tools are present."""
    tool_names = {t.name for t in all_tools}
    assert "get_cedar_template" in tool_names
    assert "term_search_from_branch" in tool_names
    assert "term_search_from_ontology" in tool_names
    assert len(tool_names) == 3
