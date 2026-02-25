"""Tests for the metadata migration workflow."""

from metadata_migration_agent.state import AgentState
from metadata_migration_agent.tools import all_tools


def test_state_has_messages():
    """Verify AgentState supports a messages list."""
    state = AgentState(messages=[], cedar_template_iri="https://example.org/template/1", metadata={})
    assert state["messages"] == []
    assert state["cedar_template_iri"] == "https://example.org/template/1"
    assert state["metadata"] == {}


def test_tools_registered():
    """Verify all expected tools are present."""
    tool_names = {t.name for t in all_tools}
    assert "get_cedar_template" in tool_names
    assert "term_search_from_branch" in tool_names
    assert "term_search_from_ontology" in tool_names
    assert "get_branch_children" in tool_names
