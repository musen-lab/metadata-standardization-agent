"""LangChain tool wrappers around cedar-mcp functions."""

from __future__ import annotations

import os
from typing import Any

from cedar_mcp.external_api import (
    async_search_terms_from_branch,
    async_search_terms_from_ontology,
    get_template,
)
from cedar_mcp.processing import clean_template_response
from langchain_core.tools import tool

from metadata_migration_agent.cache import SqliteCache
from metadata_migration_agent.logging_config import log_tool_call

_cache: SqliteCache | None = None


def _get_cache() -> SqliteCache:
    """Return the module-level cache instance, creating it on first use."""
    global _cache  # noqa: PLW0603
    if _cache is None:
        _cache = SqliteCache()
    return _cache


def _get_cedar_api_key() -> str:
    """Return the CEDAR API key from the environment."""
    key = os.environ.get("CEDAR_API_KEY", "")
    if not key:
        raise ValueError("CEDAR_API_KEY environment variable is not set")
    return key


def _get_bioportal_api_key() -> str:
    """Return the BioPortal API key from the environment."""
    key = os.environ.get("BIOPORTAL_API_KEY", "")
    if not key:
        raise ValueError("BIOPORTAL_API_KEY environment variable is not set")
    return key


@tool
@log_tool_call
def get_cedar_template(template_id: str) -> dict[str, Any]:
    """Fetch a CEDAR template by its ID or full URL and return its cleaned structure.

    The returned structure shows the template's fields, their types, constraints,
    and any controlled vocabulary requirements. Use this to understand what fields
    the migrated metadata must contain.

    Args:
        template_id: The template ID (UUID) or full CEDAR URL
            (e.g., "https://repo.metadatacenter.org/templates/<uuid>").
    """
    cached = _get_cache().get("get_cedar_template", template_id=template_id)
    if cached is not None:
        return cached

    cedar_api_key = _get_cedar_api_key()

    template_data = get_template(template_id, cedar_api_key)
    if "error" in template_data:
        return template_data

    result = clean_template_response(template_data)
    _get_cache().set("get_cedar_template", result, template_id=template_id)
    return result


@tool
@log_tool_call
async def term_search_from_branch(search_string: str, ontology_acronym: str, branch_iri: str) -> dict[str, Any]:
    """Search BioPortal for ontology terms within a specific branch.

    Use this when a CEDAR template field has a branch-level controlled vocabulary
    constraint and you need to find the correct standardized term IRI.

    Args:
        search_string: The term label or keyword to search for.
        ontology_acronym: Ontology acronym (e.g., "CHEBI", "HRAVS").
        branch_iri: IRI of the branch to restrict the search to.
    """
    cached = _get_cache().get(
        "term_search_from_branch",
        search_string=search_string,
        ontology_acronym=ontology_acronym,
        branch_iri=branch_iri,
    )
    if cached is not None:
        return cached

    bioportal_api_key = _get_bioportal_api_key()
    result = await async_search_terms_from_branch(search_string, ontology_acronym, branch_iri, bioportal_api_key)
    _get_cache().set(
        "term_search_from_branch",
        result,
        search_string=search_string,
        ontology_acronym=ontology_acronym,
        branch_iri=branch_iri,
    )
    return result


@tool
@log_tool_call
async def term_search_from_ontology(search_string: str, ontology_acronym: str) -> dict[str, Any]:
    """Search BioPortal for ontology terms within an entire ontology.

    Use this when a CEDAR template field has an ontology-level controlled vocabulary
    constraint and you need to find the correct standardized term IRI.

    Args:
        search_string: The term label or keyword to search for.
        ontology_acronym: Ontology acronym (e.g., "NCIT", "CHEBI").
    """
    cached = _get_cache().get(
        "term_search_from_ontology",
        search_string=search_string,
        ontology_acronym=ontology_acronym,
    )
    if cached is not None:
        return cached

    bioportal_api_key = _get_bioportal_api_key()
    result = await async_search_terms_from_ontology(search_string, ontology_acronym, bioportal_api_key)
    _get_cache().set(
        "term_search_from_ontology",
        result,
        search_string=search_string,
        ontology_acronym=ontology_acronym,
    )
    return result


all_tools = [get_cedar_template, term_search_from_branch, term_search_from_ontology]
