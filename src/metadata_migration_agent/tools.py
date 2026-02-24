"""LangChain tool wrappers around cedar-mcp functions."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import requests
from cedar_mcp.external_api import (
    get_children_from_branch,
    search_terms_from_branch,
    search_terms_from_ontology,
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


def _cedar_http_error(exc: requests.exceptions.HTTPError, template_id: str) -> dict[str, str]:
    """Return a user-friendly error dict for a failed CEDAR API request."""
    status_code = exc.response.status_code if exc.response is not None else None
    if status_code == 401:
        return {
            "error": (
                "Authentication failed (HTTP 401). Your CEDAR_API_KEY is invalid or expired. "
                "Please generate a new API key from your CEDAR profile at https://cedar.metadatacenter.org."
            )
        }
    if status_code == 403:
        return {
            "error": (
                f"Access denied (HTTP 403). Your CEDAR account does not have permission to read template "
                f"'{template_id}'. Ask the template owner to share it with you, or verify you can view it "
                f"in the CEDAR Workbench."
            )
        }
    if status_code == 404:
        return {
            "error": (
                f"Template not found (HTTP 404). No template exists at '{template_id}'. "
                f"Please check that the template ID or URL is correct."
            )
        }
    return {"error": f"Failed to fetch CEDAR template (HTTP {status_code}): {exc}"}


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

    encoded_template_id = quote(template_id, safe="")
    base_url = f"https://resource.metadatacenter.org/templates/{encoded_template_id}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"apiKey {cedar_api_key}",
    }

    try:
        response = requests.get(base_url, headers=headers, timeout=30)
        response.raise_for_status()
        template_data = response.json()
    except requests.exceptions.HTTPError as e:
        return _cedar_http_error(e, template_id)
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch CEDAR template: {e}"}

    result = clean_template_response(template_data)
    _get_cache().set("get_cedar_template", result, template_id=template_id)
    return result


@tool
@log_tool_call
def term_search_from_branch(search_string: str, ontology_acronym: str, branch_iri: str) -> dict[str, Any]:
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
    result = search_terms_from_branch(search_string, ontology_acronym, branch_iri, bioportal_api_key)
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
def term_search_from_ontology(search_string: str, ontology_acronym: str) -> dict[str, Any]:
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
    result = search_terms_from_ontology(search_string, ontology_acronym, bioportal_api_key)
    _get_cache().set(
        "term_search_from_ontology",
        result,
        search_string=search_string,
        ontology_acronym=ontology_acronym,
    )
    return result


@tool
@log_tool_call
def get_branch_children(branch_iri: str, ontology_acronym: str) -> dict[str, Any]:
    """Get all child terms of a branch in a BioPortal ontology.

    Use this to enumerate permissible values under a branch when a template field
    requires selecting from a controlled set of terms.

    Args:
        branch_iri: IRI of the branch to get children for.
        ontology_acronym: Ontology acronym (e.g., "HRAVS").
    """
    cached = _get_cache().get(
        "get_branch_children",
        branch_iri=branch_iri,
        ontology_acronym=ontology_acronym,
    )
    if cached is not None:
        return cached

    bioportal_api_key = _get_bioportal_api_key()
    result = get_children_from_branch(branch_iri, ontology_acronym, bioportal_api_key)
    _get_cache().set(
        "get_branch_children",
        result,
        branch_iri=branch_iri,
        ontology_acronym=ontology_acronym,
    )
    return result


all_tools = [
    get_cedar_template,
    term_search_from_branch,
    term_search_from_ontology,
    get_branch_children,
]
