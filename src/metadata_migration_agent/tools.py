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
def get_cedar_template(template_id: str) -> dict[str, Any]:
    """Fetch a CEDAR template by its ID or full URL and return its cleaned structure.

    The returned structure shows the template's fields, their types, constraints,
    and any controlled vocabulary requirements. Use this to understand what fields
    the migrated metadata must contain.

    Args:
        template_id: The template ID (UUID) or full CEDAR URL
            (e.g., "https://repo.metadatacenter.org/templates/<uuid>").
    """
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

    return clean_template_response(template_data)


@tool
def term_search_from_branch(search_string: str, ontology_acronym: str, branch_iri: str) -> dict[str, Any]:
    """Search BioPortal for ontology terms within a specific branch.

    Use this when a CEDAR template field has a branch-level controlled vocabulary
    constraint and you need to find the correct standardized term IRI.

    Args:
        search_string: The term label or keyword to search for.
        ontology_acronym: Ontology acronym (e.g., "CHEBI", "HRAVS").
        branch_iri: IRI of the branch to restrict the search to.
    """
    bioportal_api_key = _get_bioportal_api_key()
    return search_terms_from_branch(search_string, ontology_acronym, branch_iri, bioportal_api_key)


@tool
def term_search_from_ontology(search_string: str, ontology_acronym: str) -> dict[str, Any]:
    """Search BioPortal for ontology terms within an entire ontology.

    Use this when a CEDAR template field has an ontology-level controlled vocabulary
    constraint and you need to find the correct standardized term IRI.

    Args:
        search_string: The term label or keyword to search for.
        ontology_acronym: Ontology acronym (e.g., "NCIT", "CHEBI").
    """
    bioportal_api_key = _get_bioportal_api_key()
    return search_terms_from_ontology(search_string, ontology_acronym, bioportal_api_key)


@tool
def get_branch_children(branch_iri: str, ontology_acronym: str) -> dict[str, Any]:
    """Get all child terms of a branch in a BioPortal ontology.

    Use this to enumerate permissible values under a branch when a template field
    requires selecting from a controlled set of terms.

    Args:
        branch_iri: IRI of the branch to get children for.
        ontology_acronym: Ontology acronym (e.g., "HRAVS").
    """
    bioportal_api_key = _get_bioportal_api_key()
    return get_children_from_branch(branch_iri, ontology_acronym, bioportal_api_key)


all_tools = [
    get_cedar_template,
    term_search_from_branch,
    term_search_from_ontology,
    get_branch_children,
]
