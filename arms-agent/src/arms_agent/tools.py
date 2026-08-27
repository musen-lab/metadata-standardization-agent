"""LangChain tool wrappers around cedar-mcp functions."""

from __future__ import annotations

import logging
import os
from typing import Any

from cedar_mcp.external_api import (
    async_get_children_from_branch,
    async_search_terms_from_branch,
    async_search_terms_from_ontology,
    get_template,
)
from cedar_mcp.processing import clean_template_response
from langchain_core.tools import tool

from arms_agent.cache import SqliteCache
from arms_agent.logging_config import log_tool_call

logger = logging.getLogger(__name__)

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


def _extract_pref_labels(response: dict[str, Any]) -> list[str]:
    """Return the ``prefLabel`` of every entry in a BioPortal ``collection``.

    Works for both the term-search and branch-children response shapes, which
    differ in their per-entry fields but both carry ``prefLabel``.  Entries
    without a usable label are skipped, and duplicates are dropped while
    preserving the order BioPortal returned.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for entry in response.get("collection", []):
        if not isinstance(entry, dict):
            continue
        label = entry.get("prefLabel")
        if not isinstance(label, str) or not label.strip():
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


@tool
@log_tool_call
async def term_search_from_branch(search_string: str, ontology_acronym: str, branch_iri: str) -> dict[str, Any]:
    """Search a branch of a controlled vocabulary and return the candidate term labels.

    Use this when a CEDAR template field has a branch-level controlled vocabulary
    constraint. Returns ``{"labels": [...], "source": "<function that produced
    them>"}``. Choose the label from ``labels`` that best matches the legacy value
    and use it verbatim as the field value.

    When the search finds no match, the whole branch is enumerated instead, so
    ``labels`` may list every permissible value rather than search hits. The
    ``source`` field says which happened. An empty ``labels`` list means the
    branch yielded nothing at all.

    Args:
        search_string: The term label or keyword to search for, taken from the
            legacy record. An empty string returns no labels.
        ontology_acronym: Ontology acronym (e.g., "CHEBI", "HRAVS").
        branch_iri: IRI of the branch to restrict the search to.
    """
    if not search_string or not search_string.strip():
        return {"labels": [], "source": "term_search_from_branch"}

    cached = _get_cache().get(
        "term_search_from_branch",
        search_string=search_string,
        ontology_acronym=ontology_acronym,
        branch_iri=branch_iri,
    )
    if cached is not None:
        return cached

    bioportal_api_key = _get_bioportal_api_key()
    response = await async_search_terms_from_branch(search_string, ontology_acronym, branch_iri, bioportal_api_key)
    source = "term_search_from_branch"
    labels = _extract_pref_labels(response)

    if not labels:
        logger.debug("Branch search for %r returned nothing; enumerating branch %s", search_string, branch_iri)
        response = await async_get_children_from_branch(
            branch_iri=branch_iri,
            ontology_acronym=ontology_acronym,
            bioportal_api_key=bioportal_api_key,
        )
        source = "get_children_from_branch"
        labels = _extract_pref_labels(response)

    result = {"labels": labels, "source": source}
    # An empty result may mean a genuinely empty branch or a transient API
    # failure; caching it would hide the difference for the whole TTL.
    if labels:
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
    """Search a whole controlled vocabulary and return the candidate term labels.

    Use this when a CEDAR template field has an ontology-level controlled vocabulary
    constraint. Returns ``{"labels": [...], "source": "term_search_from_ontology"}``.
    Choose the label from ``labels`` that best matches the legacy value and use it
    verbatim as the field value.

    Unlike a branch search there is no fallback enumeration, since a whole ontology
    is too large to list. An empty ``labels`` list means the search found nothing.

    Args:
        search_string: The term label or keyword to search for, taken from the
            legacy record. An empty string returns no labels.
        ontology_acronym: Ontology acronym (e.g., "NCIT", "CHEBI").
    """
    source = "term_search_from_ontology"
    if not search_string or not search_string.strip():
        return {"labels": [], "source": source}

    # Namespaced apart from the tool name: entries written before this tool
    # returned labels hold the raw BioPortal response, and serving one of those
    # would hand back a shape the docstring above no longer describes.
    cache_key = "term_search_from_ontology_labels"
    cached = _get_cache().get(
        cache_key,
        search_string=search_string,
        ontology_acronym=ontology_acronym,
    )
    if cached is not None:
        return cached

    bioportal_api_key = _get_bioportal_api_key()
    response = await async_search_terms_from_ontology(search_string, ontology_acronym, bioportal_api_key)
    labels = _extract_pref_labels(response)

    result = {"labels": labels, "source": source}
    # As with the branch search, an empty result may mean a genuinely empty
    # search or a transient API failure; caching it would hide the difference
    # for the whole TTL.
    if labels:
        _get_cache().set(
            cache_key,
            result,
            search_string=search_string,
            ontology_acronym=ontology_acronym,
        )
    return result


all_tools = [get_cedar_template, term_search_from_branch, term_search_from_ontology]
