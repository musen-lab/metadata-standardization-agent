"""Tests for the LangChain tool wrappers, chiefly the two term searches."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from arms_agent import tools
from arms_agent.cache import SqliteCache

if TYPE_CHECKING:
    from pathlib import Path

SEARCH_RESPONSE: dict[str, Any] = {
    "page": 1,
    "pageCount": 1,
    "totalCount": 2,
    "collection": [
        {"prefLabel": "Axio Scan.Z1", "@id": "https://identifiers.org/RRID:SCR_020927"},
        {"prefLabel": "Axio Zoom.V16", "@id": "https://identifiers.org/RRID:SCR_027090"},
    ],
}

EMPTY_RESPONSE: dict[str, Any] = {"page": 1, "pageCount": 0, "totalCount": 0, "collection": []}

CHILDREN_RESPONSE: dict[str, Any] = {
    "page": 1,
    "pageCount": 1,
    "totalCount": 3,
    "collection": [
        {"prefLabel": "Lipid", "inScheme": ["x"], "@id": "obo/C616"},
        {"prefLabel": "Metabolite", "inScheme": ["x"], "@id": "obo/C61154"},
        {"prefLabel": "Polysaccharide", "inScheme": ["x"], "@id": "obo/CHEBI_18154"},
    ],
}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module-level cache at a temporary database and set a fake API key."""
    monkeypatch.setattr(tools, "_cache", SqliteCache(db_path=tmp_path / "cache.db"))
    monkeypatch.setenv("BIOPORTAL_API_KEY", "test-key")


class TestExtractPrefLabels:
    """Tests for the response-cleaning helper."""

    def test_extracts_search_shape(self) -> None:
        assert tools._extract_pref_labels(SEARCH_RESPONSE) == ["Axio Scan.Z1", "Axio Zoom.V16"]

    def test_extracts_children_shape(self) -> None:
        assert tools._extract_pref_labels(CHILDREN_RESPONSE) == ["Lipid", "Metabolite", "Polysaccharide"]

    def test_empty_collection(self) -> None:
        assert tools._extract_pref_labels(EMPTY_RESPONSE) == []

    def test_missing_collection_key(self) -> None:
        assert tools._extract_pref_labels({"error": "boom"}) == []

    def test_skips_unusable_entries_and_dedupes(self) -> None:
        response = {
            "collection": [
                {"prefLabel": "Lipid"},
                {"prefLabel": ""},
                {"prefLabel": None},
                {"@id": "no-label"},
                "not-a-dict",
                {"prefLabel": "Lipid"},
            ]
        }
        assert tools._extract_pref_labels(response) == ["Lipid"]


class TestTermSearchFromBranch:
    """Tests for the branch search tool and its branch-children fallback."""

    def _invoke(self, search_string: str = "AxioScan.Z1") -> dict[str, Any]:
        return asyncio.run(
            tools.term_search_from_branch.ainvoke(
                {
                    "search_string": search_string,
                    "ontology_acronym": "HRAVS",
                    "branch_iri": "https://example.org/branch",
                }
            )
        )

    def test_returns_labels_from_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return SEARCH_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_branch", fake_search)
        result = self._invoke()
        assert result == {"labels": ["Axio Scan.Z1", "Axio Zoom.V16"], "source": "term_search_from_branch"}

    def test_falls_back_to_branch_children_when_search_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return EMPTY_RESPONSE

        async def fake_children(**kwargs: Any) -> dict[str, Any]:
            assert kwargs["branch_iri"] == "https://example.org/branch"
            assert kwargs["ontology_acronym"] == "HRAVS"
            return CHILDREN_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_branch", fake_search)
        monkeypatch.setattr(tools, "async_get_children_from_branch", fake_children)
        result = self._invoke()
        assert result["source"] == "get_children_from_branch"
        assert result["labels"] == ["Lipid", "Metabolite", "Polysaccharide"]

    def test_empty_search_string_returns_no_labels_without_calling_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("no API call expected for an empty search string")

        monkeypatch.setattr(tools, "async_search_terms_from_branch", boom)
        monkeypatch.setattr(tools, "async_get_children_from_branch", boom)
        for value in ("", "   "):
            assert self._invoke(value) == {"labels": [], "source": "term_search_from_branch"}

    def test_both_sources_empty_returns_empty_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return EMPTY_RESPONSE

        async def fake_children(**_kwargs: Any) -> dict[str, Any]:
            return {"error": "branch not found"}

        monkeypatch.setattr(tools, "async_search_terms_from_branch", fake_search)
        monkeypatch.setattr(tools, "async_get_children_from_branch", fake_children)
        result = self._invoke()
        assert result == {"labels": [], "source": "get_children_from_branch"}

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return SEARCH_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_branch", fake_search)
        first = self._invoke()
        second = self._invoke()
        assert calls == 1
        assert first["labels"] == second["labels"]
        assert second["_cached"] is True

    def test_empty_result_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return EMPTY_RESPONSE

        async def fake_children(**_kwargs: Any) -> dict[str, Any]:
            return EMPTY_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_branch", fake_search)
        monkeypatch.setattr(tools, "async_get_children_from_branch", fake_children)
        self._invoke()
        self._invoke()
        assert calls == 2, "a transient empty result must not be cached"


class TestTermSearchFromOntology:
    """Tests for the ontology search tool, which must present the branch tool's shape."""

    def _invoke(self, search_string: str = "AxioScan.Z1") -> dict[str, Any]:
        return asyncio.run(
            tools.term_search_from_ontology.ainvoke({"search_string": search_string, "ontology_acronym": "NCIT"})
        )

    def test_returns_labels_not_the_raw_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return SEARCH_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", fake_search)
        result = self._invoke()
        assert result == {"labels": ["Axio Scan.Z1", "Axio Zoom.V16"], "source": "term_search_from_ontology"}

    def test_empty_search_string_returns_no_labels_without_calling_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("no API call expected for an empty search string")

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", boom)
        for value in ("", "   "):
            assert self._invoke(value) == {"labels": [], "source": "term_search_from_ontology"}

    def test_no_match_returns_empty_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return EMPTY_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", fake_search)
        assert self._invoke() == {"labels": [], "source": "term_search_from_ontology"}

    def test_error_response_returns_empty_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"error": "ontology not found"}

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", fake_search)
        assert self._invoke() == {"labels": [], "source": "term_search_from_ontology"}

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return SEARCH_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", fake_search)
        first = self._invoke()
        second = self._invoke()
        assert calls == 1
        assert first["labels"] == second["labels"]
        assert second["_cached"] is True

    def test_empty_result_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return EMPTY_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", fake_search)
        self._invoke()
        self._invoke()
        assert calls == 2, "a transient empty result must not be cached"

    def test_a_stale_raw_response_is_not_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Entries written before this tool returned labels must never be handed back."""
        tools._get_cache().set(
            "term_search_from_ontology",
            SEARCH_RESPONSE,
            search_string="AxioScan.Z1",
            ontology_acronym="NCIT",
        )

        async def fake_search(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return SEARCH_RESPONSE

        monkeypatch.setattr(tools, "async_search_terms_from_ontology", fake_search)
        assert self._invoke() == {"labels": ["Axio Scan.Z1", "Axio Zoom.V16"], "source": "term_search_from_ontology"}
