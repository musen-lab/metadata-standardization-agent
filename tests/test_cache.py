"""Tests for the SqliteCache class."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from metadata_standardization_agent.cache import SqliteCache


@pytest.fixture
def cache(tmp_path: Path) -> SqliteCache:
    """Create a SqliteCache with a temp DB and short TTL for testing."""
    return SqliteCache(db_path=tmp_path / "test_cache.db", ttl_seconds=10)


class TestGetSetRoundtrip:
    """Test basic cache get/set operations."""

    def test_miss_returns_none(self, cache: SqliteCache) -> None:
        assert cache.get("my_func", key="value") is None

    def test_set_then_get_returns_result(self, cache: SqliteCache) -> None:
        cache.set("my_func", {"data": "hello"}, key="value")
        result = cache.get("my_func", key="value")
        assert result is not None
        assert result["data"] == "hello"
        assert result["_cached"] is True
        assert "_cache_age_seconds" in result

    def test_different_params_are_separate_entries(self, cache: SqliteCache) -> None:
        cache.set("my_func", {"data": "a"}, key="1")
        cache.set("my_func", {"data": "b"}, key="2")
        assert cache.get("my_func", key="1")["data"] == "a"
        assert cache.get("my_func", key="2")["data"] == "b"

    def test_different_funcs_are_separate_entries(self, cache: SqliteCache) -> None:
        cache.set("func_a", {"data": "a"}, key="same")
        cache.set("func_b", {"data": "b"}, key="same")
        assert cache.get("func_a", key="same")["data"] == "a"
        assert cache.get("func_b", key="same")["data"] == "b"

    def test_param_order_does_not_matter(self, cache: SqliteCache) -> None:
        cache.set("my_func", {"data": "hello"}, x="1", y="2")
        result = cache.get("my_func", y="2", x="1")
        assert result is not None
        assert result["data"] == "hello"


class TestTTLExpiry:
    """Test that entries expire after TTL."""

    def test_expired_entry_returns_none(self, tmp_path: Path) -> None:
        cache = SqliteCache(db_path=tmp_path / "ttl_cache.db", ttl_seconds=1)
        cache.set("my_func", {"data": "hello"}, key="value")

        # Entry should be available immediately
        assert cache.get("my_func", key="value") is not None

        # Wait for expiry
        time.sleep(1.1)
        assert cache.get("my_func", key="value") is None


class TestErrorExclusion:
    """Test that error responses are not cached."""

    def test_error_dict_is_not_cached(self, cache: SqliteCache) -> None:
        cache.set("my_func", {"error": "something went wrong"}, key="value")
        assert cache.get("my_func", key="value") is None

    def test_error_key_with_other_data_is_not_cached(self, cache: SqliteCache) -> None:
        cache.set("my_func", {"error": "bad", "details": "more info"}, key="value")
        assert cache.get("my_func", key="value") is None

    def test_non_error_dict_is_cached(self, cache: SqliteCache) -> None:
        cache.set("my_func", {"result": "ok"}, key="value")
        assert cache.get("my_func", key="value") is not None


class TestRemoveStale:
    """Test stale entry cleanup."""

    def test_remove_stale_deletes_expired_entries(self, tmp_path: Path) -> None:
        cache = SqliteCache(db_path=tmp_path / "stale_cache.db", ttl_seconds=1)
        cache.set("old_func", {"data": "old"}, key="old")

        time.sleep(1.1)

        # Add a fresh entry
        cache.set("new_func", {"data": "new"}, key="new")

        result = cache.remove_stale()
        assert result["removed_count"] == 1
        assert result["remaining_count"] == 1

        # Old entry gone, new entry still there
        assert cache.get("old_func", key="old") is None
        assert cache.get("new_func", key="new") is not None


class TestClearAll:
    """Test clearing all entries."""

    def test_clear_all_removes_everything(self, cache: SqliteCache) -> None:
        cache.set("func_a", {"data": "a"}, key="1")
        cache.set("func_b", {"data": "b"}, key="2")

        result = cache.clear_all()
        assert result["cleared_count"] == 2
        assert cache.get("func_a", key="1") is None
        assert cache.get("func_b", key="2") is None

    def test_clear_all_on_empty_cache(self, cache: SqliteCache) -> None:
        result = cache.clear_all()
        assert result["cleared_count"] == 0
