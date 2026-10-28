"""Local corpus pairing, selection, validation and leakage boundaries."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from experiment.dataset import TEMPLATE_IRI, load_items

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_sorted_local_items_and_provenance(corpus: tuple[Path, Callable]) -> None:
    """Gold never appears in agent input; selected pairs are deterministic."""
    root, pair = corpus
    pair("z.json", gold={"answer": "GOLD_SENTINEL"})
    pair("a.json", gold={"answer": "OTHER_GOLD"})
    items = load_items(root, limit=1)
    assert len(items) == 1
    item = items[0]
    assert item["input"] == {"legacy_metadata": {"legacy": "a.json"}, "template_iri": TEMPLATE_IRI}
    assert item["expected_output"] == {"answer": "OTHER_GOLD"}
    assert item["metadata"]["record_id"] == "a"
    assert item["metadata"]["assay"] == "rnaseq"
    provenance = item["metadata"]["provenance"]
    assert provenance["input"] == "rnaseq/input/a.json"
    assert provenance["gold_sha256"] == hashlib.sha256((root / "rnaseq/gold/a.json").read_bytes()).hexdigest()
    assert len(load_items(root)) == 2


@pytest.mark.parametrize("side", ["input", "gold"])
def test_unpaired_names_even_outside_limit(corpus: tuple[Path, Callable], side: str) -> None:
    """A small selection must not hide corpus pairing errors."""
    root, pair = corpus
    pair("a.json")
    (root / f"rnaseq/{side}/z.json").write_text("{}")
    with pytest.raises(ValueError, match="Unpaired filenames"):
        load_items(root, limit=1)


@pytest.mark.parametrize("value", [[], {}, "not an object", 4])
def test_invalid_object_outside_limit(corpus: tuple[Path, Callable], value: object) -> None:
    """Validate all JSON objects, including records not selected for execution."""
    root, pair = corpus
    pair("a.json")
    pair("z.json", gold=value)
    with pytest.raises(ValueError, match="nonempty JSON object"):
        load_items(root, limit=1)


def test_missing_empty_and_invalid_limit(corpus: tuple[Path, Callable]) -> None:
    """Missing directories, empty corpora and zero limits fail before execution."""
    root, _ = corpus
    with pytest.raises(ValueError, match="no JSON pairs"):
        load_items(root)
    with pytest.raises(ValueError, match="positive"):
        load_items(root, limit=0)
    with pytest.raises(ValueError, match="Missing corpus directory"):
        load_items(root / "absent")


def test_malformed_json(corpus: tuple[Path, Callable]) -> None:
    """Do not accept unreadable JSON."""
    root, pair = corpus
    pair("a.json")
    (root / "rnaseq/input/a.json").write_text("{")
    with pytest.raises(ValueError):
        load_items(root)
