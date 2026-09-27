"""Tests for where a condition's predictions live: one run, or one directory per run.

A condition run once keeps its predictions in its own directory; one run several times
holds a ``run-<n>`` directory per run.  Writers state which they are writing
(:meth:`Assay.run_output_dir`); readers find out from the disk (:meth:`Assay.output_dir`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from analysis.corpus import get_assay

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def condition_dir(tmp_path: Path) -> Path:
    return tmp_path / "atacseq" / "output" / "m" / "sys"


class TestRunOutputDir:
    def test_a_single_run_is_the_condition_directory(self, tmp_path: Path, condition_dir: Path) -> None:
        assert get_assay(tmp_path, "atacseq").run_output_dir("m", "sys", None) == condition_dir

    def test_a_numbered_run_is_its_own_directory(self, tmp_path: Path, condition_dir: Path) -> None:
        assert get_assay(tmp_path, "atacseq").run_output_dir("m", "sys", 3) == condition_dir / "run-3"

    def test_it_does_not_look_at_the_disk(self, tmp_path: Path, condition_dir: Path) -> None:
        """A writer states the layout it is about to create, so run 1 of a repeat is run-1 even on an empty disk."""
        assert get_assay(tmp_path, "atacseq").run_output_dir("m", "sys", 1) == condition_dir / "run-1"


class TestOutputDir:
    def test_run_1_of_a_single_run_is_the_condition_directory(self, tmp_path: Path, condition_dir: Path) -> None:
        (condition_dir).mkdir(parents=True)
        (condition_dir / "r.json").write_text("{}")
        assert get_assay(tmp_path, "atacseq").output_dir("m", "sys") == condition_dir

    def test_run_1_of_a_repeated_condition_is_run_1(self, tmp_path: Path, condition_dir: Path) -> None:
        for run in (1, 2):
            (condition_dir / f"run-{run}").mkdir(parents=True)
        assay = get_assay(tmp_path, "atacseq")
        assert assay.output_dir("m", "sys") == condition_dir / "run-1"
        assert assay.output_dir("m", "sys", run=2) == condition_dir / "run-2"

    def test_a_run_never_made_is_a_directory_that_does_not_exist(self, tmp_path: Path, condition_dir: Path) -> None:
        condition_dir.mkdir(parents=True)
        (condition_dir / "r.json").write_text("{}")
        assert not get_assay(tmp_path, "atacseq").output_dir("m", "sys", run=2).exists()

    def test_a_condition_never_made_reads_as_a_single_run(self, tmp_path: Path, condition_dir: Path) -> None:
        assert get_assay(tmp_path, "atacseq").output_dir("m", "sys") == condition_dir

    def test_a_file_named_like_a_run_does_not_make_it_repeated(self, tmp_path: Path, condition_dir: Path) -> None:
        condition_dir.mkdir(parents=True)
        (condition_dir / "run-notes.json").write_text("{}")
        assert get_assay(tmp_path, "atacseq").output_dir("m", "sys") == condition_dir
