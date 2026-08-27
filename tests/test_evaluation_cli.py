"""Tests for the evaluation CLI entry point.

These run the CLI in a subprocess deliberately.  ``tests/conftest.py`` puts
``evaluation/`` on ``sys.path`` for the whole session, so an in-process import
would resolve the bare sibling imports that ``python -m evaluation`` has to
resolve on its own -- and would keep passing even with the CLI broken.  Only a
fresh interpreter started from the project root reproduces what a user runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Every condition the CLI can dispatch to, since each pulls in a different module.  The
# names are written out rather than read from the registry: a test that asked the code
# under test what to test would pass just as well with the registry returning nothing.
_CONDITIONS = [
    ["--condition", "baseline"],
    ["--condition", "arms-agent"],
]

# --output is the parent; the leaf directory is the run's name, which is the condition's
# own name unless --run-name gave it another.
_RUN_DIRECTORIES = [
    (["--condition", "baseline"], "baseline"),
    (["--condition", "arms-agent"], "arms-agent"),
    (["--condition", "arms-agent", "--run-name", "arms-agent-r2"], "arms-agent-r2"),
]


def _run_cli(tmp_path: Path, workflow_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the evaluation CLI over an empty input directory.

    An empty directory makes ``run_experiment`` return before it builds a
    workflow or reaches any API, so this exercises argument parsing and the
    whole import chain without spending anything.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "evaluation",
        "--input",
        str(input_dir),
        "--target-schema",
        "https://example.org/templates/test",
        "--output",
        str(tmp_path / "output"),
        *workflow_args,
    ]
    return subprocess.run(command, cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.parametrize("workflow_args", _CONDITIONS, ids=" ".join)
def test_condition_imports_resolve(tmp_path: Path, workflow_args: list[str]) -> None:
    """Each condition must import cleanly when the package is run as ``python -m evaluation``."""
    result = _run_cli(tmp_path, workflow_args)
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("workflow_args", _CONDITIONS, ids=" ".join)
def test_condition_reaches_the_runner(tmp_path: Path, workflow_args: list[str]) -> None:
    """Dispatch must get as far as run_experiment, not exit early for some other reason."""
    result = _run_cli(tmp_path, workflow_args)
    assert "No *.json files found" in result.stderr, result.stderr


@pytest.mark.parametrize(("workflow_args", "run_directory"), _RUN_DIRECTORIES, ids=lambda arg: str(arg))
def test_output_goes_under_the_run_name(tmp_path: Path, workflow_args: list[str], run_directory: str) -> None:
    """The run writes to <--output>/<run name>, not to --output itself."""
    result = _run_cli(tmp_path, workflow_args)
    assert str(tmp_path / "output" / run_directory) in result.stderr, result.stderr


def test_a_condition_is_required(tmp_path: Path) -> None:
    """No condition given is an error: there is no default arm to fall back on."""
    result = _run_cli(tmp_path, [])
    assert result.returncode != 0
    assert "the following arguments are required: --condition" in result.stderr, result.stderr


def test_an_unknown_condition_is_rejected(tmp_path: Path) -> None:
    """--condition takes the declared conditions and nothing else."""
    result = _run_cli(tmp_path, ["--condition", "baseline+extra"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr, result.stderr
    # Nothing ran: the runner never got as far as reading the input directory.
    assert "No *.json files found" not in result.stderr, result.stderr


def test_the_declared_conditions_are_offered(tmp_path: Path) -> None:
    """The help lists what the registry found, so a dropped-in module is reachable here."""
    result = _run_cli(tmp_path, ["--help"])
    assert result.returncode == 0, result.stderr
    assert "--condition {baseline,arms-agent}" in result.stdout.replace("\n", " ").replace("  ", " "), result.stdout


def test_a_run_name_does_not_change_the_condition(tmp_path: Path) -> None:
    """--run-name names the output; it must not be read as a condition of its own."""
    result = _run_cli(tmp_path, ["--condition", "baseline", "--run-name", "arms-agent"])
    assert result.returncode == 0, result.stderr
    assert "Running condition baseline as arms-agent" in result.stderr, result.stderr
