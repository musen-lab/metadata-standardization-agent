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

# Every workflow the CLI can dispatch to, since each pulls in a different module.  The
# workflow flag carries the run's name as its value, and both spellings -- bare, taking
# the default, and with a name -- have to reach argument parsing.
_WORKFLOWS = [
    ["--prompt-only"],
    ["--prompt-only", "baseline"],
    ["--agent-tool"],
    ["--agent-tool", "arms-agent"],
]

# --output is the parent; the leaf directory is the run's name.
_RUN_DIRECTORIES = [
    (["--prompt-only"], "baseline"),
    (["--prompt-only", "baseline"], "baseline"),
    (["--agent-tool"], "arms-agent"),
    (["--agent-tool", "other-agent"], "other-agent"),
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


@pytest.mark.parametrize("workflow_args", _WORKFLOWS, ids=" ".join)
def test_workflow_imports_resolve(tmp_path: Path, workflow_args: list[str]) -> None:
    """Each workflow must import cleanly when the package is run as ``python -m evaluation``."""
    result = _run_cli(tmp_path, workflow_args)
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("workflow_args", _WORKFLOWS, ids=" ".join)
def test_workflow_reaches_the_runner(tmp_path: Path, workflow_args: list[str]) -> None:
    """Dispatch must get as far as run_experiment, not exit early for some other reason."""
    result = _run_cli(tmp_path, workflow_args)
    assert "No *.json files found" in result.stderr, result.stderr


@pytest.mark.parametrize(("workflow_args", "run_directory"), _RUN_DIRECTORIES)
def test_output_goes_under_the_run_name(tmp_path: Path, workflow_args: list[str], run_directory: str) -> None:
    """The run writes to <--output>/<the name the workflow flag was given>, not to --output itself."""
    result = _run_cli(tmp_path, workflow_args)
    assert str(tmp_path / "output" / run_directory) in result.stderr, result.stderr


def test_a_workflow_flag_is_required(tmp_path: Path) -> None:
    """Neither flag given is an error: there is no default workflow to fall back on."""
    result = _run_cli(tmp_path, [])
    assert result.returncode != 0
    assert "one of the arguments --prompt-only --agent-tool is required" in result.stderr, result.stderr


def test_an_unknown_condition_is_rejected(tmp_path: Path) -> None:
    """--prompt-only takes a fixed set of conditions; --agent-tool takes any name."""
    result = _run_cli(tmp_path, ["--prompt-only", "baseline+extra"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr, result.stderr


@pytest.mark.parametrize(
    "workflow_args",
    [
        ["--prompt-only", "--agent-tool"],
        ["--prompt-only", "baseline", "--agent-tool", "arms-agent"],
        ["--agent-tool", "arms-agent", "--prompt-only", "baseline"],
    ],
    ids=" ".join,
)
def test_both_workflow_flags_is_refused(tmp_path: Path, workflow_args: list[str]) -> None:
    """Both flags given must say so and stop, rather than pick one and run under it.

    The two would name the run differently, so there is no safe way to proceed.
    """
    result = _run_cli(tmp_path, workflow_args)
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr, result.stderr
    # Nothing ran: the runner never got as far as reading the input directory.
    assert "No *.json files found" not in result.stderr, result.stderr
