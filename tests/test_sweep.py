"""Tests for the notebook's two calls: :func:`sweep.plan_sweep` and :func:`sweep.run_sweep`.

Nothing here reaches an API.  ``run_experiment`` is stubbed, which is the seam where a
job would start spending, so what these check is everything before that: which jobs, in
which order, reading and writing where, and every reason a sweep is stopped before it
starts.

``load_dotenv`` is stubbed for the same reason ``conftest`` clears the environment -- a
developer's ``.env`` would otherwise put real keys into a test about missing ones.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

import sweep
from conditions import CONDITIONS, build_condition
from sweep import SweepPlan, plan_sweep, run_sweep

if TYPE_CHECKING:
    from pathlib import Path

_KEYS = ("OPENAI_API_KEY", "CEDAR_API_KEY", "BIOPORTAL_API_KEY")


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's ``.env`` out of tests about what the environment holds."""
    monkeypatch.setattr(sweep, "load_dotenv", lambda *args, **kwargs: False)


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every API key a condition can ask for."""
    for name in _KEYS:
        monkeypatch.setenv(name, "test-key")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A data root holding two input records for each of two assays."""
    for assay in ("atacseq", "lcms"):
        input_dir = tmp_path / assay / "input"
        input_dir.mkdir(parents=True)
        for index in range(2):
            (input_dir / f"{assay}-{index}.json").write_text(json.dumps({"id": index}))
    return tmp_path


def test_plan_is_assay_major(data_root: Path, keys: None) -> None:
    """Every condition of one assay runs before the next assay starts."""
    plan = plan_sweep(data_root, "test-model", assays=["atacseq", "lcms"], conditions=["baseline", "arms-agent"])

    assert plan.jobs == (
        ("atacseq", "baseline"),
        ("atacseq", "arms-agent"),
        ("lcms", "baseline"),
        ("lcms", "arms-agent"),
    )
    assert plan.assays == ["atacseq", "lcms"]
    assert plan.conditions == ["baseline", "arms-agent"]
    assert plan.migrations == 8  # 2 records x 2 assays x 2 conditions


def test_plan_reads_and_writes_where_the_cli_does(data_root: Path, keys: None) -> None:
    """The directories the analysis section reads back."""
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["arms-agent"])

    condition_dir = data_root / "atacseq" / "output" / "test-model" / "arms-agent"
    assert plan.output_dir("atacseq", "arms-agent") == condition_dir / "run-1"
    assert plan.output_dir("atacseq", "arms-agent", 3) == condition_dir / "run-3"
    assert [path.name for path in plan.input_records("atacseq")] == ["atacseq-0.json", "atacseq-1.json"]


def test_plan_defaults_to_every_condition(data_root: Path, keys: None) -> None:
    """Naming no conditions runs both."""
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"])

    assert plan.conditions == list(CONDITIONS)


@pytest.mark.parametrize(
    ("assays", "conditions", "expected"),
    [
        (["atacseq", "not-an-assay"], ["baseline"], "not-an-assay"),
        (["atacseq"], ["baseline-typo"], "baseline-typo"),
        ([], ["baseline"], "Nothing to run"),
        (["atacseq"], [], "Nothing to run"),
    ],
)
def test_plan_refuses_a_bad_name(
    data_root: Path, keys: None, assays: list[str], conditions: list[str], expected: str
) -> None:
    """A typo costs nothing rather than failing partway through a sweep."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        plan_sweep(data_root, "test-model", assays=assays, conditions=conditions)


def test_plan_refuses_an_assay_with_no_records(data_root: Path, keys: None) -> None:
    """An assay whose input directory is empty is named before anything runs."""
    for record in (data_root / "lcms" / "input").glob("*.json"):
        record.unlink()

    with pytest.raises(FileNotFoundError, match="lcms"):
        plan_sweep(data_root, "test-model", assays=["atacseq", "lcms"], conditions=["baseline"])


def test_plan_refuses_a_missing_key(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The keys every condition needs are checked before the first call is made."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(OSError, match="CEDAR_API_KEY"):
        plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])


def test_only_some_conditions_need_bioportal(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``baseline`` asks BioPortal nothing; ARMS looks terms up there."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CEDAR_API_KEY", "test-key")

    plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(OSError, match="BIOPORTAL_API_KEY"):
        plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["arms-agent"])


def _record_jobs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub out the two things a run does, and collect what each one was asked for."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(sweep, "build_condition", lambda condition: (lambda **kwargs: condition, lambda *args: ""))
    monkeypatch.setattr(sweep, "run_experiment", lambda **kwargs: calls.append(kwargs))
    return calls


def test_dry_run_spends_nothing(data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default lists the jobs it would run and stops."""
    calls = _record_jobs(monkeypatch)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline", "arms-agent"])

    run_sweep(plan)

    assert calls == []


def test_run_sweep_runs_every_job_in_order(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One call to ``run_experiment`` per run, in the plan's order, pointed at its own directories."""
    calls = _record_jobs(monkeypatch)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq", "lcms"], conditions=["baseline"])
    capsys.readouterr()

    run_sweep(plan, dry_run=False, max_concurrency=3)

    assert [call["input_dir"].parent.name for call in calls] == ["atacseq", "lcms"]
    assert [call["output_dir"] for call in calls] == [
        plan.output_dir("atacseq", "baseline"),
        plan.output_dir("lcms", "baseline"),
    ]
    assert [call["max_concurrency"] for call in calls] == [3, 3]
    assert [call["config"]["metadata"]["condition"] for call in calls] == ["baseline", "baseline"]
    assert "[1/2] atacseq | baseline | run 1 | 2 record(s)" in capsys.readouterr().out


def test_each_run_repeats_the_whole_plan_into_its_own_directory(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run is the outer loop: every job finishes once before any job runs twice."""
    calls = _record_jobs(monkeypatch)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq", "lcms"], conditions=["baseline"])
    capsys.readouterr()

    run_sweep(plan, n_repeat=2, dry_run=False)

    assert [call["output_dir"] for call in calls] == [
        plan.output_dir("atacseq", "baseline", 1),
        plan.output_dir("lcms", "baseline", 1),
        plan.output_dir("atacseq", "baseline", 2),
        plan.output_dir("lcms", "baseline", 2),
    ]
    assert [call["config"]["metadata"]["run"] for call in calls] == [1, 1, 2, 2]
    assert "[4/4] lcms | baseline | run 2 | 2 record(s)" in capsys.readouterr().out


def test_a_dry_run_lists_every_run(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry run shows what every run would cost, and still spends nothing."""
    calls = _record_jobs(monkeypatch)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])
    capsys.readouterr()

    run_sweep(plan, n_repeat=3)

    out = capsys.readouterr().out
    assert calls == []
    assert out.count("would run") == 3
    assert "run-3" in out


@pytest.mark.parametrize("n_repeat", [0, -1])
def test_fewer_than_one_run_is_refused(data_root: Path, keys: None, n_repeat: int) -> None:
    """A sweep that would run nothing says so rather than quietly doing nothing."""
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="n_repeat"):
        run_sweep(plan, n_repeat=n_repeat)


def test_each_assay_is_traced_under_its_own_environment(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One Langfuse environment per assay, so a sweep can be read one assay at a time."""
    environments: list[str | None] = []
    monkeypatch.setattr(sweep, "build_condition", lambda condition: (lambda **kwargs: condition, lambda *args: ""))
    monkeypatch.setattr(
        sweep,
        "run_experiment",
        lambda **kwargs: environments.append(sweep.os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")),
    )
    plan = plan_sweep(data_root, "test-model", assays=["atacseq", "lcms"], conditions=["baseline"])

    run_sweep(plan, dry_run=False)

    assert environments == ["experiment-atacseq", "experiment-lcms"]


def test_plan_is_a_value(data_root: Path, keys: None) -> None:
    """The plan can be trimmed and re-read without going back through the checks."""
    plan = plan_sweep(data_root, "test-model", assays=["atacseq", "lcms"], conditions=["baseline"])

    trimmed = SweepPlan(plan.data_root, plan.model, plan.jobs[:1])

    assert trimmed.assays == ["atacseq"]
    assert trimmed.migrations == 2


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_builds(condition: str) -> None:
    """Each name reaches a workflow builder and a prompt builder of its own."""
    build_workflow, build_user_prompt = build_condition(condition)

    assert callable(build_workflow)
    assert callable(build_user_prompt)


def test_an_unknown_condition_raises() -> None:
    """The dispatch the CLI and the sweep share names what it expected."""
    with pytest.raises(ValueError, match="armsagent"):
        build_condition("armsagent")
