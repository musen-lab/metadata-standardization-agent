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
    assert plan.output_dir("atacseq", "arms-agent") == condition_dir  # a single run, as the CLI writes it
    assert plan.output_dir("atacseq", "arms-agent", 3) == condition_dir / "run-3"  # run 3 of a repeated sweep
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
    assert [call["config"]["metadata"]["run"] for call in calls] == [1, 1]
    out = capsys.readouterr().out
    assert "[1/2] atacseq | baseline | 2 record(s)" in out
    assert "run-" not in out, "a single run writes to the condition's own directory"


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


def test_a_repeated_sweep_writes_numbered_runs_even_for_run_1(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_jobs(monkeypatch)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    run_sweep(plan, n_repeat=2, dry_run=False)

    condition_dir = data_root / "atacseq" / "output" / "test-model" / "baseline"
    assert [call["output_dir"] for call in calls] == [condition_dir / "run-1", condition_dir / "run-2"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_one_run_is_refused_where_numbered_runs_already_are(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """Mixing the layouts would leave a reader unable to tell which run 1 is -- refused before anything runs."""
    calls = _record_jobs(monkeypatch)
    (data_root / "atacseq" / "output" / "test-model" / "baseline" / "run-1").mkdir(parents=True)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="a single run and numbered runs in one condition folder"):
        run_sweep(plan, dry_run=dry_run)
    assert calls == []


@pytest.mark.parametrize("dry_run", [True, False])
def test_repeats_are_refused_where_a_single_run_already_is(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    calls = _record_jobs(monkeypatch)
    condition_dir = data_root / "atacseq" / "output" / "test-model" / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "atacseq-0.json").write_text("{}")
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="a single run and numbered runs in one condition folder"):
        run_sweep(plan, n_repeat=3, dry_run=dry_run)
    assert calls == []


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_single_run_is_not_overwritten_unless_asked(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    calls = _record_jobs(monkeypatch)
    condition_dir = data_root / "atacseq" / "output" / "test-model" / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "atacseq-0.json").write_text("{}")
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="already hold predictions") as refusal:
        run_sweep(plan, dry_run=dry_run)
    assert str(condition_dir) in str(refusal.value)
    assert "first_run" not in str(refusal.value), "there are no numbered runs to follow"
    assert calls == []


def test_overwrite_replaces_a_single_run(data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_jobs(monkeypatch)
    condition_dir = data_root / "atacseq" / "output" / "test-model" / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "atacseq-0.json").write_text("{}")
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    run_sweep(plan, overwrite=True, dry_run=False)

    assert [call["output_dir"] for call in calls] == [condition_dir]
    assert [call["overwrite"] for call in calls] == [True], "run_experiment is told, so its own check agrees"


def test_overwrite_never_allows_mixing_the_layouts(data_root: Path, keys: None) -> None:
    condition_dir = data_root / "atacseq" / "output" / "test-model" / "baseline"
    condition_dir.mkdir(parents=True)
    (condition_dir / "atacseq-0.json").write_text("{}")
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="a single run and numbered runs in one condition folder"):
        run_sweep(plan, n_repeat=2, overwrite=True)


def _existing_run(data_root: Path, condition: str, run: int) -> Path:
    """A run-<run> directory of atacseq/<condition> already holding a prediction."""
    run_dir = data_root / "atacseq" / "output" / "test-model" / condition / f"run-{run}"
    run_dir.mkdir(parents=True)
    (run_dir / "atacseq-0.json").write_text("{}")
    return run_dir


def test_new_runs_can_follow_the_runs_already_on_disk(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The case this exists for: run-1 is there, and two more runs become run-2 and run-3."""
    calls = _record_jobs(monkeypatch)
    run_1 = _existing_run(data_root, "baseline", 1)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])
    capsys.readouterr()

    run_sweep(plan, n_repeat=2, first_run=2, dry_run=False)

    assert [call["output_dir"] for call in calls] == [run_1.parent / "run-2", run_1.parent / "run-3"]
    assert [call["config"]["metadata"]["run"] for call in calls] == [2, 3]
    assert "[1/2] atacseq | baseline | run 2 |" in capsys.readouterr().out
    assert (run_1 / "atacseq-0.json").exists(), "run-1 is left alone"


def test_a_single_later_run_is_numbered(data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_jobs(monkeypatch)
    run_1 = _existing_run(data_root, "baseline", 1)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    run_sweep(plan, first_run=2, dry_run=False)

    assert [call["output_dir"] for call in calls] == [run_1.parent / "run-2"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_numbered_run_is_not_overwritten_unless_asked(
    data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """Forgetting first_run would renumber from 1 and overwrite run-1: refused, with the fix named."""
    calls = _record_jobs(monkeypatch)
    run_1 = _existing_run(data_root, "baseline", 1)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="pass first_run=2") as refusal:
        run_sweep(plan, n_repeat=2, dry_run=dry_run)
    assert str(run_1) in str(refusal.value)
    assert calls == []


def test_the_suggested_first_run_follows_the_last_run_on_disk(data_root: Path, keys: None) -> None:
    for run in (1, 3):
        _existing_run(data_root, "baseline", run)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="pass first_run=4"):
        run_sweep(plan, n_repeat=2, first_run=2)


def test_an_empty_run_directory_is_not_a_run(data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run-<n> left empty -- a sweep stopped before writing -- holds nothing to protect."""
    calls = _record_jobs(monkeypatch)
    (data_root / "atacseq" / "output" / "test-model" / "baseline" / "run-1").mkdir(parents=True)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    run_sweep(plan, n_repeat=2, dry_run=False)

    assert len(calls) == 2


def test_overwrite_replaces_numbered_runs(data_root: Path, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_jobs(monkeypatch)
    run_1 = _existing_run(data_root, "baseline", 1)
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    run_sweep(plan, n_repeat=2, overwrite=True, dry_run=False)

    assert [call["output_dir"] for call in calls] == [run_1, run_1.parent / "run-2"]


@pytest.mark.parametrize("first_run", [0, -1])
def test_a_first_run_below_1_is_refused(data_root: Path, keys: None, first_run: int) -> None:
    plan = plan_sweep(data_root, "test-model", assays=["atacseq"], conditions=["baseline"])

    with pytest.raises(ValueError, match="first_run"):
        run_sweep(plan, n_repeat=2, first_run=first_run)


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
