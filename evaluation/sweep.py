"""The sweep: every assay run through every condition, one job at a time.

A *job* is one assay under one condition: one call to :func:`evaluate.run_experiment`.
A *run* is one repeat of the whole sweep.  A sweep made once writes each job to its
condition's own directory, as the CLI does; a sweep repeated N times numbers its runs from
1 and writes run *n* to ``<condition>/run-<n>/``.

``experiment.ipynb`` calls two functions from here.  :func:`plan_sweep` settles what the
sweep covers and checks it; :func:`run_sweep` runs it.  The split is what makes a typo
cheap -- an unknown assay, an unknown condition, an empty input directory or a missing
API key raises from the plan, before any job starts and before anything is spent.

Both print as they go, because a sweep that spends money should say what it is doing.
Neither does any of the work: the conditions come from :func:`conditions.build_condition`
and each job is driven by :func:`evaluate.run_experiment`.  This module only arranges
them -- which jobs, how many runs, in which order, writing where, traced under which
environment.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from analysis.corpus import get_assay
from arms_agent.tracing import tracing_enabled
from assays import ASSAY_SCHEMAS
from conditions import build_condition, condition_names, get_condition
from evaluate import run_experiment

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

#: How many of one job's records are migrated at a time.  Within a job only: the sweep
#: never starts a job before the one before it has finished.
DEFAULT_CONCURRENCY = 8

#: The keys every condition needs: the LLM to call, and the CEDAR template to migrate
#: to.  A condition that calls anything else declares it in its own ``requires_keys``,
#: which is what keeps this list from having to know what a dropped-in module does.
_REQUIRED_KEYS = ("OPENAI_API_KEY", "CEDAR_API_KEY")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepPlan:
    """What a sweep will run, and where each of its jobs reads and writes.

    Built by :func:`plan_sweep`, which is what checks it.  Holding it as a value means
    the plan can be printed, trimmed or inspected before :func:`run_sweep` acts on it.
    """

    data_root: Path
    model: str
    jobs: tuple[tuple[str, str], ...]

    @property
    def assays(self) -> list[str]:
        """The assays covered, in the order they run."""
        return list(dict.fromkeys(assay for assay, _condition in self.jobs))

    @property
    def conditions(self) -> list[str]:
        """The conditions each assay is run through, in the order they run."""
        return list(dict.fromkeys(condition for _assay, condition in self.jobs))

    @property
    def migrations(self) -> int:
        """How many record migrations one run of the sweep makes."""
        return sum(len(self.input_records(assay)) for assay, _condition in self.jobs)

    def input_records(self, assay: str) -> list[Path]:
        """Every legacy record of *assay*: what one job of it migrates."""
        return sorted(get_assay(self.data_root, assay).input_dir.glob("*.json"))

    def output_dir(self, assay: str, condition: str, run: int | None = None) -> Path:
        """Where one (assay, condition) job writes: its condition's directory, or ``run-<run>`` in it.

        *run* is ``None`` for a sweep made once, as the CLI's is, and the run's number for
        one of a repeated sweep's runs.
        """
        return get_assay(self.data_root, assay).run_output_dir(self.model, condition, run)

    def __str__(self) -> str:
        return (
            f"{len(self.assays)} assay(s) x {len(self.conditions)} condition(s) = "
            f"{len(self.jobs)} job(s), {self.migrations} record migration(s) per run"
        )


def plan_sweep(
    data_root: str | Path,
    model: str,
    *,
    assays: Sequence[str],
    conditions: Sequence[str] | None = None,
) -> SweepPlan:
    """Check what a sweep over *assays* x *conditions* would run, print its size, return it.

    Loads the API keys from the project's ``.env`` first, then raises on anything that
    would fail partway through: an unknown assay, an unknown condition, an assay with no
    input records, or a key the chosen conditions need and the environment does not have.
    A bad name should cost nothing.

    Assay outermost, so every condition of one assay finishes before the next assay
    starts and stopping early leaves whole assays comparable across conditions.

    Args:
        data_root: The root data directory, holding one directory per assay.
        model: The LLM the jobs call, which is also the directory they write under.
        assays: The assays to cover, by key -- the keys of ``assays.ASSAY_SCHEMAS``.
        conditions: The conditions to run each assay through (default: every condition
            declared under ``conditions/``, so a module dropped in is covered).

    Returns:
        The checked :class:`SweepPlan`, ready to hand to :func:`run_sweep`.

    Raises:
        ValueError: On an unknown assay or condition name.
        FileNotFoundError: If any assay has no input records to migrate.
        OSError: If a required API key is not set.
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=True)

    known = condition_names()
    if conditions is None:
        conditions = known

    if not assays or not conditions:
        raise ValueError("Nothing to run: name at least one assay and one condition.")

    unknown_assays = [name for name in assays if name not in ASSAY_SCHEMAS]
    if unknown_assays:
        raise ValueError(f"Unknown assay(s): {', '.join(unknown_assays)}")

    unknown_conditions = [name for name in conditions if name not in known]
    if unknown_conditions:
        raise ValueError(f"Unknown condition(s): {', '.join(unknown_conditions)}; expected one of {', '.join(known)}")

    # Each condition says what it calls out to, so a new one brings its own key check.
    needed = {*_REQUIRED_KEYS}.union(*(get_condition(condition).requires_keys for condition in conditions))
    missing = [key for key in sorted(needed) if not os.environ.get(key)]
    if missing:
        raise OSError(
            f"These conditions need {', '.join(missing)}, which is not set. "
            f"Put it in {_PROJECT_ROOT / '.env'} or in the environment."
        )

    plan = SweepPlan(Path(data_root), model, tuple(product(assays, conditions)))
    for assay in plan.assays:
        if not plan.input_records(assay):
            raise FileNotFoundError(f"No input records found in {get_assay(data_root, assay).input_dir}")

    print(f"Sweep: {plan}.")
    print(f"  model      {plan.model}")
    print(f"  assays     {', '.join(plan.assays)}")
    print(f"  conditions {', '.join(plan.conditions)}")
    print(f"  writing to {plan.data_root}/<assay>/output/{plan.model}/<condition>/  (run-<n>/ in it when repeated)")
    return plan


def run_sweep(
    plan: SweepPlan,
    *,
    n_repeat: int = 1,
    first_run: int = 1,
    dry_run: bool = True,
    max_concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    """Run every job in *plan*, *n_repeat* times over, printing each job as it starts.

    The only function here that spends money, and it spends nothing while *dry_run*
    stands: it lists what it would run and stops.  That is the default, so a cell run by
    accident costs nothing.

    A sweep made once -- *n_repeat* 1, *first_run* 1 -- writes each job to its condition's
    own directory, as the CLI does.  Otherwise its runs are numbered from *first_run*, and
    run *n* writes to ``<condition>/run-<n>/``.  So ``n_repeat=3`` makes runs 1 to 3, and
    ``n_repeat=2, first_run=2`` adds runs 2 and 3 beside a ``run-1`` already there.  The
    run is the outermost loop, so every job finishes once before any job runs again, and
    stopping early leaves every assay and condition with the same number of complete runs.

    Nothing already written is overwritten by a numbered run, and a condition directory
    holds one layout or the other, never both, because a reader finds run 1 by looking at
    which one is there.  So a sweep that would write into a ``run-<n>`` directory already
    holding predictions, or mix the layouts -- one run where ``run-<n>`` directories are,
    numbered runs where a single run is -- is refused before any job starts, dry run
    included.

    Args:
        plan: A plan from :func:`plan_sweep`.
        n_repeat: How many runs of the whole plan to make.
        first_run: The number the first of them takes; later runs follow on from it.
        dry_run: While true, list the jobs instead of running them.
        max_concurrency: How many of one job's records are migrated at a time.

    Raises:
        ValueError: If *n_repeat* or *first_run* is less than 1, or the sweep would
            overwrite a numbered run or mix the two layouts in a condition directory.
    """
    if n_repeat < 1:
        raise ValueError(f"n_repeat must be at least 1, not {n_repeat}.")
    if first_run < 1:
        raise ValueError(f"first_run must be at least 1, not {first_run}.")
    runs: list[int | None] = (
        [None] if n_repeat == 1 and first_run == 1 else list(range(first_run, first_run + n_repeat))
    )
    _refuse_clashes(plan, runs)
    jobs = [(run, assay, condition) for run in runs for assay, condition in plan.jobs]

    if dry_run:
        for position, (run, assay, condition) in enumerate(jobs, start=1):
            print(f"would run  {_describe(plan, position, len(jobs), run, assay, condition)}")
        print(
            f"\nDry run: nothing was run, nothing was spent ({plan}; x {n_repeat} run(s) = "
            f"{len(jobs)} job(s), {plan.migrations * n_repeat} record migration(s) in total)."
        )
        print("Pass dry_run=False to run the sweep above.")
        return

    # Every job is handed to the same worker thread: run_experiment drives its per-file
    # concurrency with asyncio.run, which needs a thread of its own to own the loop, and
    # the tracing context is a context variable, so it is entered inside that thread.
    # One worker, so the sweep waits for each job to finish before starting the next.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for position, (run, assay, condition) in enumerate(jobs, start=1):
            print(_describe(plan, position, len(jobs), run, assay, condition))
            pool.submit(_run_job, plan, run, assay, condition, max_concurrency).result()


def _refuse_clashes(plan: SweepPlan, runs: list[int | None]) -> None:
    """Raise if the sweep would overwrite a numbered run, or mix the layouts in a condition directory.

    *runs* is ``[None]`` for a single run into the condition's own directory, else the run
    numbers about to be written.
    """
    numbered = runs != [None]
    mixed: list[Path] = []
    taken: list[Path] = []
    highest = 0  # the highest run on disk holding predictions, in any of the plan's conditions
    for assay, condition in plan.jobs:
        condition_dir = plan.output_dir(assay, condition)
        if not condition_dir.is_dir():
            continue
        run_dirs = [child for child in condition_dir.iterdir() if child.is_dir() and child.name.startswith("run-")]
        has_runs = bool(run_dirs)
        has_predictions = any(condition_dir.glob("*.json"))
        for run_dir in run_dirs:
            number = run_dir.name.removeprefix("run-")
            if number.isdigit() and any(run_dir.glob("*.json")):
                highest = max(highest, int(number))
        if (not numbered and has_runs) or (numbered and has_predictions):
            mixed.append(condition_dir)
        if numbered:
            taken += [
                plan.output_dir(assay, condition, run)
                for run in runs
                if any(plan.output_dir(assay, condition, run).glob("*.json"))
            ]
    if mixed:
        held = "run-<n> directories" if not numbered else "predictions of a single run"
        raise ValueError(
            f"{'One run' if not numbered else 'Numbered runs'} cannot be written where {held} already are, "
            f"or the analyses could not tell which to read.  Move or delete these first:\n  "
            + "\n  ".join(str(path) for path in mixed)
        )
    if taken:
        raise ValueError(
            "These runs already hold predictions, and a numbered run is never overwritten.  To add runs after "
            f"the last one on disk pass first_run={highest + 1}; to redo them, move or delete them first:\n  "
            + "\n  ".join(str(path) for path in taken)
        )


def _describe(plan: SweepPlan, position: int, total: int, run: int | None, assay: str, condition: str) -> str:
    """One line saying which job this is, how big it is, and where it lands."""
    run_part = "" if run is None else f"run {run} | "
    return (
        f"[{position}/{total}] {assay} | {condition} | {run_part}"
        f"{len(plan.input_records(assay))} record(s) -> {plan.output_dir(assay, condition, run)}"
    )


def _run_job(plan: SweepPlan, run: int | None, assay: str, condition: str, max_concurrency: int) -> None:
    """Migrate every record of one assay under one condition, as part of run *run* (``None``: the only run)."""
    build_workflow, build_user_prompt = build_condition(condition)
    schema_iri = ASSAY_SCHEMAS[assay]
    with _traced_as(f"experiment-{assay}"):
        run_experiment(
            template_iri=schema_iri,
            input_dir=get_assay(plan.data_root, assay).input_dir,
            output_dir=plan.output_dir(assay, condition, run),
            workflow_factory=partial(build_workflow, model=plan.model, template_iri=schema_iri),
            user_prompt_builder=build_user_prompt,
            max_concurrency=max_concurrency,
            config={
                "tags": ["experiment", condition],
                "metadata": {"assay": assay, "condition": condition, "run": run or 1, "template_iri": schema_iri},
            },
        )


def _traced_as(environment: str) -> AbstractContextManager[Any]:
    """File the traces of the job inside this context under *environment* in Langfuse.

    One environment per assay, ``experiment-<assay>``, so a sweep can be read one assay
    at a time in the Langfuse UI; the condition rides along as a trace tag.  Whatever
    ``.env`` set for ``LANGFUSE_TRACING_ENVIRONMENT`` is overridden here.
    """
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = environment
    if not tracing_enabled():
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(environment=environment)
