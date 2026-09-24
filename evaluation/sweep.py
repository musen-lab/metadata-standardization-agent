"""The sweep: every assay run through every condition, one run at a time.

``experiment.ipynb`` calls two functions from here.  :func:`plan_sweep` settles what the
sweep covers and checks it; :func:`run_sweep` runs it.  The split is what makes a typo
cheap -- an unknown assay, an unknown condition, an empty input directory or a missing
API key raises from the plan, before any run starts and before anything is spent.

Both print as they go, because a sweep that spends money should say what it is doing.
Neither does any of the work: the conditions come from :func:`conditions.build_condition`
and each run is driven by :func:`evaluate.run_experiment`.  This module only arranges
them -- which runs, in which order, writing where, traced under which environment.
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

#: How many of one run's records are migrated at a time.  Within a run only: the sweep
#: never starts a run before the one before it has finished.
DEFAULT_CONCURRENCY = 8

#: The keys every condition needs: the LLM to call, and the CEDAR template to migrate
#: to.  A condition that calls anything else declares it in its own ``requires_keys``,
#: which is what keeps this list from having to know what a dropped-in module does.
_REQUIRED_KEYS = ("OPENAI_API_KEY", "CEDAR_API_KEY")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepPlan:
    """What a sweep will run, and where each of its runs reads and writes.

    Built by :func:`plan_sweep`, which is what checks it.  Holding it as a value means
    the plan can be printed, trimmed or inspected before :func:`run_sweep` acts on it.
    """

    data_root: Path
    model: str
    runs: tuple[tuple[str, str], ...]

    @property
    def assays(self) -> list[str]:
        """The assays covered, in the order they run."""
        return list(dict.fromkeys(assay for assay, _run_type in self.runs))

    @property
    def run_types(self) -> list[str]:
        """The conditions each assay is run through, in the order they run."""
        return list(dict.fromkeys(run_type for _assay, run_type in self.runs))

    @property
    def migrations(self) -> int:
        """How many record migrations the whole sweep makes."""
        return sum(len(self.input_records(assay)) for assay, _run_type in self.runs)

    def input_records(self, assay: str) -> list[Path]:
        """Every legacy record of *assay*: what one run of it migrates."""
        return sorted(get_assay(self.data_root, assay).input_dir.glob("*.json"))

    def output_dir(self, assay: str, run_type: str) -> Path:
        """Where one (assay, condition) run writes, named as the CLI names it."""
        return get_assay(self.data_root, assay).output_dir(self.model, run_type)

    def __str__(self) -> str:
        return (
            f"{len(self.assays)} assay(s) x {len(self.run_types)} condition(s) = "
            f"{len(self.runs)} run(s), {self.migrations} record migration(s) in total"
        )


def plan_sweep(
    data_root: str | Path,
    model: str,
    *,
    assays: Sequence[str],
    run_types: Sequence[str] | None = None,
) -> SweepPlan:
    """Check what a sweep over *assays* x *run_types* would run, print its size, return it.

    Loads the API keys from the project's ``.env`` first, then raises on anything that
    would fail partway through: an unknown assay, an unknown condition, an assay with no
    input records, or a key the chosen conditions need and the environment does not have.
    A bad name should cost nothing.

    Assay outermost, so every condition of one assay finishes before the next assay
    starts and stopping early leaves whole assays comparable across conditions.

    Args:
        data_root: The root data directory, holding one directory per assay.
        model: The LLM the runs call, which is also the directory they write under.
        assays: The assays to cover, by key -- the keys of ``assays.ASSAY_SCHEMAS``.
        run_types: The conditions to run each assay through (default: every condition
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
    if run_types is None:
        run_types = known

    if not assays or not run_types:
        raise ValueError("Nothing to run: name at least one assay and one condition.")

    unknown_assays = [name for name in assays if name not in ASSAY_SCHEMAS]
    if unknown_assays:
        raise ValueError(f"Unknown assay(s): {', '.join(unknown_assays)}")

    unknown_run_types = [name for name in run_types if name not in known]
    if unknown_run_types:
        raise ValueError(f"Unknown run type(s): {', '.join(unknown_run_types)}; expected one of {', '.join(known)}")

    # Each condition says what it calls out to, so a new one brings its own key check.
    needed = {*_REQUIRED_KEYS}.union(*(get_condition(run_type).requires_keys for run_type in run_types))
    missing = [key for key in sorted(needed) if not os.environ.get(key)]
    if missing:
        raise OSError(
            f"These conditions need {', '.join(missing)}, which is not set. "
            f"Put it in {_PROJECT_ROOT / '.env'} or in the environment."
        )

    plan = SweepPlan(Path(data_root), model, tuple(product(assays, run_types)))
    for assay in plan.assays:
        if not plan.input_records(assay):
            raise FileNotFoundError(f"No input records found in {get_assay(data_root, assay).input_dir}")

    print(f"Sweep: {plan}.")
    print(f"  model      {plan.model}")
    print(f"  assays     {', '.join(plan.assays)}")
    print(f"  conditions {', '.join(plan.run_types)}")
    print(f"  writing to {plan.data_root}/<assay>/output/{plan.model}/<condition>/")
    return plan


def run_sweep(plan: SweepPlan, *, dry_run: bool = True, max_concurrency: int = DEFAULT_CONCURRENCY) -> None:
    """Run every run in *plan*, in order, printing each one as it starts.

    The only function here that spends money, and it spends nothing while *dry_run*
    stands: it lists what it would run and stops.  That is the default, so a cell run by
    accident costs nothing.

    Args:
        plan: A plan from :func:`plan_sweep`.
        dry_run: While true, list the runs instead of making them.
        max_concurrency: How many of one run's records are migrated at a time.
    """
    if dry_run:
        for position, (assay, run_type) in enumerate(plan.runs, start=1):
            print(f"would run  {_describe(plan, position, assay, run_type)}")
        print(f"\nDry run: nothing was run, nothing was spent ({plan}).")
        print("Pass dry_run=False to run the sweep above.")
        return

    # Every run is handed to the same worker thread: run_experiment drives its per-file
    # concurrency with asyncio.run, which needs a thread of its own to own the loop, and
    # the tracing context is a context variable, so it is entered inside that thread.
    # One worker, so the sweep waits for each run to finish before starting the next.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for position, (assay, run_type) in enumerate(plan.runs, start=1):
            print(_describe(plan, position, assay, run_type))
            pool.submit(_run_one, plan, assay, run_type, max_concurrency).result()


def _describe(plan: SweepPlan, position: int, assay: str, run_type: str) -> str:
    """One line saying which run this is, how big it is, and where it lands."""
    return (
        f"[{position}/{len(plan.runs)}] {assay} | {run_type} | "
        f"{len(plan.input_records(assay))} record(s) -> {plan.output_dir(assay, run_type)}"
    )


def _run_one(plan: SweepPlan, assay: str, run_type: str, max_concurrency: int) -> None:
    """Migrate every record of one assay under one condition."""
    build_workflow, build_user_prompt = build_condition(run_type)
    schema_iri = ASSAY_SCHEMAS[assay]
    with _traced_as(f"experiment-{assay}"):
        run_experiment(
            template_iri=schema_iri,
            input_dir=get_assay(plan.data_root, assay).input_dir,
            output_dir=plan.output_dir(assay, run_type),
            workflow_factory=partial(build_workflow, model=plan.model, template_iri=schema_iri),
            user_prompt_builder=build_user_prompt,
            max_concurrency=max_concurrency,
            config={
                "tags": ["experiment", run_type],
                "metadata": {"assay": assay, "run_type": run_type, "template_iri": schema_iri},
            },
        )


def _traced_as(environment: str) -> AbstractContextManager[Any]:
    """File the traces of the run inside this context under *environment* in Langfuse.

    One environment per assay, ``experiment-<assay>``, so a sweep can be read one assay
    at a time in the Langfuse UI; the condition rides along as a trace tag.  Whatever
    ``.env`` set for ``LANGFUSE_TRACING_ENVIRONMENT`` is overridden here.
    """
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = environment
    if not tracing_enabled():
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(environment=environment)
