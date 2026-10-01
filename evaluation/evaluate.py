"""Orchestration functions for running experiments and computing metrics."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import openai

from arms_agent.token_tracker import TokenUsageTracker
from arms_agent.tracing import flush_tracing, instrument, traced_run
from arms_agent.workflow import RECURSION_LIMIT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

#: The errors that say the endpoint, not the record, is at fault: the connection is gone,
#: the key is refused or out of budget, or the server is failing.  Each is raised after
#: the OpenAI client's own retries, and every record after it would fail the same way, so
#: one of them stops the run instead of being recorded against its record.
OUTAGE_ERRORS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.RateLimitError,
    openai.InternalServerError,
    ConnectionError,
)


@dataclass(frozen=True)
class RecordFailure:
    """One record whose migration raised, and so has no prediction."""

    input_file: Path
    #: The exception's type and message, on one line.
    error: str
    #: The file holding every failed attempt at this record, traceback included.
    log: Path


@dataclass(frozen=True)
class ExperimentResult:
    """What one call to :func:`run_experiment` wrote, and which records it could not migrate."""

    written: list[Path]
    failed: list[RecordFailure]


def _usage_record(tracker: TokenUsageTracker) -> dict[str, Any]:
    """Return a tracker's accumulated usage as a JSON-serialisable dict."""
    return {
        "prompt_tokens": tracker.prompt_tokens,
        "cached_tokens": tracker.cached_tokens,
        "completion_tokens": tracker.completion_tokens,
        "reasoning_tokens": tracker.reasoning_tokens,
        "total_tokens": tracker.total_tokens,
        "estimated_cost_usd": round(tracker.total_cost, 6),
    }


def _sum_usage(trackers: Iterable[TokenUsageTracker]) -> TokenUsageTracker:
    """Total several per-file trackers into one, to reuse its summary formatting."""
    combined = TokenUsageTracker()
    for tracker in trackers:
        combined.prompt_tokens += tracker.prompt_tokens
        combined.cached_tokens += tracker.cached_tokens
        combined.completion_tokens += tracker.completion_tokens
        combined.reasoning_tokens += tracker.reasoning_tokens
        combined.total_tokens += tracker.total_tokens
        combined.total_cost += tracker.total_cost
    return combined


def _tracker_from_record(record: dict[str, Any]) -> TokenUsageTracker:
    """Rebuild a tracker from a usage record written by an earlier call, so it can be summed."""
    tracker = TokenUsageTracker()
    tracker.prompt_tokens = record["prompt_tokens"]
    tracker.cached_tokens = record["cached_tokens"]
    tracker.completion_tokens = record["completion_tokens"]
    tracker.reasoning_tokens = record["reasoning_tokens"]
    tracker.total_tokens = record["total_tokens"]
    tracker.total_cost = record["estimated_cost_usd"]
    return tracker


def _write_json(path: Path, data: Any) -> None:
    """Write *data* to *path* whole or not at all, so a crash never leaves half a file behind."""
    staging = path.with_name(f".{path.name}.tmp")
    with open(staging, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(staging, path)


def pending_records(input_dir: Path, output_dir: Path) -> list[Path]:
    """The records of *input_dir* with no prediction in *output_dir* yet: what a resumed run migrates."""
    return [path for path in sorted(input_dir.glob("*.json")) if not (output_dir / path.name).exists()]


def _is_run_dir(path: Path) -> bool:
    """Whether *path* is one of a repeated condition's ``run-<n>`` directories."""
    number = path.name.removeprefix("run-")
    return path.name.startswith("run-") and number.isdigit()


def _holds_predictions(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.json"))


def find_output_clashes(output_dirs: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """The directories in *output_dirs* that cannot be written as they are: ``(mixed, occupied)``.

    *occupied* already hold predictions, which writing would overwrite.  *mixed* would put
    both layouts in one condition directory -- a ``run-<n>`` directory whose condition
    directory holds a single run of its own, or a condition directory that already holds
    ``run-<n>`` directories -- and then a reader could not tell which is run 1.  A
    directory that does not exist yet, or exists but is empty, clashes with nothing.
    """
    mixed: list[Path] = []
    occupied: list[Path] = []
    for output_dir in output_dirs:
        if _is_run_dir(output_dir):
            if _holds_predictions(output_dir.parent):
                mixed.append(output_dir)
        elif output_dir.is_dir() and any(child.is_dir() and _is_run_dir(child) for child in output_dir.iterdir()):
            mixed.append(output_dir)
        if _holds_predictions(output_dir):
            occupied.append(output_dir)
    return mixed, occupied


def refuse_output_clashes(output_dirs: Iterable[Path], *, overwrite: bool = False, hint: str = "") -> None:
    """Raise before anything is written if writing to *output_dirs* would lose or confuse predictions.

    Predictions already there are overwritten only when *overwrite* is true.  Mixing the two
    layouts in one condition directory is refused either way, since no reader could then
    tell which run is run 1.  *hint* is added to the message about occupied directories:
    what the caller can do instead.

    Raises:
        ValueError: Naming every directory that clashes.
    """
    mixed, occupied = find_output_clashes(output_dirs)
    if mixed:
        raise ValueError(
            "Writing here would put a single run and numbered runs in one condition folder, and the "
            "analyses could not tell which is run 1.  Move or delete one of them first:\n  "
            + "\n  ".join(str(path) for path in mixed)
        )
    if occupied and not overwrite:
        raise ValueError(
            "These folders already hold predictions, and nothing is overwritten unless asked.  "
            f"{hint}Move or delete them first, or pass overwrite=True to replace them:\n  "
            + "\n  ".join(str(path) for path in occupied)
        )


def run_experiment(
    template_iri: str,
    input_dir: Path,
    output_dir: Path,
    workflow_factory: Callable[[], CompiledStateGraph],
    user_prompt_builder: Callable[[dict[str, Any], str], str],
    *,
    config: dict[str, Any] | None = None,
    max_concurrency: int = 5,
    overwrite: bool = False,
    resume: bool = False,
) -> ExperimentResult:
    """Run the migration workflow on all JSON files in *input_dir*.

    The workflow is built once via *workflow_factory* and reused for every
    input file.  Up to *max_concurrency* files are processed in parallel.
    Each result is written to *output_dir* with the same filename as the
    input.  The user message is constructed by *user_prompt_builder*.

    Token usage and estimated cost are recorded per file under
    ``<output_dir>/usage/``, with the sweep total in ``usage/_sweep_total.json``.

    Nothing already in *output_dir* is overwritten unless *overwrite* is true, and a single
    run is never written beside numbered ones: :func:`refuse_output_clashes` raises before
    the first record is migrated.  With *resume*, the records that already have a
    prediction are skipped and only the rest are migrated, which is how a run stopped
    partway -- by a dropped connection, say -- is finished.  A prediction is written last
    and whole, so one that exists is one whose record finished.

    A record whose migration raises does not stop the others.  It gets no prediction, so
    *resume* retries it, and the attempt -- error, traceback and the tokens it spent -- is
    added to ``<output_dir>/failures/<record>.json``, which keeps every failed attempt even
    after a later one succeeds.  Its tokens are left out of the sweep total, which counts
    the predictions.  An error in :data:`OUTAGE_ERRORS` is different: every record after it
    would fail the same way, so no further record is started, the ones already running are
    let finish, and the error is raised once they have.

    Returns what was written and which records failed.

    Raises:
        ValueError: If both *overwrite* and *resume* are set, or the output would clash.
        Exception: The first of :data:`OUTAGE_ERRORS` any record raised, after the records
            already running have finished and the sweep total is written.
    """
    if overwrite and resume:
        raise ValueError("Pass overwrite=True to redo every record or resume=True to finish the rest, not both.")
    refuse_output_clashes([output_dir], overwrite=overwrite or resume)
    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        logger.warning("No *.json files found in %s", input_dir)
        return ExperimentResult(written=[], failed=[])

    to_run = pending_records(input_dir, output_dir) if resume else input_files
    done = [path for path in input_files if path not in to_run]
    if done:
        logger.info("Resuming %s: %d record(s) already done, %d to run", output_dir, len(done), len(to_run))

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = workflow_factory()
    # Filled by the first record to hit an outage; every record not yet started then skips.
    outages: list[Exception] = []

    async def _run_all() -> list[tuple[Path, TokenUsageTracker] | RecordFailure | None]:
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [
            _process_file(
                workflow, input_file, output_dir, template_iri, user_prompt_builder, config, semaphore, outages
            )
            for input_file in to_run
        ]
        return list(await asyncio.gather(*tasks))

    try:
        outcomes = asyncio.run(_run_all())
    finally:
        # One flush for the whole sweep; per-file flushing would serialise the
        # exporter against the concurrent runs.
        flush_tracing()

    results = [outcome for outcome in outcomes if isinstance(outcome, tuple)]
    failed = [outcome for outcome in outcomes if isinstance(outcome, RecordFailure)]

    # The total covers every record in the directory, so a resumed run's total is the
    # whole run's and not just the part that finished last.
    trackers = [tracker for _, tracker in results] + _earlier_usage(done, output_dir)
    total = _sum_usage(trackers)
    total_path = output_dir / "usage" / "_sweep_total.json"
    total_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(total_path, {"files": len(trackers), **_usage_record(total)})
    logger.info("Sweep usage over %d file(s): %s", len(trackers), total.usage_summary())
    if failed:
        logger.error("%d record(s) of %s failed; their errors are in %s", len(failed), output_dir, failed[0].log.parent)

    if outages:
        unfinished = len(to_run) - len(results)
        outage = outages[0]
        outage.add_note(
            f"The endpoint failed, so {output_dir} was stopped with {unfinished} record(s) left unmigrated.  "
            "Once it is back, run the same call again with resume=True to finish them."
        )
        raise outage

    return ExperimentResult(written=[output_path for output_path, _ in results], failed=failed)


def _earlier_usage(done: list[Path], output_dir: Path) -> list[TokenUsageTracker]:
    """The usage an earlier call recorded for each record in *done*, skipping any it did not record."""
    trackers = []
    for input_file in done:
        usage_path = output_dir / "usage" / input_file.name
        if not usage_path.exists():
            logger.warning("No usage recorded for %s; the run total leaves it out", input_file.name)
            continue
        with open(usage_path) as f:
            trackers.append(_tracker_from_record(json.load(f)))
    return trackers


async def _process_file(
    workflow: CompiledStateGraph,
    input_file: Path,
    output_dir: Path,
    template_iri: str,
    user_prompt_builder: Callable[[dict[str, Any], str], str],
    config: dict[str, Any] | None,
    semaphore: asyncio.Semaphore,
    outages: list[Exception],
) -> tuple[Path, TokenUsageTracker] | RecordFailure | None:
    """Process a single input file through the migration workflow.

    Acquires *semaphore* before invoking the workflow so that at most
    *max_concurrency* files are processed in parallel.

    Returns the output path and this file's token usage, or the failure when the
    workflow raised.  An outage is appended to *outages* instead, and ``None`` is
    returned, as it is for a file not started because an outage came first.
    """
    from langchain_core.messages import HumanMessage

    async with semaphore:
        if outages:
            return None
        task_name = asyncio.current_task().get_name()
        logger.info("[%s] Processing %s", task_name, input_file.name)
        with open(input_file) as f:
            legacy_metadata = json.load(f)

        user_message = user_prompt_builder(legacy_metadata, template_iri)

        tracker = TokenUsageTracker()
        run_config = dict(config) if config else {}
        run_config.setdefault("recursion_limit", RECURSION_LIMIT)
        run_config["run_name"] = f"evaluate-{input_file.stem}"
        run_config.setdefault("tags", [])
        run_config["tags"] = [*run_config["tags"], input_file.stem]
        run_config.setdefault("metadata", {})
        run_config["metadata"] = {**run_config["metadata"], "input_file": input_file.name}
        # A fresh list, so counting this file's tokens never touches the shared config.
        run_config["callbacks"] = [*(run_config.get("callbacks") or []), tracker]
        # Each file gets its own handler so concurrent runs keep separate traces.
        run_config = instrument(run_config)

        # Entered inside this file's own task, so the tracing context each file
        # attaches stays private to it while files are processed concurrently.
        try:
            for attempt in range(2):
                try:
                    with traced_run(
                        run_config["run_name"], {"input_file": input_file.name, "template_iri": template_iri}
                    ):
                        result = await workflow.ainvoke(
                            {
                                "messages": [HumanMessage(content=user_message)],
                                "cedar_template_iri": template_iri,
                            },
                            config=run_config,
                        )
                    break
                except ValueError as error:
                    if str(error) != "Agent produced no text response." or attempt:
                        raise
                    logger.warning("[%s] %s returned no answer; retrying once", task_name, input_file.name)
        except OUTAGE_ERRORS as error:
            logger.error("[%s] %s: the endpoint failed: %s", task_name, input_file.name, _one_line(error))
            outages.append(error)
            return None
        except Exception as error:
            logger.error("[%s] %s failed: %s", task_name, input_file.name, _one_line(error))
            return _record_failure(input_file, output_dir, error, tracker)

        # The processing log goes in a sibling directory so that *output_dir* keeps
        # one file per input, matching the gold standard for evaluation.
        decisions = result.get("decisions") or []
        decisions_path = output_dir / "decisions" / input_file.name
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(decisions_path, decisions)
        if not decisions:
            logger.warning("[%s] No processing-log entries for %s", task_name, input_file.name)

        # Usage goes beside the processing log, for the same reason: *output_dir*
        # keeps one file per input so it lines up with the gold standard.
        usage_path = output_dir / "usage" / input_file.name
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(usage_path, {"input_file": input_file.name, **_usage_record(tracker)})
        logger.info("[%s] %s: %s", task_name, input_file.name, tracker.usage_summary())

        # Written last, because a resumed run takes an existing prediction to mean the
        # record finished: its processing log and usage are then already on disk.
        output_path = output_dir / input_file.name
        _write_json(output_path, result["metadata"])
        logger.info("[%s] Wrote %s", task_name, output_path)

        return output_path, tracker


def _one_line(error: BaseException) -> str:
    """The exception's type and the first line of its message."""
    message = str(error).strip().splitlines()
    return f"{type(error).__name__}: {message[0]}" if message else type(error).__name__


def _record_failure(input_file: Path, output_dir: Path, error: Exception, tracker: TokenUsageTracker) -> RecordFailure:
    """Add this failed attempt at *input_file* to its failures file, beside the processing logs.

    Earlier attempts are kept, so a record that failed once and was then resumed to a
    prediction still shows that it failed.
    """
    log = output_dir / "failures" / input_file.name
    log.parent.mkdir(parents=True, exist_ok=True)
    attempts = json.loads(log.read_text()) if log.exists() else []
    attempts.append(
        {
            "input_file": input_file.name,
            "failed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "error": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "usage": _usage_record(tracker),
        }
    )
    _write_json(log, attempts)
    return RecordFailure(input_file=input_file, error=_one_line(error), log=log)
