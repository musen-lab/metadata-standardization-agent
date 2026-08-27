"""What a run produced, recorded into its trace.

The callback handler traces how the agent worked -- the calls, their timings, their
cost.  These two events are what it worked out: the record it migrated, and its own
account of how it decided each field.  Both are written from inside a node body, so
both depend on the span :mod:`spans` puts in context.

They cost no tokens.  An event is an HTTPS post to Langfuse; nothing here is sent to a
model, and nothing here is ever read back into a prompt.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from arms_agent.tracing.client import recording_client

logger = logging.getLogger(__name__)

# The observation names the record and the log are stored under, and the ones to filter
# on when reading them back out with ``api.observations.get_many(name=...)``.
MIGRATED_RECORD_OBSERVATION = "migrated_record"
PROCESSING_LOG_OBSERVATION = "processing_log"


def _is_empty(value: Any) -> bool:
    """Return True when *value* carries no information.

    Nesting is followed, because a CEDAR element whose children are all null is itself
    empty and counting it as populated would overstate how much of the record the agent
    filled.  ``False`` and ``0`` are values, not absences, so neither counts as empty.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(_is_empty(child) for child in value.values())
    if isinstance(value, list):
        return all(_is_empty(item) for item in value)
    return False


def record_migrated_record(record: dict[str, Any], *, source: str) -> None:
    """Store the migrated record in the current trace.

    The record is what the run produced -- ``structured_response.record`` on the
    validated path -- so it is recorded next to the processing log, letting a trace show
    both what the agent decided and what it decided it into.  Retrievable later with
    ``api.observations.get_many(name=MIGRATED_RECORD_OBSERVATION, parse_io_as_json=True)``.
    The field counts go in ``metadata`` so a sweep can find thin records without
    fetching every one, and *source* says which extraction path produced the record.

    Record this before the processing log: Langfuse orders sibling events by creation
    time, and a trace reads better with the answer above the working.

    An empty record is recorded at ``WARNING`` so runs that produced none can be found.
    Nothing is sent when tracing is off or when no span is in context.
    """
    client = recording_client("migrated record")
    if client is None:
        return

    populated = sum(1 for value in record.values() if not _is_empty(value))
    client.create_event(
        name=MIGRATED_RECORD_OBSERVATION,
        output=record,
        metadata={"source": source, "fields": len(record), "populated": populated},
        level="WARNING" if not record else "DEFAULT",
        status_message=None if record else "The agent produced no record",
    )
    logger.debug("Traced a %d-field migrated record, %d populated (source=%s)", len(record), populated, source)


def record_processing_log(decisions: list[dict[str, Any]], *, source: str) -> None:
    """Store the agent's processing log in the current trace.

    The log is the agent's own account of the migration, so it is recorded as one
    observation per run holding every entry, retrievable later with
    ``api.observations.get_many(name=PROCESSING_LOG_OBSERVATION, parse_io_as_json=True)``.
    The resolution counts go in ``metadata`` so a sweep can be summarised without
    fetching every entry, and *source* says which extraction path produced the log.

    An empty log is recorded at ``WARNING`` so runs that produced none can be found.
    Nothing is sent when tracing is off or when no span is in context.
    """
    client = recording_client("processing log")
    if client is None:
        return

    resolutions = Counter(str(entry.get("resolution", "unknown")) for entry in decisions)
    client.create_event(
        name=PROCESSING_LOG_OBSERVATION,
        output=decisions,
        metadata={"source": source, "entries": len(decisions), "resolutions": dict(resolutions)},
        level="WARNING" if not decisions else "DEFAULT",
        status_message=None if decisions else "The agent produced no processing log",
    )
    logger.debug("Traced %d processing-log entries (source=%s)", len(decisions), source)
