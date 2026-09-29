"""Where the evaluation files live, how to walk them, and how a field instance is keyed.

Every table in :mod:`analysis.data_analysis` and every test in
:mod:`analysis.significance` asks a different question of the same corpus, but they
all reach it the same way: for each assay in ``ASSAY_ORDER``, pair each gold record
with its counterpart under ``data/<assay>/output/<model>/<condition>/`` -- or, for the
do-nothing reference point, under ``data/<assay>/input/``.  That walk is written once
here so the analyses differ only in what they count, not in which files they see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis.metrics import _get_ontology_constrained_fields, _get_permissible_values
from assays import ASSAY_ORDER

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Assay:
    """The files belonging to one assay under a ``data/`` root."""

    key: str
    label: str
    root: Path

    @property
    def schema_path(self) -> Path:
        """The CEDAR template specification this assay's records are migrated to."""
        return self.root / "schemas" / f"{self.key}.json"

    @property
    def gold_dir(self) -> Path:
        """The curated reference records."""
        return self.root / self.key / "gold"

    @property
    def input_dir(self) -> Path:
        """The legacy records, as they were before any migration."""
        return self.root / self.key / "input"

    def run_output_dir(self, model: str, condition: str, run: int | None) -> Path:
        """Where run *run* of *condition* is written: the layout, stated rather than looked up.

        A condition run once keeps its predictions in its own directory, ``baseline`` for
        the prompt-only arm and ``arms-agent`` for ARMS; pass ``None`` for that.  A
        condition run several times holds one ``run-<n>`` directory per run instead; pass
        the run's number.  Writers use this, since what they are about to write decides
        the layout; readers use :meth:`output_dir`, which finds it on disk.
        """
        condition_dir = self.root / self.key / "output" / model / condition
        return condition_dir if run is None else condition_dir / f"run-{run}"

    def output_dir(self, model: str, condition: str, *, run: int = 1) -> Path:
        """The predictions *model* wrote under *condition* in its *run*-th run.

        Found on disk: when the condition's directory holds ``run-<n>`` directories it was
        run several times, and run *run* is ``run-<run>``; otherwise it was run once, and
        its one run -- run 1 -- is the directory itself.  So ``run=1``, the default every
        analysis reads, is the first run whichever way the condition was made, and a run
        that was never made is a directory that does not exist.
        """
        condition_dir = self.root / self.key / "output" / model / condition
        repeated = condition_dir.is_dir() and any(
            child.is_dir() and child.name.startswith("run-") for child in condition_dir.iterdir()
        )
        return self.run_output_dir(model, condition, run if repeated or run != 1 else None)

    def has_predictions(self, model: str, condition: str, *, run: int = 1) -> bool:
        """Whether *model* wrote any prediction under *condition* in its *run*-th run.

        A run not made yet has none, so it can be told apart from a run that left some
        records out -- the one worth warning about -- without scoring either.
        """
        return any(self.output_dir(model, condition, run=run).glob("*.json"))

    @property
    def has_gold(self) -> bool:
        """Whether this assay has both a schema and gold records to score against.

        Callers skip an assay without them rather than raising, so a partial data root
        (a pilot over two or three assays, say) can still be analysed.
        """
        return self.schema_path.exists() and self.gold_dir.is_dir()

    def ontology_fields(self) -> set[str]:
        """The field names this assay's schema constrains to an ontology or a branch."""
        return set(_get_ontology_constrained_fields(self.schema_path))

    def permissible_values(self) -> dict[str, list[str]]:
        """Field name -> the values this assay's schema enumerates, where it enumerates any."""
        return _get_permissible_values(self.schema_path)


_ASSAY_LABELS = dict(ASSAY_ORDER)


def get_assay(data_root: str | Path, assay_key: str) -> Assay:
    """The :class:`Assay` named *assay_key* under *data_root*.

    An assay outside ``ASSAY_ORDER`` gets its key as its label, so a one-off directory
    can still be analysed.
    """
    return Assay(key=assay_key, label=_ASSAY_LABELS.get(assay_key, assay_key), root=Path(data_root))


def iter_assays(data_root: str | Path) -> Iterator[Assay]:
    """Yield one :class:`Assay` per entry in ``ASSAY_ORDER``, in that order.

    The order is the one the paper's tables use, so a table built by appending a row
    per assay comes out already sorted.
    """
    for key, _label in ASSAY_ORDER:
        yield get_assay(data_root, key)


#: One distinct correction: assay, field name, and the gold value serialized so that
#: values of any JSON type can be compared and used as a dict key.
ValueKey = tuple[str, str, str]


def value_key(assay_key: str, field: str, gold_val: Any) -> ValueKey:
    """Identify the correction *field* -> *gold_val* within *assay_key*.

    The corpus repeats the same correction across many records, so this key is what
    turns "839 records" into "how many *distinct* things were asked for".  Both the
    repetition tables and the effective-sample-size estimate cluster on it, and they
    have to cluster on the same thing to be comparable.
    """
    return assay_key, field, json.dumps(gold_val, sort_keys=True)


def matches_field_type(field: str, ontology_fields: set[str], field_type: str | None) -> bool:
    """Whether *field* belongs to the requested half of the schema.

    *field_type* is ``"ontology"``, ``"non_ontology"``, or ``None`` for all fields.
    """
    if field_type == "ontology":
        return field in ontology_fields
    if field_type == "non_ontology":
        return field not in ontology_fields
    return True


def load_record(path: Path) -> dict[str, Any]:
    """Read one JSON metadata record."""
    with open(path) as f:
        return json.load(f)


def iter_records(directory: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(path, record)`` for every JSON file in *directory*, in filename order.

    A directory that does not exist yields nothing, which is what makes an assay with
    no predictions for the requested model a skip rather than an error.
    """
    for path in sorted(directory.glob("*.json")):
        yield path, load_record(path)


def iter_pairs(gold_dir: Path, other_dir: Path) -> Iterator[tuple[Path, dict[str, Any], dict[str, Any] | None]]:
    """Yield ``(gold_file, gold, other)`` for every gold record in *gold_dir*.

    Driven by the gold files rather than by *other_dir*, so a gold record with no
    counterpart arrives as ``other=None``: a skip the caller can see and count rather
    than an invisible omission.  That matters wherever the totals have to correspond
    to the gold fields actually evaluated.
    """
    for gold_file, gold in iter_records(gold_dir):
        other_file = other_dir / gold_file.name
        yield gold_file, gold, load_record(other_file) if other_file.exists() else None


def iter_predictions(output_dir: Path, gold_dir: Path) -> Iterator[tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Yield ``(pred_file, predicted, gold)`` for every prediction with a gold record.

    The mirror of :func:`iter_pairs`: driven by the prediction files, so a gold record
    that was never predicted does not appear at all.  Use it where the question is
    "how good are the predictions that exist" rather than "how much of gold was
    reproduced".
    """
    for pred_file, predicted in iter_records(output_dir):
        gold_file = gold_dir / pred_file.name
        if not gold_file.exists():
            continue
        yield pred_file, predicted, load_record(gold_file)
