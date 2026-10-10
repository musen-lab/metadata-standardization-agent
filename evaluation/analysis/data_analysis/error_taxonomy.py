"""What each counted error looks like, one error at a time.

:mod:`analysis.data_analysis.errors` labels a mismatch by its *shape* -- a formatting
difference, a boolean written the other way.  This module asks the next question: what did
gold hold, what did the run assert, and what does the legacy record say about how the run
got there?  It reaches for the evidence that can settle it -- the record the run was given,
the template's permissible values, and, when the run kept one, the run's own decision log.

Every error is labelled at two levels, and carries both:

* its **category** is the confusion case, named for what the run did:
  :data:`SUBSTITUTIONS`, :data:`OMISSIONS`, :data:`INSERTIONS`.
* its **sub-category** says which of the several ways that came about.

Omissions require legacy evidence that establishes the reference value for the target
field. Related context alone is insufficient. Insertions split on whether the legacy
record holds the asserted value.
Substitutions first check representation equivalence or explicitly confirmed term
mappings. Remaining substitutions are candidate wrong mappings when the asserted value
is found anywhere in the legacy record, including the same field or a renamed source
field. Different values are asserted values not found in the legacy record.
These labels describe reference disagreements, including cases where the reference
preserves a legacy value that does not meet the specification. External-gap candidates
lack sufficient legacy evidence, even when a method or kit provides a starting hint.
External provenance still requires separate verification.

**One row per error.**  The category *is* the confusion case, so a substitution has one
label rather than one per side, and the frame holds one row per (record, field).  Which
side each row costs is the ``costs`` column, and ``FP`` and ``FN`` are still recoverable
from it: :func:`reconcile_with_confusion` checks that they are.

Every row carries a *pointer*: the gold file, the prediction file and the field, so any
label can be opened and argued with.  Nothing here changes what counts as wrong -- that
stays :func:`~analysis.metrics.matching._classify_field`, the rule the scores use.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from analysis.corpus import iter_assays, load_record
from analysis.metrics import _is_missing
from analysis.metrics.matching import DELETION, INSERTION, SUBSTITUTION, _classify_field

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pandas as pd

    from analysis.corpus import Assay

#: Which side of the score a row costs.  Read off the confusion case rather than stored
#: twice: an insertion can only cost precision, a deletion only recall, a substitution both.
COSTS_PRECISION = "precision"
COSTS_RECALL = "recall"
COSTS_BOTH = "precision and recall"

COSTS_BY_CASE = {INSERTION: COSTS_PRECISION, DELETION: COSTS_RECALL, SUBSTITUTION: COSTS_BOTH}

#: The whole corpus, where a per-assay frame carries a pooled row beside its assays.
POOLED_ASSAY = "All assays"

# ---------------------------------------------------------------------------
# The categories: the confusion case, named for what the run did.
# ---------------------------------------------------------------------------

#: Gold holds a value and the run asserted a different one.
SUBSTITUTIONS = "substitutions"

#: Gold holds a value and the run asserted nothing.
OMISSIONS = "omissions"

#: Gold leaves the field blank and the run asserted something.
INSERTIONS = "insertions"

CATEGORIES = (SUBSTITUTIONS, OMISSIONS, INSERTIONS)

CATEGORY_BY_CASE = {SUBSTITUTION: SUBSTITUTIONS, DELETION: OMISSIONS, INSERTION: INSERTIONS}

#: Which confusion cells a category lands in, in the metric's own notation.  The same fact
#: the ``costs`` column carries in words, kept separately because a figure has room for
#: ``FP + FN`` where it has none for "precision and recall", and a reader coming from the
#: precision/recall tables is already holding the short form.
CONFUSION_CELLS_BY_CATEGORY = {
    SUBSTITUTIONS: "FP + FN",
    OMISSIONS: "FN",
    INSERTIONS: "FP",
}

# ---------------------------------------------------------------------------
# Sub-categories of a substitution.
# ---------------------------------------------------------------------------

#: The asserted value differs from the reference but conveys the same information.
NEAR_MATCH = "near_match"

#: A candidate mapping error: the asserted value is found anywhere in the legacy record,
#: differs from the reference, and does not qualify as a near match.
WRONG_MAPPING = "wrong_mapping"

#: The asserted value differs from the reference, is not a near match, and is not found
#: anywhere in the legacy record.
DIFFERENT_VALUE = "different_value"

# ---------------------------------------------------------------------------
# Sub-categories of an omission, and their mirrors under an insertion.
# ---------------------------------------------------------------------------

#: The run leaves the field blank although the expected value is available in the legacy
#: record at the target field or a supported source alias.
MISSED_VALUE = "missed_value"

#: The legacy record does not establish the omitted reference value. A related hint may
#: still exist; external provenance must be verified separately.
EXTERNAL_GAP = "external_gap"

#: The run fills a field the reference leaves blank with a value found in the legacy record.
UNEXPECTED_COPY = "unexpected_copy"

#: The run fills a field the reference leaves blank with a value not found in the legacy record.
UNEXPECTED_FILL = "unexpected_fill"

#: Reporting order, grouped so each sub-category sits under its category.
SUBCATEGORIES = (
    NEAR_MATCH,
    WRONG_MAPPING,
    DIFFERENT_VALUE,
    MISSED_VALUE,
    EXTERNAL_GAP,
    UNEXPECTED_COPY,
    UNEXPECTED_FILL,
)

CATEGORY_BY_SUBCATEGORY = {
    NEAR_MATCH: SUBSTITUTIONS,
    WRONG_MAPPING: SUBSTITUTIONS,
    DIFFERENT_VALUE: SUBSTITUTIONS,
    MISSED_VALUE: OMISSIONS,
    EXTERNAL_GAP: OMISSIONS,
    UNEXPECTED_COPY: INSERTIONS,
    UNEXPECTED_FILL: INSERTIONS,
}

#: The reasons a substitution counts as a near match, in the order
#: :func:`near_match_reasons` returns them.
SAME_VALUE_OTHER_SHAPE = "same value, other shape"
SAME_READ_FORMAT = "same read lengths, different separators"
CONFIRMED_VALUE_MAPPING = "confirmed value mapping"

NEAR_MATCH_REASONS = (
    SAME_VALUE_OTHER_SHAPE,
    SAME_READ_FORMAT,
    CONFIRMED_VALUE_MAPPING,
)

ERROR_COLUMNS = [
    "assay",
    "field",
    "field_type",
    "case",
    "costs",
    "category",
    "subcategory",
    "near_match_reasons",
    "gold_value",
    "predicted_value",
    "legacy_value",
    "legacy_sources",
    "legacy_hint_sources",
    "legacy_support_sources",
    "legacy_target_present",
    "omission_basis",
    "resolution",
    "reasoning",
    "pointer",
]

_WHITESPACE = re.compile(r"\s+")

#: Stripped before comparing, so a bare DOI and the same DOI as a URL are one value.
#: The legacy records store them bare; the templates ask for the resolver URL.
_DOI_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/")

# A complete quantity, rather than a number occurring inside an identifier or free text.
_QUANTITY = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+([a-zµμ%][a-zµμ%0-9/^·.-]*)", re.I)

#: Answers that commit to nothing.  Kept small and literal: a longer list would start
#: swallowing real vocabulary terms.
_PLACEHOLDERS = frozenset(
    {"custom", "in-house", "in house", "unknown", "other", "none", "n/a", "na", "not applicable", "not specified"}
)

# Explicit source relationships for omission review, not arbitrary word overlap. They
# identify potentially useful context, not proof of the reference's exact value. The
# agent's decision log is not used to decide whether a hint exists: a missed source can
# be absent from that log. Assay-specific context is restricted to the relevant fields.
_OMISSION_HINT_ALIASES = {
    "barcode_offset": ("cell_barcode_offset",),
    "barcode_read": ("cell_barcode_read",),
    "barcode_size": ("cell_barcode_size",),
    "preparation_matrix": ("preparation_maldi_matrix",),
    "matrix_deposition_method": ("preparation_type", "preparation_instrument_model"),
    "nuclear_marker_or_stain": ("nuclear_stain", "nuclear_marker", "stain"),
    "is_staining_automated": ("staining_automated", "is_automated_staining"),
}

_SEQUENCING_BARCODE_HINTS = (
    "cell_barcode_offset",
    "cell_barcode_read",
    "cell_barcode_size",
    "sequencing_read_format",
    "rnaseq_assay_method",
    "transposition_method",
    "assay_type",
)
_PREPARATION_HINT_TARGETS = frozenset(
    {
        "library_preparation_kit",
        "sample_indexing_kit",
        "preparation_instrument_kit",
        "preparation_instrument_model",
        "preparation_instrument_vendor",
    }
)
_NO_LEGACY_HINT = frozenset({"unknown", "not specified", "not provided", "none", "n/a", "na", "nan"})

# Source aliases describe the same property as the target. Generic method/kit context
# belongs in legacy_hint_sources, never here. In particular, cell barcodes, read lengths
# and PCR counts do not describe UMI structure. No 1-based/0-based offset conversion is
# assumed without an explicit source convention.
_OMISSION_SOURCE_ALIASES = {
    "tissue": ("organ",),
    "parent_sample_id": ("tissue_id",),
    "barcode_offset": ("cell_barcode_offset",),
    "barcode_read": ("cell_barcode_read",),
    "barcode_size": ("cell_barcode_size",),
    "lc_gradient_value": ("lc_gradient",),
    "mass_to_charge_resolving_power": ("mass_resolving_power", "mz_resolving_power"),
    "library_output_amount_unit": ("library_final_yield_unit",),
    "sequencing_batch_id": ("Seq_run",),
    "preparation_matrix": ("preparation_maldi_matrix",),
    "nuclear_marker_or_stain": ("nuclear_stain", "nuclear_marker"),
    "is_staining_automated": ("staining_automated", "is_automated_staining"),
}
_ASSAY_OMISSION_SOURCE_ALIASES = {
    "atacseq": {"transposition_reagent_kit": ("transposition_kit_number",)},
    "rnaseq": {"number_of_iterations_of_cdna_amplification": ("library_pcr_cycles",)},
    "codex": {"number_of_biomarker_imaging_rounds": ("number_of_cycles",)},
    "celldive": {
        "number_of_biomarker_imaging_rounds": ("number_of_cycles",),
        "number_of_total_imaging_rounds": ("number_of_imaging_rounds",),
    },
    "desi": {"analysis_protocol_doi": ("overall_protocols_io_doi", "protocols_io_doi")},
}


def _flatten(value: Any) -> str:
    """One comparable string for a value of any JSON type."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _loose(value: Any) -> str:
    """Case-folded, whitespace-collapsed form that preserves signs and punctuation."""
    return _WHITESPACE.sub(" ", _flatten(value).strip().casefold())


def _loose_identifier(value: Any) -> str:
    """:func:`_loose`, with any DOI resolver prefix removed.

    Without this, a DOI copied out of the legacy record looks unsupported: the record
    holds ``10.17504/protocols.io.abc`` and the run writes
    ``https://dx.doi.org/10.17504/protocols.io.abc``, which is the same identifier
    wearing the prefix the template asks for.
    """
    return _DOI_PREFIX.sub("", _loose(value))


def _as_number(value: Any) -> Decimal | None:
    """*value* as an exact finite decimal when it reads as one, else ``None``."""
    try:
        number = Decimal(_flatten(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _dataset_path(value: Any, *, directory: bool) -> str | None:
    """Normalize notation for a path within an uploaded dataset, preserving case and parents."""
    if not isinstance(value, str):
        return None
    path = value.strip()
    if directory and path in {".", "./", "/"}:
        return "."
    if path.startswith("./"):
        path = path[2:]
    elif directory and path.startswith("/") and not path.startswith("//"):
        # Legacy directory names use /Proteomics/; the ingestion specification asks
        # for ./Proteomics.  This convention applies only to dataset-directory fields.
        path = path[1:]
    return path.rstrip("/") if directory else path


def _equivalent_to_gold(predicted_value: Any, gold_value: Any, predicted_loose: str, *, field: str) -> bool:
    """Whether representation differences preserve the reference's information.

    Numeric equivalence uses exact decimals, keeping signs and precision.  Punctuation
    is preserved.  Dataset paths have field-specific notation rules, and identifiers
    retain case and leading zeros.
    """
    if field.endswith("_path"):
        directory = field in {"data_path", "metadata_path"}
        predicted_path = _dataset_path(predicted_value, directory=directory)
        return bool(predicted_path) and predicted_path == _dataset_path(gold_value, directory=directory)
    if field.endswith(("_id", "_sequence")):
        return _flatten(predicted_value).strip() == _flatten(gold_value).strip()
    if predicted_loose and predicted_loose == _loose(gold_value):
        return True
    predicted_identifier = _loose_identifier(predicted_value)
    if predicted_identifier and predicted_identifier == _loose_identifier(gold_value):
        return True
    predicted_number = _as_number(predicted_value)
    return predicted_number is not None and predicted_number == _as_number(gold_value)


def near_match_reasons(
    gold_value: Any,
    predicted_value: Any,
    field: str,
    legacy: dict[str, Any],
    permissible: set[str] | None,
    *,
    confirmed_value_mappings: dict[str, dict[tuple[str, str], str]] | None = None,
) -> list[str]:
    """Evidence that a substitution conveys the same information as the reference.

    Automatic checks allow formatting, DOI resolver, numeric, and sequencing-read
    separator differences.  Text containment, copying a legacy value, and permitted-term
    membership alone do not qualify.  ``legacy`` and ``permissible`` remain accepted for
    callers of the previous interface, but are not evidence of equivalence.

    *confirmed_value_mappings* is keyed by field, then by the exact (reference, asserted)
    strings, with a nonempty review justification confirming that each pair conveys the
    same information.  A permitted or more generic standard term does not qualify merely
    because it is standard.  These equivalences are supplied by the reviewer, not inferred.
    """
    reasons: list[str] = []
    predicted_loose = _loose(predicted_value)

    if _equivalent_to_gold(predicted_value, gold_value, predicted_loose, field=field):
        reasons.append(SAME_VALUE_OTHER_SHAPE)
    if field == "sequencing_read_format" and isinstance(gold_value, str) and isinstance(predicted_value, str):
        read_lengths = re.compile(r"\d+(?:\s*[,/+]\s*\d+)+")
        if read_lengths.fullmatch(gold_value.strip()) and read_lengths.fullmatch(predicted_value.strip()):
            gold_reads = tuple(map(int, re.findall(r"\d+", gold_value)))
            predicted_reads = tuple(map(int, re.findall(r"\d+", predicted_value)))
            if gold_reads == predicted_reads:
                reasons.append(SAME_READ_FORMAT)

    if confirmed_value_mappings is not None and isinstance(gold_value, str) and isinstance(predicted_value, str):
        justification = confirmed_value_mappings.get(field, {}).get((gold_value, predicted_value))
        if justification and justification.strip():
            reasons.append(f"{CONFIRMED_VALUE_MAPPING}: {justification}")

    return reasons


def _classify_error(
    case: str,
    gold_value: Any,
    predicted_value: Any,
    field: str,
    legacy: dict[str, Any],
    permissible: set[str] | None,
    *,
    confirmed_value_mappings: dict[str, dict[tuple[str, str], str]] | None = None,
    legacy_support_sources: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """The sub-category for one error, with the near-match reasons behind it.

    Insertions split on whether the legacy record holds the asserted value. Omissions
    require a matching value at the target field or a supported source alias; contextual
    hints and coincidental matches in unrelated fields cannot establish a reference value.
    Substitutions check representation equivalence or confirmed mappings first. Otherwise,
    an asserted value found anywhere in the legacy record is a candidate wrong mapping;
    a different value must be absent from the legacy record.
    """
    if case == INSERTION:
        return (UNEXPECTED_COPY if _appears_in_legacy(predicted_value, legacy) else UNEXPECTED_FILL), []
    if case == DELETION:
        supported = bool(
            legacy_support_sources
            if legacy_support_sources is not None
            else _omission_legacy_support(gold_value, field, legacy, assay_key="")
        )
        return (MISSED_VALUE if supported else EXTERNAL_GAP), []

    reasons = near_match_reasons(
        gold_value, predicted_value, field, legacy, permissible, confirmed_value_mappings=confirmed_value_mappings
    )
    if reasons:
        return NEAR_MATCH, reasons
    return (WRONG_MAPPING if _appears_in_legacy(predicted_value, legacy) else DIFFERENT_VALUE), []


def _iter_legacy_values(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    """Yield a legacy value and its descendants with their field paths."""
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_legacy_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_legacy_values(child, f"{path}[{index}]")


def _matches_legacy_value(value: Any, legacy_value: Any) -> bool:
    """Match normalized whole values, or a numeric value recorded with an explicit unit."""
    if _is_missing(legacy_value):
        return False
    target = _loose_identifier(value)
    if target and _loose_identifier(legacy_value) == target:
        return True
    number = _as_number(value)
    if number is None:
        return False
    recorded_number = _as_number(legacy_value)
    if recorded_number is None and isinstance(legacy_value, str):
        quantity = _QUANTITY.fullmatch(legacy_value.strip())
        if quantity:
            recorded_number = Decimal(quantity.group(1))
    return recorded_number is not None and recorded_number == number


def _legacy_fields_carrying(value: Any, legacy: dict[str, Any]) -> set[str]:
    """Which legacy field paths hold *value*, comparing loosely and ignoring a DOI prefix.

    The one definition of "the record contains this", so "the run copied the same field"
    and "the run copied a different field" cannot disagree about what counts as carried.
    A blank value is carried by nothing.  Signs and punctuation are retained, so
    ``-5`` is not reported as present merely because the record contains ``5``.

    Nested objects and list items are searched as individual values, rather than only
    comparing their containing object as a whole.  For example, ``other_metadata.Seq_run``
    can account for an asserted sequencing batch identifier.  Matching still compares
    whole values and numbers recorded with explicit units, such as ``225 pM`` for ``225``;
    a substring inside an identifier or a dictionary key alone does not establish provenance.
    """
    target = _loose_identifier(value)
    if not target:
        return set()
    return {
        path
        for legacy_field, value_at_field in legacy.items()
        for path, legacy_value in _iter_legacy_values(value_at_field, legacy_field)
        if _matches_legacy_value(value, legacy_value)
    }


def _appears_in_legacy(value: Any, legacy: dict[str, Any]) -> bool:
    """Whether any legacy field or nested value holds *value*."""
    return bool(_legacy_fields_carrying(value, legacy))


def _omission_legacy_hints(field: str, legacy: dict[str, Any], *, assay_key: str) -> dict[str, Any]:
    """Find explicit source fields that provide context for an omitted target field.

    A nonblank value at the target field or a known source alias is a hint even when
    its wording differs from the reference. Related assay-method and platform fields
    supply context for preparation and barcode fields. These hints can be incomplete
    or inconsistent with the reference; they do not certify a particular inference.
    Generic identifiers, operator details, and protocol links alone are not hints.
    """
    source_fields = {field, *_OMISSION_HINT_ALIASES.get(field, ())}
    if assay_key in {"atacseq", "rnaseq"}:
        if field.startswith(("barcode_", "umi_")):
            source_fields.update(_SEQUENCING_BARCODE_HINTS)
        if field in _PREPARATION_HINT_TARGETS:
            if assay_key == "rnaseq":
                source_fields.update({"rnaseq_assay_method", "assay_type"})
            else:
                source_fields.update(
                    {"transposition_method", "transposition_kit_number", "transposition_transposase_source"}
                )
        if field == "transposition_reagent_kit":
            source_fields.update(
                {"transposition_kit_number", "transposition_transposase_source", "library_preparation_kit"}
            )
        if field == "sample_indexing_kit":
            kit = legacy.get("library_preparation_kit")
            if isinstance(kit, str) and re.search(r"\bindex(?:es|ing)?\b", kit, re.I):
                source_fields.add("library_preparation_kit")
        if field == "assay_input_entity":
            source_fields.update(
                {"sc_isolation_entity", "bulk_transposition_input_number_nuclei", "rnaseq_assay_method"}
            )
    if assay_key == "af" and field == "analyte_class":
        source_fields.add("assay_type")
    if assay_key == "maldi" and field == "ion_mobility":
        model = legacy.get("acquisition_instrument_model")
        if isinstance(model, str) and re.search(r"\btims", model, re.I):
            source_fields.add("acquisition_instrument_model")
    if assay_key == "desi" and field in {
        "matrix_deposition_method",
        "preparation_matrix",
        "preparation_instrument_model",
        "preparation_instrument_vendor",
    }:
        source_fields.update({"assay_type", "ms_source"})

    return {
        path: value
        for name, top_value in legacy.items()
        for path, value in _iter_legacy_values(top_value, name)
        if (path in source_fields or path.rsplit(".", 1)[-1] in source_fields)
        and not _is_missing(value)
        and not isinstance(value, (dict, list))
        and _loose(value) not in _NO_LEGACY_HINT
    }


def _omission_legacy_support(gold_value: Any, field: str, legacy: dict[str, Any], *, assay_key: str) -> dict[str, Any]:
    """Values that establish the reference at a source describing the target property.

    Search nested source fields too. Read labels permit the deterministic conversion
    R1 -> Read 1 (R1), but only within the same property: a barcode read is not a UMI
    read. Numeric coincidences at unrelated fields are retained in legacy_sources for
    audit, but cannot support a missed-value assignment.
    """
    source_fields = {
        field,
        *_OMISSION_SOURCE_ALIASES.get(field, ()),
        *_ASSAY_OMISSION_SOURCE_ALIASES.get(assay_key, {}).get(field, ()),
    }
    sources = {}
    for name, top_value in legacy.items():
        for path, value in _iter_legacy_values(top_value, name):
            if re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1]) not in source_fields or _is_missing(value):
                continue
            matches = _equivalent_to_gold(value, gold_value, _loose(value), field=field)
            if not field.endswith(("_id", "_sequence")) and _as_number(gold_value) is not None:
                matches = matches or _matches_legacy_value(gold_value, value)
            if not matches and field in {"barcode_read", "umi_read"}:
                pattern = r"(?:r([12])|read\s*([12])(?:\s*\(r[12]\))?)"
                source_read = re.fullmatch(pattern, _loose(value))
                reference_read = re.fullmatch(pattern, _loose(gold_value))
                if source_read and reference_read:
                    # Require the parenthesized label, if present, to agree as well.
                    def read_number(match: re.Match[str]) -> str:
                        return next(group for group in match.groups() if group is not None)

                    number = read_number(source_read)
                    matches = number == read_number(reference_read) and all(
                        not re.search(r"\(r[12]\)", text) or f"(r{number})" in text
                        for text in (_loose(value), _loose(gold_value))
                    )
            if matches:
                sources[path] = value
    return sources


def _decision_log(
    assay: Assay, model: str, condition: str, record_name: str, *, run: int = 1
) -> dict[str, dict[str, Any]]:
    """The run's own account of each field, keyed by field, when it kept one.

    ARMS writes a decision per field under ``decisions/``; the prompt-only conditions do
    not, so this is empty for them and the two columns it fills stay blank.
    """
    path = assay.output_dir(model, condition, run=run) / "decisions" / record_name
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return {entry["key"]: entry for entry in entries if isinstance(entry, dict) and "key" in entry}


def _require_legacy(legacy_path: Path, predicted_path: Path) -> dict[str, Any]:
    """The legacy record at *legacy_path*, or a refusal to categorise without it.

    Provenance-based labels need the legacy record.  A missing input would silently
    move errors into :data:`UNEXPECTED_FILL`, :data:`DIFFERENT_VALUE`, or
    :data:`EXTERNAL_GAP`.  Reconciliation would still pass because the error count is
    unchanged, even though its attribution is wrong.

    A record with a prediction had an input when the run read it, which is what makes
    this a broken corpus rather than a partial one: an assay missing its gold or its
    schema is skipped, but a prediction without its input cannot be scored honestly.
    """
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"no legacy record at {legacy_path}, but {predicted_path} exists to categorise. "
            "Provenance is decided against the legacy record, so continuing would report "
            "these errors as invented, unsupported or unavailable when they may be none of those."
        )
    return load_record(legacy_path)


def collect_field_errors(
    data_root: str | Path,
    model: str,
    condition: str,
    *,
    match_case: bool = True,
    match_whole_word: bool = True,
    run: int = 1,
    confirmed_value_mappings: dict[str, dict[tuple[str, str], str]] | None = None,
) -> pd.DataFrame:
    """One row per counted error, labelled at both levels of the taxonomy.

    One row per (record, field), not per side of the score: the category *is* the confusion
    case, so a substitution has one label rather than one per side.  ``costs`` says which
    side it lands on, and :func:`reconcile_with_confusion` checks that ``FP`` and ``FN``
    still come back out.  ``pointer`` is ``<assay>/<record>#<field>``, and ``resolution``
    and ``reasoning`` carry the run's own account of the field where it kept one.

    Returns an empty frame with :data:`ERROR_COLUMNS` when the run has no predictions.
    Raises :class:`FileNotFoundError` when a record has a prediction but no legacy input,
    which :func:`_require_legacy` explains.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for assay in iter_assays(data_root):
        if not assay.has_gold:
            continue
        ontology_fields = assay.ontology_fields()
        permissible = {
            field: {_loose_identifier(option) for option in options}
            for field, options in assay.permissible_values().items()
        }
        output_dir = assay.output_dir(model, condition, run=run)

        for gold_path in sorted(assay.gold_dir.glob("*.json")):
            predicted_path = output_dir / gold_path.name
            if not predicted_path.exists():
                continue
            gold = load_record(gold_path)
            predicted = load_record(predicted_path)
            legacy = _require_legacy(assay.input_dir / gold_path.name, predicted_path)
            legacy_paths = {
                path: value for name, value in legacy.items() for path, value in _iter_legacy_values(value, name)
            }
            decisions = _decision_log(assay, model, condition, gold_path.name, run=run)

            for field in gold:
                case = _classify_field(predicted, gold, field, match_case=match_case, match_whole_word=match_whole_word)
                if case not in (INSERTION, DELETION, SUBSTITUTION):
                    continue

                hints = _omission_legacy_hints(field, legacy, assay_key=assay.key) if case == DELETION else {}
                support = (
                    _omission_legacy_support(gold.get(field), field, legacy, assay_key=assay.key)
                    if case == DELETION
                    else {}
                )
                subcategory, reasons = _classify_error(
                    case,
                    gold.get(field),
                    predicted.get(field),
                    field,
                    legacy,
                    permissible.get(field),
                    confirmed_value_mappings=confirmed_value_mappings,
                    legacy_support_sources=support,
                )
                source_value = gold.get(field) if case == DELETION else predicted.get(field)
                sources = _legacy_fields_carrying(source_value, legacy)
                decision = decisions.get(field, {})
                omission_basis = ""
                if case == DELETION:
                    omission_basis = (
                        "reference established by legacy source"
                        if support
                        else "legacy hint insufficient to establish reference"
                        if hints
                        else "no supporting legacy value detected"
                    )
                rows.append(
                    {
                        "assay": assay.label,
                        "field": field,
                        "field_type": "ontology" if field in ontology_fields else "non_ontology",
                        "case": case,
                        "costs": COSTS_BY_CASE[case],
                        "category": CATEGORY_BY_CASE[case],
                        "subcategory": subcategory,
                        "near_match_reasons": ", ".join(reasons),
                        "gold_value": gold.get(field),
                        "predicted_value": predicted.get(field),
                        "legacy_value": legacy.get(field),
                        "legacy_sources": {path: legacy_paths[path] for path in sorted(sources)},
                        "legacy_hint_sources": hints,
                        "legacy_support_sources": support,
                        "legacy_target_present": any(path.rsplit(".", 1)[-1] == field for path in legacy_paths),
                        "omission_basis": omission_basis,
                        "resolution": decision.get("resolution"),
                        "reasoning": decision.get("reasoning"),
                        "pointer": f"{assay.key}/{gold_path.stem}#{field}",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=ERROR_COLUMNS)
    frame = pd.DataFrame(rows)[ERROR_COLUMNS]
    _check_levels_agree(frame)
    return frame


def _check_levels_agree(errors: pd.DataFrame) -> None:
    """Refuse a frame whose two levels contradict each other.

    The category comes from the confusion case and the sub-category from the classifier, by
    two separate routes, so they could drift apart without anything raising -- and a
    sub-category filed under the wrong category would make one table's rows sum to
    something no other table agrees with.
    """
    wrong = {
        (subcategory, category)
        for subcategory, category in zip(errors["subcategory"], errors["category"], strict=True)
        if CATEGORY_BY_SUBCATEGORY.get(subcategory) != category
    }
    if wrong:
        raise ValueError(f"sub-category filed under the wrong category: {sorted(wrong)}")


def summarize_error_subcategories_by_field_type(errors: pd.DataFrame) -> pd.DataFrame:
    """Count each error once within its subcategory and template-defined field type.

    Pass a deduplicated frame to count distinct disagreements, or the complete collection
    to count record-field instances. Zero-count subcategories and field types are retained.
    """
    import pandas as pd

    table = (
        pd.crosstab(errors["subcategory"], errors["field_type"])
        .reindex(index=SUBCATEGORIES, columns=["ontology", "non_ontology"], fill_value=0)
        .fillna(0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    table.columns.name = None
    return table.reset_index()


def deduplicate_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """One row per *distinct* error, rather than one per record the error occurs in.

    The corpus repeats itself: the same field of the same assay is wrong the same way in
    dozens of records, so an instance-weighted count is carried by a handful of recurring
    values.  This answers the other question -- how many different things the run gets
    wrong -- the way :mod:`analysis.data_analysis.repetition` does for the headline
    numbers, and against the same notion of a distinct value.

    Two errors are the same when they agree on assay, field, what gold held, what the run
    asserted, and what the taxonomy calls it.  The sub-category is in the key because it is
    not implied by the rest: the same gold and asserted values can be labelled differently
    in two records when the *legacy* records differ -- a value absent from one record and
    present in another is a different mistake, and the taxonomy already says so.  It costs
    two rows on this corpus, and keeps every table below a partition of these rows.

    ``n_instances`` counts the records each distinct error covered, so nothing about the
    repetition is lost by collapsing it.
    """
    import pandas as pd

    if errors.empty:
        return errors.assign(n_instances=pd.Series(dtype=int))

    keys = ["assay", "field", "subcategory"]
    # Values of any JSON type reach this frame, and an unhashable one -- a list, say --
    # would make groupby raise rather than group.  Serialised, they compare as they read.
    serialised = errors.assign(
        _gold=errors["gold_value"].map(lambda value: json.dumps(value, sort_keys=True, default=str)),
        _predicted=errors["predicted_value"].map(lambda value: json.dumps(value, sort_keys=True, default=str)),
    )
    grouped = serialised.groupby([*keys, "_gold", "_predicted"], dropna=False, sort=False)
    distinct = grouped.head(1).copy()
    distinct["n_instances"] = (
        grouped.size().reindex(pd.MultiIndex.from_frame(distinct[[*keys, "_gold", "_predicted"]])).to_numpy()
    )
    return distinct.drop(columns=["_gold", "_predicted"]).reset_index(drop=True)


def _summarize_level(
    errors: pd.DataFrame,
    *,
    level: str,
    order: tuple[str, ...],
    field_type: str | None,
) -> pd.DataFrame:
    """Counts of *level* per assay, in *order*, over every counted error."""
    import pandas as pd

    selected = errors if field_type is None else errors[errors["field_type"] == field_type]
    if selected.empty:
        return pd.DataFrame(columns=[level, "total", "share"])

    table = (
        pd.crosstab(selected[level], selected["assay"])
        .reindex([name for name in order if name in set(selected[level])])
        .fillna(0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    table["share"] = (table["total"] / len(selected)).round(3)
    return table.reset_index()


def summarize_error_categories(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """Frequency of each **category** per assay, over every counted error.

    One row per category, one column per assay, plus a ``total`` and a ``share``.  The
    coarse level: what the figures draw and what a headline can carry.  *field_type*
    narrows to ``"ontology"`` or ``"non_ontology"``.
    """
    return _summarize_level(errors, level="category", order=CATEGORIES, field_type=field_type)


def summarize_error_subcategories(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """Frequency of each **sub-category** per assay, over every counted error.

    Same shape as :func:`summarize_error_categories` one level down, in an order that runs
    through the categories in theirs -- so printed underneath it, each block of rows sits
    below the category it rolls into.  A sub-category that dominates one assay and is absent
    elsewhere is visible here rather than averaged away.
    """
    return _summarize_level(errors, level="subcategory", order=SUBCATEGORIES, field_type=field_type)


def _level_shares(errors: pd.DataFrame, *, level: str, order: tuple[str, ...], field_type: str | None) -> pd.DataFrame:
    """Counts and within-assay shares of *level*, per assay and pooled over the corpus."""
    import pandas as pd

    selected = errors if field_type is None else errors[errors["field_type"] == field_type]
    if selected.empty:
        return pd.DataFrame(columns=["assay", level, "n", "n_errors", "share"])

    rows: list[dict[str, Any]] = []
    for assay_label, frame in [*selected.groupby("assay", observed=True), (POOLED_ASSAY, selected)]:
        counts = frame[level].value_counts()
        for name in order:
            rows.append(
                {
                    "assay": assay_label,
                    level: name,
                    "n": int(counts.get(name, 0)),
                    "n_errors": len(frame),
                    "share": int(counts.get(name, 0)) / len(frame),
                }
            )
    return pd.DataFrame(rows)


def category_shares(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """The categories per assay, as counts and as shares of that assay's errors.

    One row per (assay, category) plus one per category for the pooled corpus under
    :data:`POOLED_ASSAY`, so a figure can draw both from one frame.  ``share`` is within the
    assay, which is what makes composition comparable across assays that differ in size by
    more than tenfold; ``n_errors`` is carried alongside so the reader can see how much a
    share is standing on.  Every category is present even at zero, so a figure's segments
    keep their order between rows.
    """
    return _level_shares(errors, level="category", order=CATEGORIES, field_type=field_type)


def subcategory_shares(errors: pd.DataFrame, *, field_type: str | None = None) -> pd.DataFrame:
    """The same, one level down.

    The sub-categories run in their categories' order, so a bar stacked from this frame has
    each category as a single unbroken span -- which is what lets a figure brace the
    categories over the top of the sub-category segments.
    """
    return _level_shares(errors, level="subcategory", order=SUBCATEGORIES, field_type=field_type)


def reconcile_with_confusion(
    errors: pd.DataFrame,
    data_root: str | Path,
    model: str,
    condition: str,
    *,
    run: int = 1,
) -> dict[str, dict[str, int]]:
    """Check the labelled rows against the confusion counts they claim to explain.

    With one row per error rather than one per side, ``FP`` and ``FN`` are recovered from
    the confusion case: an insertion costs precision, a deletion recall, a substitution
    both.  Returns ``{"FP": {"counted": ..., "categorised": ...}, "FN": {...}}``.  A
    mismatch means an error was missed or counted twice, which would make every share in
    the tables above wrong -- so this is worth asserting rather than assuming.
    """
    from analysis.data_analysis.precision_recall_tables import _accumulate_confusion

    counts, _n_pairs, _skipped = _accumulate_confusion(iter_assays(data_root), model, condition, run=run)
    return {
        "FP": {
            "counted": counts["all"]["FP"],
            "categorised": int(errors["case"].isin((INSERTION, SUBSTITUTION)).sum()),
        },
        "FN": {
            "counted": counts["all"]["FN"],
            "categorised": int(errors["case"].isin((DELETION, SUBSTITUTION)).sum()),
        },
    }
