"""Explicitly scoped RNA-seq experiments using the compatible Langfuse v4 SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiment.dataset import DEFAULT_DATA_DIR, TEMPLATE_IRI, load_items
from experiment.evaluators import ITEM_SCORE_NAMES, RUN_SCORE_NAMES, make_evaluators

if TYPE_CHECKING:
    from langfuse.experiment import ExperimentResult

SDK_VERSION = "4.18.0"


def check_environment(model: str) -> None:
    """Check configuration presence without printing secrets or making requests."""
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "CEDAR_API_KEY", "BIOPORTAL_API_KEY"]
    if not model.startswith("ollama:"):
        required.append("OPENAI_API_KEY")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    host = os.environ.get("LANGFUSE_HOST", "").strip() or os.environ.get("LANGFUSE_BASE_URL", "").strip()
    if not host:
        missing.append("LANGFUSE_HOST (or LANGFUSE_BASE_URL)")
    if missing:
        raise ValueError(f"Set environment variables: {', '.join(missing)}; never paste credentials into chat")
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        raise ValueError("LANGFUSE_TRACING_ENABLED must not be false for experiments")
    os.environ["LANGFUSE_HOST"] = host
    if version("langfuse") != SDK_VERSION:
        raise ValueError(f"Use experiment/requirements.txt: this runner requires langfuse=={SDK_VERSION}")


def run_experiment(
    *,
    model: str,
    limit: int | None,
    max_concurrency: int = 1,
    data_dir: Path = DEFAULT_DATA_DIR,
    name: str = "ARMS RNA-seq",
    run_name: str | None = None,
) -> ExperimentResult:
    """Run explicitly selected local items, never creating or uploading a hosted dataset."""
    if not model.strip() or max_concurrency < 1:
        raise ValueError("model must be nonempty and max_concurrency must be positive")
    items = load_items(data_dir, limit=limit)
    check_environment(model)
    # Imports and agent construction stay after validation and configuration checks.
    from langfuse import get_client

    from experiment.adapter import make_task

    client = get_client()
    try:
        if not client.auth_check():
            raise ValueError("Langfuse authentication failed; check the configured project and host")
        schema_path = data_dir / "schemas" / "rnaseq.json"
        item_evaluator, run_evaluator = make_evaluators(schema_path, expected_records=len(items))
        return client.run_experiment(
            name=name,
            run_name=run_name,
            description="Local RNA-seq pairs; unchanged ARMS workflow; existing strict deterministic metrics",
            data=items,
            task=make_task(model),
            evaluators=[item_evaluator],
            run_evaluators=[run_evaluator],
            max_concurrency=max_concurrency,
            metadata={
                "assay": "rnaseq",
                "model": model,
                "template_iri": TEMPLATE_IRI,
                "langfuse_sdk": SDK_VERSION,
                "selected_records": str(len(items)),
                "selection": "all" if limit is None else f"first-{limit}-sorted-filenames",
                "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                "matching": "existing strict: match_case=True, match_whole_word=True",
            },
        )
    finally:
        client.flush()


def summarize(result: ExperimentResult) -> dict[str, Any]:
    """Return only scores and coverage (no records, gold, credentials or error payloads)."""
    scores = {evaluation.name: evaluation.value for evaluation in result.run_evaluations}
    complete = (
        set(scores) == RUN_SCORE_NAMES
        and scores.get("failed_records") == 0
        and scores.get("completed_records") == len(result.item_results)
        and all(
            {evaluation.name for evaluation in item.evaluations} == ITEM_SCORE_NAMES for item in result.item_results
        )
    )
    return {"name": result.name, "run_name": result.run_name, "complete": complete, "scores": scores}


def main(argv: list[str] | None = None) -> int:
    """Validate offline, or run with explicit model and corpus selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--validate-only", action="store_true", help="No credentials, SDK client or network calls")
    parser.add_argument("--env-file", type=Path, help="Explicit dotenv file; existing shell variables win")
    parser.add_argument("--model", help="Explicit ARMS model identifier; no paid-model default")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Positive number of records, sorted by filename (smoke: 5)")
    selection.add_argument("--all", action="store_true", help="Full corpus; use only after reviewing smoke and cost")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--name", default="ARMS RNA-seq")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    if (args.limit is not None and args.limit < 1) or args.max_concurrency < 1:
        parser.error("--limit and --max-concurrency must be positive")
    if not args.validate_only and (not args.model or not (args.limit or args.all)):
        parser.error("execution requires --model and either --limit (start with 5) or --all")
    if args.validate_only:
        items = load_items(args.data_dir, limit=args.limit)
        print(f"Validated all RNA-seq pairs; selected {len(items)} local items. No network calls or dataset upload.")
        return 0
    if args.env_file:
        from dotenv import load_dotenv

        if not args.env_file.is_file():
            parser.error("--env-file must be an existing file")
        load_dotenv(args.env_file, override=False)
    result = run_experiment(
        model=args.model,
        limit=args.limit,
        max_concurrency=args.max_concurrency,
        data_dir=args.data_dir,
        name=args.name,
        run_name=args.run_name,
    )
    summary = summarize(result)
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
