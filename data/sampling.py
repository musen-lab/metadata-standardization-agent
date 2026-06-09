"""Randomly sample metadata records from multiple source directories into a single output directory."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample metadata records from multiple sources into one output directory.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="Source directory names (relative to data/), followed by counts string and output name.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args(argv)

    # The last positional arg is the output directory name,
    # the second-to-last is the quoted counts string,
    # and everything before that is a source directory name.
    positionals = args.sources
    if len(positionals) < 3:
        parser.error("Need at least: <source> <counts> <output>")

    args.output = positionals[-1]
    counts_str = positionals[-2]
    args.sources = positionals[:-2]

    try:
        args.counts = [int(c) for c in counts_str.split()]
    except ValueError:
        parser.error(f"Counts must be space-separated integers, got: {counts_str!r}")

    if len(args.sources) != len(args.counts):
        parser.error(
            f"Number of sources ({len(args.sources)}) does not match "
            f"number of counts ({len(args.counts)})."
        )

    return args


def validate_sources(sources: list[str], counts: list[int]) -> dict[str, list[str]]:
    """Validate source directories and return available JSON filenames per source."""
    available: dict[str, list[str]] = {}
    for source, count in zip(sources, counts):
        input_dir = DATA_DIR / source / "input"
        gold_dir = DATA_DIR / source / "gold"

        if not input_dir.is_dir():
            sys.exit(f"Error: input directory not found: {input_dir}")
        if not gold_dir.is_dir():
            sys.exit(f"Error: gold directory not found: {gold_dir}")

        input_files = sorted(f.name for f in input_dir.iterdir() if f.suffix == ".json")
        gold_files = set(f.name for f in gold_dir.iterdir() if f.suffix == ".json")

        # Only consider files that have both input and gold versions
        matched = [f for f in input_files if f in gold_files]
        if not matched:
            sys.exit(f"Error: no matching input/gold JSON files found in {source}/")

        unmatched = set(input_files) - gold_files
        if unmatched:
            print(f"Warning: {len(unmatched)} input file(s) in {source}/ have no gold match, skipping them.")

        if count > len(matched):
            sys.exit(
                f"Error: requested {count} samples from {source}/ "
                f"but only {len(matched)} matched files available."
            )

        available[source] = matched

    return available


def handle_existing_output(output_dir: Path) -> None:
    """Prompt user before overwriting an existing output directory."""
    if not output_dir.exists():
        return

    response = input(f"Output directory {output_dir} already exists. Overwrite? [y/N] ").strip().lower()
    if response != "y":
        sys.exit("Aborted.")

    for subdir in ("input", "gold"):
        target = output_dir / subdir
        if target.is_dir():
            shutil.rmtree(target)


def sample_and_copy(
    sources: list[str],
    counts: list[int],
    available: dict[str, list[str]],
    output_dir: Path,
    seed: int | None,
) -> None:
    """Sample files from each source and copy to the output directory."""
    if seed is not None:
        random.seed(seed)

    output_input = output_dir / "input"
    output_gold = output_dir / "gold"
    output_input.mkdir(parents=True, exist_ok=True)
    output_gold.mkdir(parents=True, exist_ok=True)

    total = 0
    for source, count in zip(sources, counts):
        sampled = random.sample(available[source], count)
        for filename in sampled:
            shutil.copy2(DATA_DIR / source / "input" / filename, output_input / filename)
            shutil.copy2(DATA_DIR / source / "gold" / filename, output_gold / filename)
        total += count
        print(f"  {source}: sampled {count} files")

    print(f"  Total: {total} files in {output_dir}/")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the sampling script."""
    args = parse_args(argv)
    available = validate_sources(args.sources, args.counts)
    output_dir = DATA_DIR / args.output
    handle_existing_output(output_dir)
    sample_and_copy(args.sources, args.counts, available, output_dir, args.seed)


if __name__ == "__main__":
    main()
