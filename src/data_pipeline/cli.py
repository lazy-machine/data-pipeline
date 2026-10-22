"""Command line entry point: ``data-pipeline [CONFIG] [options]``."""

from __future__ import annotations

import argparse
import sys

from . import config
from .orchestrator import Orchestrator
from .report import Reporter


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="data-pipeline",
        description="Run a TOML-defined pipeline, one isolated step at a time.",
    )
    ap.add_argument("config", nargs="?", default="pipeline.toml",
                    help="pipeline definition (default: ./pipeline.toml)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="verify and show what would run, without running anything")
    ap.add_argument("-f", "--force", action="store_true",
                    help="ignore all checkpoints and re-run every step")
    ap.add_argument("-r", "--rerun", action="append", default=[], metavar="STEP",
                    help="ignore the checkpoint of STEP (repeatable)")
    ap.add_argument("-o", "--only", nargs="+", metavar="STEP",
                    help="run only these steps (their inputs must already exist)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also show the captured output of successful steps")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = config.load(args.config)
    except config.ConfigError as exc:
        print(f"✗ config error: {exc}")
        return 2

    names = {s.name for s in pipeline.steps}
    unknown = sorted((set(args.only or []) | set(args.rerun)) - names)
    if unknown:
        print(f"✗ unknown step(s): {', '.join(unknown)} (known: {', '.join(sorted(names))})")
        return 2

    reporter = Reporter(verbose=args.verbose, color=False if args.no_color else None)
    try:
        result = Orchestrator(pipeline, reporter).run(
            only=args.only, force=args.force, rerun=args.rerun,
            dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n■ interrupted")
        return 130
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
