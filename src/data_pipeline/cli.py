"""Command line entry point: ``data-pipeline [CONFIG] [options]``."""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import threading
from collections.abc import Iterator

from . import config
from .orchestrator import Orchestrator
from .report import Reporter, display_path


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
    ap.add_argument("-F", "--follow", action="store_true",
                    help="print each step's output live, as it is written")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return ap


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


@contextlib.contextmanager
def _stop_signals_interrupt() -> Iterator[None]:
    """Treat SIGTERM and SIGHUP like Ctrl-C.

    Workers run in their own session, so a closed terminal or a cancelled CI job reaches
    only this process. It has to stop the step itself.
    """
    if threading.current_thread() is not threading.main_thread():
        yield  # signal handlers can only be set from the main thread
        return
    stops = [getattr(signal, n) for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]
    previous = {s: signal.signal(s, _interrupt) for s in stops}
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


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

    reporter = Reporter(verbose=args.verbose, color=False if args.no_color else None,
                        follow=args.follow)
    orchestrator = Orchestrator(pipeline, reporter)
    try:
        with _stop_signals_interrupt():
            result = orchestrator.run(
                only=args.only, force=args.force, rerun=args.rerun,
                dry_run=args.dry_run)
    except KeyboardInterrupt:
        log = orchestrator.running_log
        print("\n■ interrupted" + (f" · log: {display_path(log)}" if log else ""))
        return 130
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
