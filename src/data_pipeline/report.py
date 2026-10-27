"""Console status output: one line per step, full output dump on failure."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .config import Pipeline, Step
    from .orchestrator import Outcome, RunResult

_STYLE = {
    "ran": ("✓", "ran", "32"),
    "cached": ("↷", "cached", "2"),
    "would run": ("•", "will run", "36"),
    "warned": ("⚠", "warning", "33"),
    "skipped": ("–", "skipped", "33"),
    "failed": ("✗", "FAILED", "31;1"),
}


def display_path(path: Path) -> str:
    """``path`` relative to the cwd when it is below it, else absolute."""
    try:
        rel = os.path.relpath(path)
    except ValueError:  # different drive on Windows
        return str(path)
    return str(path) if rel.startswith("..") else rel


class Reporter:
    def __init__(self, stream: TextIO | None = None, verbose: bool = False,
                 color: bool | None = None, follow: bool = False):
        self.out = stream or sys.stdout
        self.verbose = verbose
        self.follow = follow  # stream step output live instead of printing it at the end
        self._current = ""  # prefix of the running step, printed before its first line
        self._streamed = False
        tty = hasattr(self.out, "isatty") and self.out.isatty()
        self.live = tty
        self.color = (tty and "NO_COLOR" not in os.environ) if color is None else color
        self.width = 0

    def _c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _print(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def _prefix(self, idx: int, total: int, step: Step) -> str:
        n = len(str(total))
        return f"[{idx:>{n}}/{total}] {step.name:<{self.width}}"

    # -- run lifecycle -----------------------------------------------------------------

    def header(self, pipeline: Pipeline, count: int, dry_run: bool) -> None:
        self.width = max(len(s.name) for s in pipeline.steps)
        where = display_path(pipeline.path)
        mode = " · dry run" if dry_run else ""
        self._print(self._c(f"▶ {pipeline.name}", "1") + f" · {count} steps · {where}{mode}")

    def preflight_failed(self, issues: dict[str, list[str]]) -> None:
        self._print(self._c("✗ preflight failed — nothing was run", "31;1"))
        for name, errs in issues.items():
            for e in errs:
                self._print(f"  {name}: {e}")

    def begin(self, idx: int, total: int, step: Step) -> None:
        self._current = self._prefix(idx, total, step)
        self._streamed = False
        if self.live:
            print(self._current + self._c("  … running", "2"),
                  end="", file=self.out, flush=True)

    def line(self, text: str) -> None:
        """One line of a running step's output (follow mode; called from a reader thread)."""
        if not self._streamed:
            self._streamed = True
            if self.live:
                self.out.write("\r\033[K")
            self._print(self._current.rstrip())
        self._print(self._c("    │ ", "2") + text)

    def end(self, idx: int, total: int, step: Step, o: Outcome) -> None:
        sym, label, code = _STYLE[o.status]
        line = f"{self._prefix(idx, total, step)}  {self._c(f'{sym} {label:<8}', code)}"
        line += f" {o.duration:6.2f}s" if o.duration else " " * 8
        if o.detail:
            line += f"  {o.detail}"
        if o.reason:
            line += self._c(f"  ({o.reason})", "2")
        if o.status in ("warned", "skipped"):
            line += self._c("  [non-critical, continuing]", "33")
        if self.live:
            self.out.write("\r\033[K")
        self._print(line)

        if o.status in ("warned", "failed") and o.output:
            self._dump(step, o, "33" if o.status == "warned" else "31")
        elif self.verbose and o.output and not self._streamed:
            for ln in o.output.rstrip().splitlines():
                self._print(self._c("    │ ", "2") + ln)

    def _dump(self, step: Step, o: Outcome, code: str) -> None:
        bar = "─" * 60
        self._print(self._c(f"┌─ {step.name}: {o.detail} ─ {step.function}", code))
        self._print(self._c("(output above)", "2") if self._streamed else o.output.rstrip())
        if o.log:
            self._print(f"log: {display_path(o.log)}")
        self._print(self._c(f"└{bar}", code))

    def footer(self, result: RunResult, dry_run: bool) -> None:
        counts: dict[str, int] = {}
        for o in result.outcomes.values():
            counts[o.status] = counts.get(o.status, 0) + 1
        names = {"ran": "ran", "cached": "cached", "would run": "to run",
                 "warned": "warning(s)", "skipped": "skipped", "failed": "failed"}
        parts = [f"{counts[k]} {v}" for k, v in names.items() if counts.get(k)]
        summary = " · ".join(parts + [f"{result.elapsed:.2f}s"])
        if result.halted_by:
            self._print(self._c(f"■ halted at '{result.halted_by}'", "31;1") + f" · {summary}")
        elif counts.get("warned") or counts.get("skipped"):
            self._print(self._c("✓ completed with warnings", "33;1") + f" · {summary}")
        else:
            self._print(self._c("✓ dry run complete" if dry_run else "✓ completed", "32;1")
                        + f" · {summary}")
