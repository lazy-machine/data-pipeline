"""Worker process: runs exactly one step function, then exits.

Invoked by the orchestrator as ``python -m data_pipeline.worker SPEC RESULT``. Each step
gets a fresh interpreter, so no memory, globals or imported modules leak between steps.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .context import StepContext


def _summarize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        return " ".join(parts)
    return str(value)


def _stop_with_runner(runner: int) -> None:
    """End the step, and what it started, if the runner dies.

    A worker leads its own session, so a signal that ends the runner never reaches it, and
    nothing else would stop it writing outputs that no checkpoint will record. This relies on
    the worker being the runner's direct child: a launcher that forks would look like a
    runner that died.
    """
    if not hasattr(os, "killpg"):
        return

    def watch() -> None:
        while os.getppid() == runner:
            time.sleep(0.5)
        os.killpg(os.getpgid(0), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()


def main(argv: list[str]) -> int:
    spec_path, result_path = argv
    spec = json.loads(Path(spec_path).read_text())
    _stop_with_runner(spec["runner"])  # from the runner: it may die before this line runs

    for p in reversed(spec["python_path"]):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        module = importlib.import_module(spec["module"])
        func = getattr(module, spec["attr"])
        ctx = StepContext(
            name=spec["name"],
            inputs={k: Path(v) for k, v in spec["inputs"].items()},
            outputs={k: Path(v) for k, v in spec["outputs"].items()},
            params=spec["params"],
            workdir=Path(spec["workdir"]),
        )
        summary = _summarize(func(ctx))
    except BaseException as exc:  # report everything, including SystemExit/KeyboardInterrupt
        sys.stdout.flush()
        tb = exc.__traceback__.tb_next or exc.__traceback__  # hide this module's frame
        traceback.print_exception(type(exc), exc, tb, file=sys.stdout)
        return 1

    sys.stdout.flush()
    Path(result_path).write_text(json.dumps({"summary": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
