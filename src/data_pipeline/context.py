"""The object handed to every step function."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepContext:
    """Everything a step needs: resolved paths and its parameters.

    A step function has the signature ``def step(ctx: StepContext) -> Any``. Its return
    value (``None``, a string, or a flat dict of metrics) is shown on the status line.
    Anything printed is captured and only shown if the step fails (or with ``-v``).
    """

    name: str
    inputs: dict[str, Path]
    outputs: dict[str, Path]
    params: dict[str, Any] = field(default_factory=dict)
    workdir: Path = field(default_factory=Path.cwd)

    def log(self, *parts: Any) -> None:
        print(f"[{self.name}]", *parts, flush=True)
