"""Content-addressed checkpoints.

A step is up to date when its fingerprint (code + config + input contents) matches the
one recorded after its last successful run *and* every output still exists with the
content it had then. File hashes are cached by (size, mtime) so unchanged large files
are not re-read on every run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Pipeline, Step

_STATE_VERSION = 1
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Fingerprint:
    code: str
    config: str
    inputs: dict[str, str]


class State:
    """Persistent record of successful runs, stored as JSON in the state dir."""

    def __init__(self, state_dir: Path):
        self.path = state_dir / "state.json"
        self.steps: dict[str, dict[str, Any]] = {}
        self._files: dict[str, list] = {}
        try:
            data = json.loads(self.path.read_text())
            if data.get("version") == _STATE_VERSION:
                self.steps = data.get("steps", {})
                self._files = data.get("files", {})
        except (FileNotFoundError, ValueError):
            pass

    # -- hashing ------------------------------------------------------------------

    def hash_file(self, path: Path) -> str:
        st = path.stat()
        key = str(path)
        cached = self._files.get(key)
        if cached and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
            return cached[2]
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
        digest = h.hexdigest()
        self._files[key] = [st.st_size, st.st_mtime_ns, digest]
        return digest

    def hash_path(self, path: Path) -> str | None:
        """Hash a file or a directory tree; None if it does not exist."""
        if path.is_file():
            return self.hash_file(path)
        if path.is_dir():
            h = hashlib.sha256(b"dir\0")
            for f in sorted(p for p in path.rglob("*") if p.is_file()):
                h.update(f.relative_to(path).as_posix().encode() + b"\0")
                h.update(self.hash_file(f).encode() + b"\0")
            return h.hexdigest()
        return None

    # -- persistence --------------------------------------------------------------

    def record(self, name: str, fp: Fingerprint, outputs: dict[str, str], **extra: Any) -> None:
        self.steps[name] = {
            "code": fp.code, "config": fp.config, "inputs": fp.inputs,
            "outputs": outputs, **extra,
        }
        self.save()

    def forget(self, name: str) -> None:
        if self.steps.pop(name, None) is not None:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        live = {k: v for k, v in self._files.items() if os.path.exists(k)}
        payload = json.dumps(
            {"version": _STATE_VERSION, "steps": self.steps, "files": live}, indent=1
        )
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".state-")
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp, self.path)


def module_file(pipeline: Pipeline, module: str) -> Path | None:
    """Locate a step module's source without importing it (or its parents)."""
    rel = Path(*module.split("."))
    for root in pipeline.python_path:
        for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
            if cand.is_file():
                return cand
    if "." not in module:  # top-level installed module: find_spec does not execute it
        spec = importlib.util.find_spec(module)
        if spec and spec.origin and os.path.isfile(spec.origin):
            return Path(spec.origin)
    return None


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def fingerprint(pipeline: Pipeline, step: Step, state: State) -> Fingerprint:
    code = hashlib.sha256()
    src = module_file(pipeline, step.module)
    code.update(state.hash_path(src).encode() if src else step.module.encode())
    for rel in step.watch:
        code.update(rel.encode() + b"\0" + (state.hash_path(pipeline.resolve(rel)) or "-").encode())
    config = _digest({
        "function": step.function, "params": step.params,
        "inputs": step.inputs, "outputs": step.outputs,
    })
    inputs = {k: state.hash_path(pipeline.resolve(rel)) or "-" for k, rel in step.inputs.items()}
    return Fingerprint(code=code.hexdigest(), config=config, inputs=inputs)


def stale_reason(pipeline: Pipeline, step: Step, state: State, fp: Fingerprint) -> str | None:
    """Why the step must run, or None if its checkpoint is valid."""
    prev = state.steps.get(step.name)
    if prev is None:
        return "no checkpoint"
    if prev.get("code") != fp.code:
        return "code changed"
    if prev.get("config") != fp.config:
        return "config changed"
    if prev.get("inputs") != fp.inputs:
        old = prev.get("inputs") or {}
        changed = sorted(k for k in fp.inputs if old.get(k) != fp.inputs[k])
        return f"inputs changed: {', '.join(changed)}" if changed else "inputs changed"
    recorded = prev.get("outputs", {})
    for key, rel in step.outputs.items():
        h = state.hash_path(pipeline.resolve(rel))
        if h is None:
            return f"output missing: {key}"
        if recorded.get(key) != h:
            return f"output modified: {key}"
    return None
