"""Content-addressed checkpoints.

A step is up to date when its fingerprint (code + config + input contents) matches the
one recorded after its last successful run *and* every output still exists with the
content it had then. File hashes are cached by (size, mtime) so unchanged large files
are not re-read on every run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement

from .config import Pipeline, Step

_STATE_VERSION = 1
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Fingerprint:
    code: str
    config: str
    inputs: dict[str, str]
    imports: dict[str, str] = field(default_factory=dict)
    packages: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


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
            "imports": fp.imports, "packages": fp.packages, "env": fp.env,
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


def _local_module(pipeline: Pipeline, module: str) -> Path | None:
    """The file ``module`` resolves to under ``python_path``; installed modules are not local."""
    rel = Path(*module.split("."))
    for root in pipeline.python_path:
        for cand in (root / rel / "__init__.py", root / rel.with_suffix(".py")):  # as Python does
            if cand.is_file():
                return cand
    return None


def module_file(pipeline: Pipeline, module: str) -> Path | None:
    """Locate a step module's source without importing it (or its parents)."""
    if found := _local_module(pipeline, module):
        return found
    if "." not in module:  # top-level installed module: find_spec does not execute it
        spec = importlib.util.find_spec(module)
        if spec and spec.origin and os.path.isfile(spec.origin):
            return Path(spec.origin)
    return None


def _prefixes(dotted: str) -> list[str]:
    parts = dotted.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _imported_names(module: str, path: Path) -> set[str]:
    """Every module name ``path`` may import, found statically (the file is never run)."""
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return set()  # still hashed by the caller; the worker reports the real error
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(_prefixes(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                if node.level - 1 >= len(parts):
                    continue  # beyond the top-level package: fails at import time anyway
                base = ".".join(parts[:len(parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            elif node.module:
                target = node.module
            else:
                continue
            names.update(_prefixes(target))
            names.update(f"{target}.{a.name}" for a in node.names if a.name != "*")
    return names


def local_imports(pipeline: Pipeline, module: str) -> list[Path]:
    """Local files a step module imports, directly or through other local modules.

    Parent packages count too, since Python runs their ``__init__.py`` on import.
    Dynamic imports (``importlib.import_module``) cannot be seen; ``watch`` covers them.
    """
    src = _local_module(pipeline, module)
    if src is None:
        return []
    seen = {src}
    queue = [(module, src)]
    parents = set(_prefixes(module)[:-1])
    while queue:
        name, path = queue.pop()
        for cand in sorted(_imported_names(name, path) | parents):
            found = _local_module(pipeline, cand)
            if found is not None and found not in seen:
                seen.add(found)
                queue.append((cand, found))
        parents = set()
    return sorted(seen - {src})


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def installed_version(name: str) -> str:
    """The installed version; ``requires`` may also name an import, such as ``sklearn``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "import-only"


def _packages(step: Step) -> dict[str, str]:
    found = {}
    for spec in step.requires:
        try:
            name = Requirement(spec).name
        except InvalidRequirement:
            continue  # reported by preflight
        found[name] = installed_version(name)
    return dict(sorted(found.items()))


def _env(step: Step) -> dict[str, str]:
    """Hashes, never values: variables often hold secrets, and state.json is a plain file."""
    values = {name: os.environ.get(name) for name in step.env}
    return {name: "unset" if v is None else hashlib.sha256(v.encode()).hexdigest()
            for name, v in sorted(values.items())}


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
    imports = {
        Path(os.path.relpath(f, pipeline.workdir)).as_posix(): state.hash_file(f)
        for f in local_imports(pipeline, step.module)
    }
    return Fingerprint(code=code.hexdigest(), config=config, inputs=inputs,
                       imports=dict(sorted(imports.items())),
                       packages=_packages(step), env=_env(step))


def _few(names: list[str]) -> str:
    return ", ".join(names[:3]) + (", …" if len(names) > 3 else "")


def stale_reason(pipeline: Pipeline, step: Step, state: State, fp: Fingerprint) -> str | None:
    """Why the step must run, or None if its checkpoint is valid.

    A checkpoint written before imports, packages or env were tracked is backfilled in
    place rather than invalidated, so upgrading does not rerun every step; the caller
    saves the state.
    """
    prev = state.steps.get(step.name)
    if prev is None:
        return "no checkpoint"
    if prev.get("code") != fp.code:
        return "code changed"
    if "imports" not in prev:
        prev["imports"] = fp.imports
    elif prev["imports"] != fp.imports:
        old = prev["imports"]
        changed = sorted(k for k in old.keys() | fp.imports.keys() if old.get(k) != fp.imports.get(k))
        return f"code changed: {_few(changed)}"
    if "packages" not in prev:
        prev["packages"] = fp.packages
    elif prev["packages"] != fp.packages:
        old = prev["packages"]
        changed = sorted(k for k in old.keys() | fp.packages.keys()
                         if old.get(k) != fp.packages.get(k))
        moves = [f"{k} {old.get(k, 'none')} → {fp.packages.get(k, 'none')}" for k in changed]
        return f"packages changed: {_few(moves)}"
    if "env" not in prev:
        prev["env"] = fp.env
    elif prev["env"] != fp.env:
        old = prev["env"]
        changed = sorted(k for k in old.keys() | fp.env.keys() if old.get(k) != fp.env.get(k))
        return f"env changed: {_few(changed)}"
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
