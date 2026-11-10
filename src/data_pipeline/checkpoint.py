"""Content-addressed checkpoints.

A step is up to date when its fingerprint (code, config, input contents, package versions
and declared environment variables) matches the one recorded after its last successful
run *and* every output still exists with the content it had then. File hashes are cached
by (size, mtime, ctime, inode) so unchanged large files are not re-read on every run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .config import Pipeline, Step

_STATE_VERSION = 1
_CHUNK = 1 << 20
# A file changed within this long of being hashed can change again without moving its
# timestamps (they tick coarsely), so its hash is never reused: it is read again.
_RACY_NS = 2_000_000_000
_RECORD_DICTS = ("inputs", "outputs", "imports", "packages", "env")


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
        self._racy: set[str] = set()
        self.problems: list[str] = []  # why saved checkpoints were dropped, for the report
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (ValueError, OSError) as exc:
            self.problems.append(f"{self.path.name} is unreadable ({exc}); every step reruns")
            return
        if not isinstance(data, dict) or type(data.get("version")) is not int \
                or data["version"] != _STATE_VERSION:
            self.problems.append(f"{self.path.name} is not a version {_STATE_VERSION} state "
                                 "file; every step reruns")
            return
        steps = data.get("steps", {})
        if not isinstance(steps, dict):
            self.problems.append(f"{self.path.name}: 'steps' is not a table; every step reruns")
            steps = {}
        for name, rec in steps.items():
            if _valid_record(rec):
                self.steps[name] = rec
            else:
                self.problems.append(f"{self.path.name}: checkpoint of '{name}' is malformed; "
                                     "it reruns")
        files = data.get("files", {})
        if isinstance(files, dict):  # a cache: a bad entry only means reading the file again
            self._files = {k: v for k, v in files.items() if _valid_file_entry(v)}

    # -- hashing ------------------------------------------------------------------

    def hash_file(self, path: Path) -> str:
        """A file's SHA-256, re-read only when its metadata moved.

        Size and mtime alone can be put back (``os.utime``) after the bytes change; ctime
        cannot be set from user space and the inode changes when a file is replaced, so all
        four have to match. A cached hash still trusts the clock and the filesystem: a full
        re-read needs a deleted state file.
        """
        now = time.time_ns()  # before stat: a change after it is what a racy hash would miss
        st = path.stat()
        key = str(path)
        meta = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
        cached = self._files.get(key)
        if cached and cached[:4] == meta and key not in self._racy:
            return cached[4]
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
        digest = h.hexdigest()
        self._files[key] = [*meta, digest]
        if now - max(st.st_mtime_ns, st.st_ctime_ns) < _RACY_NS:
            self._racy.add(key)
        else:
            self._racy.discard(key)
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
        live = {k: v for k, v in self._files.items()
                if k not in self._racy and os.path.exists(k)}
        payload = json.dumps(
            {"version": _STATE_VERSION, "steps": self.steps, "files": live}, indent=1
        )
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".state-")
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp, self.path)


def _valid_file_entry(entry: Any) -> bool:
    return (isinstance(entry, list) and len(entry) == 5
            and all(type(v) is int for v in entry[:4]) and isinstance(entry[4], str))


def _valid_record(rec: Any) -> bool:
    """A checkpoint stale_reason can compare without crashing; anything else reruns."""
    if not isinstance(rec, dict):
        return False
    if not all(isinstance(rec.get(k), str) for k in ("code", "config")):
        return False
    if not isinstance(rec.get("summary", ""), str):
        return False
    return all(k not in rec or (isinstance(rec[k], dict)
                                and all(isinstance(v, str) for v in rec[k].values()))
               for k in _RECORD_DICTS)


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
    """The installed version; ``requires`` may also name an import, such as ``sklearn``,
    which has no version to track. _packages decides what a specifier on one means."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        found = "." not in name and importlib.util.find_spec(name)
        return "import-only" if found else "not installed"


def _packages(step: Step) -> dict[str, str]:
    found = {}
    for spec in step.requires:
        try:
            req = Requirement(spec)
        except InvalidRequirement:
            continue  # reported by preflight
        version = installed_version(req.name)
        if version == "import-only" and req.specifier:  # no version to test it against
            version = "not installed"
        found[canonicalize_name(req.name)] = version
    return dict(sorted(found.items()))


def _env(step: Step) -> dict[str, str]:
    """Hashes, never values: variables often hold secrets, and state.json is a plain file."""
    values = {name: os.environ.get(name) for name in step.env}
    return {name: "unset" if v is None else hashlib.sha256(os.fsencode(v)).hexdigest()
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
