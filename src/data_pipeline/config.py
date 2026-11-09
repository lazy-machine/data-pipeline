"""Load and validate a pipeline definition from TOML.

Example::

    [pipeline]
    name = "demo"
    python_path = ["steps"]          # extra import roots, relative to the TOML file

    [[steps]]
    name = "clean"
    function = "prep:clean"          # module:function
    inputs = { raw = "data/raw.csv" }
    outputs = { table = "build/clean.csv" }
    params = { drop_na = true }
    requires = ["pandas>=2"]         # packages that must be installed
    critical = true                  # false -> failure is a warning, pipeline continues
    clean_outputs = true             # delete the outputs just before the step runs
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")
_FUNC_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")

_PIPELINE_KEYS = {"name", "workdir", "python_path", "state_dir"}
_STEP_KEYS = {
    "name", "function", "inputs", "outputs", "params", "requires",
    "depends_on", "critical", "timeout", "watch", "env", "description", "clean_outputs",
}


class ConfigError(Exception):
    """The pipeline definition is invalid."""


@dataclass(frozen=True)
class Step:
    name: str
    function: str
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    critical: bool = True
    timeout: float | None = None
    watch: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    description: str = ""
    clean_outputs: bool = False

    @property
    def module(self) -> str:
        return self.function.split(":", 1)[0]

    @property
    def attr(self) -> str:
        return self.function.split(":", 1)[1]


@dataclass(frozen=True)
class Pipeline:
    name: str
    path: Path
    workdir: Path
    python_path: list[Path]
    state_dir: Path
    steps: list[Step]
    # step name -> names of steps it must run after (explicit + inferred from files)
    upstream: dict[str, list[str]]

    def resolve(self, rel: str) -> Path:
        return (self.workdir / rel).resolve()

    def step(self, name: str) -> Step:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(name)

    def cleaning_problem(self, rel: str) -> str | None:
        """Why deleting the output ``rel`` could delete more than that output, or None.

        Checked when the file is loaded and again just before deleting, because an
        earlier step may have turned a folder on the path into a symlink in between.
        """
        entry = output_entry(self.workdir, rel)
        if entry == self.workdir or not entry.is_relative_to(self.workdir):
            return f"it must lie inside the workdir {self.workdir}"
        if entry.is_relative_to(self.state_dir):
            return f"it lies inside the state folder {self.state_dir}"
        watched = [self.resolve(w) for s in self.steps for w in s.watch]
        top = {s.module.split(".")[0] for s in self.steps}
        sources = [root / f"{name}{ext}" for root in self.python_path for name in top
                   for ext in ("", ".py")]
        for p in [self.path, self.state_dir, *self.python_path, *watched, *sources]:
            if p.is_relative_to(entry):
                return f"it contains {p}"
        return None


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _str_map(value: Any, where: str) -> dict[str, str]:
    _expect(isinstance(value, dict), f"{where} must be a table of name = \"path\"")
    for k, v in value.items():
        _expect(isinstance(v, str) and v, f"{where}.{k} must be a non-empty path string")
    return dict(value)


def _str_list(value: Any, where: str) -> list[str]:
    _expect(
        isinstance(value, list) and all(isinstance(v, str) for v in value),
        f"{where} must be a list of strings",
    )
    return list(value)


def _parse_step(raw: Any, idx: int) -> Step:
    where = f"steps[{idx}]"
    _expect(isinstance(raw, dict), f"{where} must be a table")
    unknown = set(raw) - _STEP_KEYS
    _expect(not unknown, f"{where}: unknown key(s) {sorted(unknown)}")

    name = raw.get("name")
    _expect(isinstance(name, str) and bool(_NAME_RE.match(name)),
            f"{where}.name must be an identifier-like string, got {name!r}")
    where = f"step '{name}'"

    func = raw.get("function")
    _expect(isinstance(func, str) and bool(_FUNC_RE.match(func)),
            f"{where}: function must look like 'package.module:function', got {func!r}")

    critical = raw.get("critical", True)
    _expect(isinstance(critical, bool), f"{where}: critical must be true or false")

    timeout = raw.get("timeout")
    _expect(timeout is None or (isinstance(timeout, (int, float)) and timeout > 0),
            f"{where}: timeout must be a positive number of seconds")

    params = raw.get("params", {})
    _expect(isinstance(params, dict), f"{where}: params must be a table")

    description = raw.get("description", "")
    _expect(isinstance(description, str), f"{where}: description must be a string")

    clean_outputs = raw.get("clean_outputs", False)
    _expect(isinstance(clean_outputs, bool), f"{where}: clean_outputs must be true or false")

    return Step(
        name=name,
        function=func,
        inputs=_str_map(raw.get("inputs", {}), f"{where}.inputs"),
        outputs=_str_map(raw.get("outputs", {}), f"{where}.outputs"),
        params=params,
        requires=_str_list(raw.get("requires", []), f"{where}.requires"),
        depends_on=_str_list(raw.get("depends_on", []), f"{where}.depends_on"),
        critical=critical,
        timeout=float(timeout) if timeout is not None else None,
        watch=_str_list(raw.get("watch", []), f"{where}.watch"),
        env=_str_list(raw.get("env", []), f"{where}.env"),
        description=description,
        clean_outputs=clean_outputs,
    )


def overlaps(a: Path, b: Path) -> bool:
    """True when the paths are equal or one lies inside the other, as a file in a directory."""
    return a.is_relative_to(b) or b.is_relative_to(a)


def output_entry(workdir: Path, rel: str) -> Path:
    """The directory entry an output names: its parent resolved, its last part kept.

    Cleaning removes this entry, so a symlink output is unlinked, never followed.
    """
    path = workdir / rel
    return path.parent.resolve() / path.name


def _build_graph(steps: list[Step], resolve) -> dict[str, list[str]]:
    producers: dict[Path, tuple[str, str]] = {}  # output path -> (step, path as written)
    for s in steps:
        for rel in s.outputs.values():
            p = resolve(rel)
            for out, (other, other_rel) in producers.items():
                _expect(p != out,
                        f"output '{rel}' is produced by both '{other}' and '{s.name}'")
                # the outer step's hash would include the inner step's file
                _expect(other == s.name or not overlaps(p, out),
                        f"outputs overlap: '{other_rel}' of '{other}' and '{rel}' of '{s.name}'")
            producers[p] = (s.name, rel)

    names = {s.name for s in steps}
    upstream: dict[str, list[str]] = {}
    for s in steps:
        deps: list[str] = []
        for d in s.depends_on:
            _expect(d in names, f"step '{s.name}' depends on unknown step '{d}'")
            _expect(d != s.name, f"step '{s.name}' depends on itself")
            if d not in deps:
                deps.append(d)
        for rel in s.inputs.values():
            p = resolve(rel)
            for out, (prod, _) in producers.items():
                if not overlaps(p, out):
                    continue
                _expect(prod != s.name, f"step '{s.name}' consumes its own output '{rel}'")
                if prod not in deps:
                    deps.append(prod)
        upstream[s.name] = deps
    return upstream


def _topological(steps: list[Step], upstream: dict[str, list[str]]) -> list[Step]:
    """Stable topological sort: keeps TOML order wherever dependencies allow."""
    done: set[str] = set()
    ordered: list[Step] = []
    pending = list(steps)
    while pending:
        for i, s in enumerate(pending):
            if all(d in done for d in upstream[s.name]):
                ordered.append(s)
                done.add(s.name)
                del pending[i]
                break
        else:
            cycle = ", ".join(s.name for s in pending)
            raise ConfigError(f"dependency cycle among steps: {cycle}")
    return ordered


def load(path: str | Path) -> Pipeline:
    path = Path(path).resolve()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name}: invalid TOML: {exc}") from None

    meta = data.get("pipeline", {})
    _expect(isinstance(meta, dict), "[pipeline] must be a table")
    unknown = set(meta) - _PIPELINE_KEYS
    _expect(not unknown, f"[pipeline]: unknown key(s) {sorted(unknown)}")
    unknown = set(data) - {"pipeline", "steps"}
    _expect(not unknown, f"unknown top-level key(s) {sorted(unknown)}")

    base = path.parent
    workdir = (base / meta.get("workdir", ".")).resolve()
    _expect(workdir.is_dir(), f"workdir does not exist: {workdir}")
    python_path = [(base / p).resolve() for p in _str_list(meta.get("python_path", ["."]),
                                                            "[pipeline].python_path")]
    state_dir = (workdir / meta.get("state_dir", ".pipeline")).resolve()

    raw_steps = data.get("steps", [])
    _expect(isinstance(raw_steps, list) and raw_steps, "define at least one [[steps]] entry")
    steps = [_parse_step(r, i) for i, r in enumerate(raw_steps)]

    seen: set[str] = set()
    for s in steps:
        _expect(s.name not in seen, f"duplicate step name '{s.name}'")
        seen.add(s.name)

    resolve = lambda rel: (workdir / rel).resolve()  # noqa: E731
    upstream = _build_graph(steps, resolve)
    pipeline = Pipeline(
        name=meta.get("name", path.stem),
        path=path,
        workdir=workdir,
        python_path=python_path,
        state_dir=state_dir,
        steps=_topological(steps, upstream),
        upstream=upstream,
    )
    for s in steps:
        for rel in s.outputs.values() if s.clean_outputs else []:
            where = f"step '{s.name}': clean_outputs can't delete output '{rel}'"
            _expect(Path(rel).name not in ("", ".."),
                    f"{where}: it must name a file or directory")
            problem = pipeline.cleaning_problem(rel)
            _expect(problem is None, f"{where}: {problem}")
            out = resolve(rel)
            for other in steps:
                for inp in other.inputs.values():
                    _expect(resolve(inp) == out or not resolve(inp).is_relative_to(out),
                            f"{where}: step '{other.name}' reads '{inp}' inside it, and the "
                            f"loader can't tell that file from an old output. Declare "
                            f"'{rel}' itself as the input, or don't clean it")
    return pipeline
