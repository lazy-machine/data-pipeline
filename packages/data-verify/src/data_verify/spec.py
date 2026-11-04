"""Turn the step's ``params`` (plus an optional ``spec`` input file) into a validated Spec.

Every key is checked: an unknown or misspelled key is an error with a suggestion, so a
typo can never silently switch a check off.

Values can be read from other inputs instead of being repeated in the pipeline file::

    features = { from = "schema", key = "features" }             # JSON/TOML input "schema"
    train_end = { from = "study", key = "split.train_end" }
    stride = { from = "sample", key = "scales.{scale}.stride_days" }   # {scale} is looked up first

Inside a list, a reference that resolves to a list is spliced in place::

    columns = ["serial_number", "date", { from = "schema", key = "features" }]
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


class SpecError(ValueError):
    """The verifier's configuration is invalid. The message says which key and why."""


# Split names recognised automatically, with their position in time / in the workflow:
# 0 = fitted on, 1 = used for tuning/selection, 2 = held out for the final score.
SPLIT_RANKS = {
    "train": 0, "training": 0, "fit": 0,
    "val": 1, "valid": 1, "validation": 1, "dev": 1, "tune": 1, "calib": 1, "calibration": 1,
    "test": 2, "testing": 2, "holdout": 2, "eval": 2, "evaluation": 2, "oot": 2,
}

TABULAR_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".feather", ".arrow", ".jsonl",
                    ".ndjson", ".json", ".pkl", ".pickle"}
CONFIG_SUFFIXES = {".toml", ".json", ".yaml", ".yml"}

TOP_KEYS = {
    # data layout
    "splits", "split_column", "table", "train", "source",
    # column roles
    "label", "task", "id", "group", "time", "label_time", "point_in_time", "weight",
    "features", "exclude", "forbidden",
    # rules
    "time_bounds", "time_order", "embargo", "time_order_scope",
    "split_rule", "split_ratios", "schema", "label_values", "label_range",
    "fitted_levels", "pins",
    # extensions
    "assert", "custom", "vars",
    # behaviour
    "checks", "fail_on", "sample_rows", "seed", "allow_pickle", "max_hash_mb", "max_examples",
}

SEVERITIES = ("error", "warn", "off")
TASKS = ("auto", "binary", "multiclass", "regression")


def suggest(word: str, options) -> str:
    close = difflib.get_close_matches(word, sorted(options), n=2)
    if not close:
        return ""
    return " (did you mean " + " or ".join(f"'{c}'" for c in close) + "?)"


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


# -- reading config inputs -----------------------------------------------------------------

def read_config_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix == ".toml":
        return tomllib.loads(text)
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise SpecError(f"{path.name}: reading YAML needs PyYAML installed") from None
        return yaml.safe_load(text)
    raise SpecError(f"{path.name}: references can read .toml, .json or .yaml files")


def lookup(doc: Any, key: str, where: str) -> Any:
    """Follow a dotted key; list items by index. `{a.b}` segments are looked up first."""
    key = re.sub(r"\{([^{}]+)\}", lambda m: str(lookup(doc, m.group(1), where)), key)
    cur = doc
    walked = []
    for part in key.split("."):
        walked.append(part)
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif (isinstance(cur, list) and part.lstrip("-").isdigit()
              and -len(cur) <= int(part) < len(cur)):
            cur = cur[int(part)]
        else:
            options = list(cur) if isinstance(cur, dict) else []
            raise SpecError(f"{where}: key '{'.'.join(walked)}' not found"
                            + suggest(part, options))
    return cur


def _is_ref(value: Any) -> bool:
    return isinstance(value, dict) and "from" in value and "key" in value \
        and set(value) <= {"from", "key", "default"}


class Refs:
    """Resolves ``{ from = "<input>", key = "a.b" }`` against the step's config inputs."""

    def __init__(self, inputs: dict[str, Path]):
        self.inputs = inputs
        self.docs: dict[str, Any] = {}
        self.used: set[str] = set()

    def doc(self, name: str, where: str) -> Any:
        if name not in self.docs:
            _expect(name in self.inputs, f"{where}: from = '{name}' is not one of the step's "
                    f"inputs {sorted(self.inputs)}" + suggest(name, self.inputs))
            self.docs[name] = read_config_file(self.inputs[name])
            self.used.add(name)
        return self.docs[name]

    def resolve(self, value: Any, where: str) -> Any:
        if _is_ref(value):
            doc = self.doc(value["from"], where)
            try:
                return self.resolve(lookup(doc, value["key"], f"{where} (from {value['from']})"),
                                    where)
            except SpecError:
                if "default" in value:
                    return value["default"]
                raise
        if isinstance(value, dict):
            return {k: self.resolve(v, f"{where}.{k}") for k, v in value.items()}
        if isinstance(value, list):
            out: list[Any] = []
            for i, item in enumerate(value):
                resolved = self.resolve(item, f"{where}[{i}]")
                if _is_ref(item) and isinstance(resolved, list):
                    out.extend(resolved)
                else:
                    out.append(resolved)
            return out
        return value


# -- small parsers -------------------------------------------------------------------------

def _str_list(value: Any, where: str, allow_none: bool = True) -> list[str]:
    if value is None and allow_none:
        return []
    if isinstance(value, str):
        return [value]
    _expect(isinstance(value, list) and all(isinstance(v, str) for v in value),
            f"{where} must be a column name or a list of column names, got {value!r}")
    return list(value)


def _opt_str(value: Any, where: str) -> str | None:
    _expect(value is None or isinstance(value, str), f"{where} must be a string, got {value!r}")
    return value


def parse_duration(value: Any, where: str) -> pd.Timedelta | float:
    """'7D', '12h', '30 days' -> Timedelta; a bare number -> days (or units of a numeric time)."""
    if isinstance(value, bool):
        raise SpecError(f"{where} must be a duration like '7D' or a number")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return pd.Timedelta(value)
    except (ValueError, TypeError):
        raise SpecError(f"{where} must be a duration like '7D', '12h' or a number, "
                        f"got {value!r}") from None


def parse_time_order(value: Any) -> list[tuple[str, str]]:
    """'train < val < test' (every pair of the chain) or ['train < test', 'val < test'] or
    [['train','test']]."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else value
    _expect(isinstance(items, list), "time_order must be a string like 'train < test' or a list")
    pairs: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str):
            chain = [p.strip() for p in item.split("<")]
            _expect(len(chain) >= 2 and all(chain),
                    f"time_order entry {item!r} must look like 'train < val < test'")
            pairs += [(chain[i], chain[j]) for i in range(len(chain))
                      for j in range(i + 1, len(chain))]
        elif isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            pairs.append((item[0], item[1]))
        else:
            raise SpecError(f"time_order entry {item!r} must be 'a < b' or ['a', 'b']")
    return list(dict.fromkeys(pairs))


@dataclass
class Assertion:
    name: str
    dataset: str = "*"           # split, input name, or "*" for every split
    expr: str | None = None      # pandas expression that must be true on every row
    sql: str | None = None       # DuckDB query returning the violating rows
    sql_count: str | None = None  # DuckDB query returning one number: the violation count
    severity: str = "error"
    category: str = "assert"
    description: str = ""
    hint: str = ""


@dataclass
class Spec:
    # layout
    split_inputs: dict[str, str] = field(default_factory=dict)   # split -> input (file mode)
    split_column: str | None = None
    table: str | None = None
    split_names: list[str] = field(default_factory=list)          # column mode, may be empty
    train: str | None = None
    source: str | None = None
    # roles
    labels: list[str] = field(default_factory=list)
    task: dict[str, str] = field(default_factory=dict)
    id: list[str] = field(default_factory=list)
    group: list[str] = field(default_factory=list)
    time: str | None = None
    label_time: str | None = None
    point_in_time: list[str] = field(default_factory=list)
    weight: str | None = None
    features: list[str] | None = None       # None = every non-role column
    exclude: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    # rules
    time_bounds: dict[str, dict[str, Any]] = field(default_factory=dict)
    time_order: list[tuple[str, str]] = field(default_factory=list)
    embargo: Any = None
    time_order_scope: str = "global"
    split_rule: dict[str, Any] | None = None
    split_ratios: dict[str, Any] | None = None
    schema: dict[str, Any] = field(default_factory=dict)
    label_values: list[Any] | None = None
    label_range: list[float] | None = None
    fitted_levels: list[dict[str, Any]] = field(default_factory=list)
    pins: dict[str, str] = field(default_factory=dict)
    asserts: list[Assertion] = field(default_factory=list)
    custom: list[str] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)
    # behaviour
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)  # id or glob -> options
    fail_on: str = "error"
    sample_rows: int = 250_000
    seed: int = 0
    allow_pickle: bool = False
    max_hash_mb: float = 4096
    max_examples: int = 5
    config_inputs: set[str] = field(default_factory=set)

    def role_columns(self) -> list[str]:
        cols = [*self.labels, *self.id, *self.group, *self.point_in_time]
        for c in (self.time, self.label_time, self.weight, self.split_column):
            if c:
                cols.append(c)
        return list(dict.fromkeys(cols))


def is_data_input(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in TABULAR_SUFFIXES


def split_input_names(params: dict[str, Any], inputs: dict[str, Path]) -> list[str]:
    """The inputs holding split data, without reading them. Follows the layout rules of
    ``parse`` but never raises: this also runs before the inputs exist."""
    source = params.get("source")
    data = [n for n, path in inputs.items() if is_data_input(path) and n not in ("spec", source)]
    splits = params.get("splits")
    if params.get("split_column"):
        table = params.get("table")
        return [table] if table in inputs else data
    if isinstance(splits, dict) and "from" not in splits:
        return [n for n in splits.values() if n in inputs]
    if isinstance(splits, list):
        return [n for n in splits if n in inputs]
    return [n for n in data if n.lower() in SPLIT_RANKS] or data


def _option_type(default: Any) -> str:
    if isinstance(default, bool):
        return "true or false"
    if isinstance(default, int):
        return "an integer"
    if isinstance(default, float):
        return "a number"
    if isinstance(default, list):
        return "a list"
    return "a string"


def _is_type(value: Any, want: str) -> bool:
    if want == "true or false":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    return {"an integer": isinstance(value, int),
            "a number": isinstance(value, (int, float)),
            "a list": isinstance(value, list)}.get(want, isinstance(value, str))


def normalize_checks(checks: Any, known_checks) -> dict[str, dict[str, Any]]:
    """Validate params.checks: ids (globs allowed for severity), severities, options."""
    _expect(isinstance(checks, dict), "checks must be a table: \"<check id>\" = options")
    for cid, opt in checks.items():
        if isinstance(opt, str):
            opt = {"severity": opt}
        _expect(isinstance(opt, dict), f"checks.\"{cid}\" must be a severity string or a table")
        matched = [k for k in known_checks if fnmatch.fnmatchcase(k, cid)]
        _expect(bool(matched), f"checks: no check matches '{cid}'" + suggest(cid, known_checks)
                + ". See `data-verify checks`.")
        if "severity" in opt:
            _expect(opt["severity"] in SEVERITIES,
                    f"checks.\"{cid}\".severity must be one of {SEVERITIES}")
        extra = set(opt) - {"severity"}
        if extra:
            _expect(len(matched) == 1 and cid == matched[0],
                    f"checks.\"{cid}\": options other than severity need an exact check id")
            known_opts = known_checks[cid].options
            bad = extra - set(known_opts)
            _expect(not bad, f"checks.\"{cid}\": unknown option(s) " + ", ".join(
                f"'{k}'{suggest(k, known_opts)}" for k in sorted(bad))
                + f"; options: {sorted(known_opts) or 'none'}")
            for k in sorted(extra):
                want = _option_type(known_opts[k])
                _expect(_is_type(opt[k], want), f"checks.\"{cid}\".{k} must be {want}, not "
                        f"{opt[k]!r} (default: {known_opts[k]!r})")
        checks[cid] = opt
    return checks


def check_structure(raw: dict[str, Any], known_checks) -> None:
    """The checks that need no data and no config files: keys, check ids and options,
    assertion fields. `data-verify lint` runs only this when inputs are not built yet."""
    unknown = sorted(set(raw) - TOP_KEYS)
    _expect(not unknown, "unknown params key(s): " + ", ".join(
        f"'{k}'{suggest(k, TOP_KEYS)}" for k in unknown)
        + ". See `data-verify params` for every key.")
    if isinstance(raw.get("checks"), dict):
        normalize_checks(dict(raw["checks"]), known_checks)
    asserts = raw.get("assert", [])
    for i, a in enumerate([asserts] if isinstance(asserts, dict) else asserts or []):
        if isinstance(a, dict):
            allowed = {"name", "dataset", "expr", "sql", "sql_count", "severity", "category",
                       "description", "hint"}
            bad = set(a) - allowed
            _expect(not bad, f"assert[{i}]: unknown key(s) " + ", ".join(
                f"'{k}'{suggest(k, allowed)}" for k in sorted(bad)))


def parse(params: dict[str, Any], inputs: dict[str, Path], known_checks) -> Spec:
    """Validate `params` (merged over an optional `spec` input) into a Spec."""
    refs = Refs(inputs)
    raw: dict[str, Any] = {}
    if "spec" in inputs:
        base = refs.doc("spec", "inputs.spec")
        _expect(isinstance(base, dict), "inputs.spec must hold a table of verifier params")
        raw.update(base.get("params", base) if isinstance(base.get("params"), dict) else base)
    for k, v in params.items():
        if k == "checks" and isinstance(v, dict) and isinstance(raw.get("checks"), dict):
            raw["checks"] = {**raw["checks"], **v}
        else:
            raw[k] = v

    check_structure(raw, known_checks)
    p = refs.resolve(raw, "params")
    s = Spec()

    # -- layout ----------------------------------------------------------------------------
    s.split_column = _opt_str(p.get("split_column"), "split_column")
    s.table = _opt_str(p.get("table"), "table")
    s.source = _opt_str(p.get("source"), "source")
    splits = p.get("splits")
    data_inputs = [n for n, path in inputs.items() if is_data_input(path) and n != "spec"]

    if s.split_column:
        if s.table is None:
            candidates = [n for n in data_inputs if n != s.source]
            _expect(len(candidates) == 1,
                    f"split_column is set: say which input holds the table with "
                    f"params.table = one of {candidates or sorted(inputs)}")
            s.table = candidates[0]
        _expect(s.table in inputs, f"table = '{s.table}' is not an input"
                + suggest(s.table, inputs))
        s.split_names = _str_list(splits, "splits")
    else:
        _expect(s.table is None, "table only makes sense together with split_column")
        if isinstance(splits, dict):
            for name, inp in splits.items():
                _expect(isinstance(inp, str) and inp in inputs,
                        f"splits.{name} = {inp!r} must be the name of an input"
                        + suggest(str(inp), inputs))
            s.split_inputs = dict(splits)
        elif splits is not None:
            names = _str_list(splits, "splits")
            for n in names:
                _expect(n in inputs, f"splits: '{n}' is not an input of the step"
                        + suggest(n, inputs) + "; with one table holding every split, "
                        "set split_column instead")
            s.split_inputs = {n: n for n in names}
        else:
            found = [n for n in inputs if n.lower() in SPLIT_RANKS and is_data_input(inputs[n])]
            found.sort(key=lambda n: SPLIT_RANKS[n.lower()])
            s.split_inputs = {n: n for n in found}
        if not s.split_inputs:
            _expect(len(data_inputs) <= 1 or s.source,
                    "no splits found: name the split inputs train/val/test, list them in "
                    f"params.splits, or set params.split_column (data inputs: {data_inputs})")
            # a single dataset without splits: verify it alone (schema, labels, leakage
            # in features) and treat it as the training data
            single = [n for n in data_inputs if n != s.source]
            if single:
                s.split_inputs = {single[0]: single[0]}

    s.train = _opt_str(p.get("train"), "train")
    if s.source is not None:
        _expect(s.source in inputs, f"source = '{s.source}' is not an input"
                + suggest(s.source, inputs))

    # -- roles -----------------------------------------------------------------------------
    s.labels = _str_list(p.get("label"), "label")
    task = p.get("task", "auto")
    if isinstance(task, str):
        _expect(task in TASKS, f"task must be one of {TASKS}, got {task!r}")
        s.task = {lbl: task for lbl in s.labels}
    else:
        _expect(isinstance(task, dict), "task must be a string or a table label = task")
        for lbl, t in task.items():
            _expect(lbl in s.labels, f"task.{lbl}: '{lbl}' is not a label column")
            _expect(t in TASKS, f"task.{lbl} must be one of {TASKS}, got {t!r}")
        s.task = {lbl: task.get(lbl, "auto") for lbl in s.labels}
    s.id = _str_list(p.get("id"), "id")
    s.group = _str_list(p.get("group"), "group")
    s.time = _opt_str(p.get("time"), "time")
    s.label_time = _opt_str(p.get("label_time"), "label_time")
    s.point_in_time = _str_list(p.get("point_in_time"), "point_in_time")
    s.weight = _opt_str(p.get("weight"), "weight")
    feats = p.get("features", "auto")
    s.features = None if feats == "auto" else _str_list(feats, "features", allow_none=False)
    s.exclude = _str_list(p.get("exclude"), "exclude")
    s.forbidden = _str_list(p.get("forbidden"), "forbidden")

    # -- rules -----------------------------------------------------------------------------
    tb = p.get("time_bounds", {})
    _expect(isinstance(tb, dict), "time_bounds must be a table: split = { min = ..., max = ... }")
    for split, b in tb.items():
        _expect(isinstance(b, dict) and set(b) <= {"min", "max"} and b,
                f"time_bounds.{split} must be a table with min and/or max")
    s.time_bounds = tb
    s.time_order = parse_time_order(p.get("time_order"))
    if p.get("embargo") is not None:
        s.embargo = parse_duration(p["embargo"], "embargo")
    s.time_order_scope = p.get("time_order_scope", "global")
    _expect(s.time_order_scope in ("global", "group"),
            "time_order_scope must be 'global' or 'group'")
    if (s.time_bounds or s.time_order or s.label_time or s.point_in_time) and not s.time:
        raise SpecError("time_bounds, time_order, label_time and point_in_time need "
                        "params.time (the column holding each row's timestamp)")

    if (rule := p.get("split_rule")) is not None:
        _expect(isinstance(rule, dict), "split_rule must be a table")
        allowed = {"key", "method", "template", "modulo", "buckets", "values"}
        bad = set(rule) - allowed
        _expect(not bad, f"split_rule: unknown key(s) {sorted(bad)}; allowed {sorted(allowed)}")
        _expect(isinstance(rule.get("key"), str), "split_rule.key must be a column name")
        method = rule.setdefault("method", "sha256")
        _expect(method in ("sha256", "sha1", "md5", "modulo", "values"),
                "split_rule.method must be sha256, sha1, md5, modulo or values")
        if method == "values":
            _expect(isinstance(rule.get("values"), dict),
                    "split_rule.values must map split -> list of key values")
        else:
            buckets = rule.get("buckets")
            _expect(isinstance(buckets, dict) and buckets,
                    "split_rule.buckets must map split -> [lo, hi) bucket range")
            for k, b in buckets.items():
                _expect(isinstance(b, list) and len(b) == 2 and b[0] < b[1],
                        f"split_rule.buckets.{k} must be [lo, hi) with lo < hi")
            rule.setdefault("modulo", 100)
            rule.setdefault("template", "{key}")
            _expect("{key}" in rule["template"], "split_rule.template must contain {key}")
        s.split_rule = rule

    if (ratios := p.get("split_ratios")) is not None:
        _expect(isinstance(ratios, dict), "split_ratios must be a table: split = fraction")
        opts = {"tolerance", "unit"}
        fr = {k: v for k, v in ratios.items() if k not in opts}
        _expect(fr and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in fr.values()),
                "split_ratios values must be fractions between 0 and 1")
        unit = ratios.get("unit", "rows")
        _expect(unit in ("rows", "groups"), "split_ratios.unit must be 'rows' or 'groups'")
        s.split_ratios = {"fractions": fr, "tolerance": float(ratios.get("tolerance", 0.05)),
                          "unit": unit}

    schema = p.get("schema", {})
    _expect(isinstance(schema, dict), "schema must be a table")
    allowed = {"columns", "ordered", "dtypes", "ranges", "allowed", "not_null", "unique",
               "datasets"}
    bad = set(schema) - allowed
    _expect(not bad, "schema: unknown key(s) " + ", ".join(
        f"'{k}'{suggest(k, allowed)}" for k in sorted(bad)))
    s.schema = schema

    if (lv := p.get("label_values")) is not None:
        _expect(isinstance(lv, list) and lv, "label_values must be a non-empty list")
        s.label_values = lv
    if (lr := p.get("label_range")) is not None:
        _expect(isinstance(lr, list) and len(lr) == 2, "label_range must be [min, max]")
        s.label_range = [float(lr[0]), float(lr[1])]

    fl = p.get("fitted_levels", [])
    fl = [fl] if isinstance(fl, dict) else fl
    _expect(isinstance(fl, list), "fitted_levels must be a list of tables")
    for i, item in enumerate(fl):
        where = f"fitted_levels[{i}]"
        _expect(isinstance(item, dict), f"{where} must be a table")
        allowed = {"name", "levels", "column", "dataset", "where", "allow", "top_k"}
        bad = set(item) - allowed
        _expect(not bad, f"{where}: unknown key(s) {sorted(bad)}; allowed {sorted(allowed)}")
        _expect(isinstance(item.get("levels"), list), f"{where}.levels must be a list")
        _expect(isinstance(item.get("column"), str), f"{where}.column must be a column name")
    s.fitted_levels = fl

    pins = p.get("pins", {})
    _expect(isinstance(pins, dict), "pins must be a table: input = \"<sha256>\"")
    for k, v in pins.items():
        _expect(k in inputs, f"pins.{k}: not an input of the step" + suggest(k, inputs))
        _expect(isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v or ""),
                f"pins.{k} must be a 64-character sha256 hex digest")
    s.pins = {k: v.lower() for k, v in pins.items()}

    vars_ = p.get("vars", {})
    _expect(isinstance(vars_, dict), "vars must be a table of name = value")
    s.vars = vars_

    asserts = p.get("assert", [])
    asserts = [asserts] if isinstance(asserts, dict) else asserts
    _expect(isinstance(asserts, list), "assert must be a list of tables ([[...assert]])")
    names = set()
    for i, a in enumerate(asserts):
        where = f"assert[{i}]"
        _expect(isinstance(a, dict), f"{where} must be a table")
        allowed = {"name", "dataset", "expr", "sql", "sql_count", "severity", "category",
                   "description", "hint"}
        bad = set(a) - allowed
        _expect(not bad, f"{where}: unknown key(s) " + ", ".join(
            f"'{k}'{suggest(k, allowed)}" for k in sorted(bad)))
        _expect(isinstance(a.get("name"), str) and re.fullmatch(r"[\w\-.]+", a["name"]),
                f"{where}.name must be an identifier-like string")
        _expect(a["name"] not in names, f"{where}: duplicate assertion name '{a['name']}'")
        names.add(a["name"])
        kinds = [k for k in ("expr", "sql", "sql_count") if a.get(k)]
        _expect(len(kinds) == 1, f"assert '{a['name']}' needs exactly one of expr, sql, sql_count")
        sev = a.get("severity", "error")
        _expect(sev in SEVERITIES, f"assert '{a['name']}': severity must be one of {SEVERITIES}")
        s.asserts.append(Assertion(**{**a, "severity": sev}))

    s.custom = _str_list(p.get("custom"), "custom")
    for c in s.custom:
        _expect(bool(re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", c)),
                f"custom entry {c!r} must look like 'module:function'")

    # -- behaviour -------------------------------------------------------------------------
    s.checks = normalize_checks(p.get("checks", {}), known_checks)
    s.fail_on = p.get("fail_on", "error")
    _expect(s.fail_on in ("error", "warn", "never"), "fail_on must be 'error', 'warn' or 'never'")
    s.sample_rows = int(p.get("sample_rows", s.sample_rows))
    _expect(s.sample_rows >= 1000, "sample_rows must be at least 1000")
    s.seed = int(p.get("seed", 0))
    s.allow_pickle = bool(p.get("allow_pickle", False))
    s.max_hash_mb = float(p.get("max_hash_mb", s.max_hash_mb))
    s.max_examples = int(p.get("max_examples", 5))
    s.config_inputs = set(refs.used) | ({"spec"} if "spec" in inputs else set())
    return s
