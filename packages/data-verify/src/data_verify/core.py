"""Shared machinery: loading, column typing, canonical row hashing, the check registry."""

from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from .spec import SPLIT_RANKS, Spec, SpecError, is_data_input

# -- outcomes and registry -----------------------------------------------------------------


@dataclass
class Outcome:
    status: str                    # pass | fail | skip
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    hint: str = ""
    suffix: str = ""               # appended to the check id: "<id>.<suffix>"


def passed(message: str, **details) -> Outcome:
    return Outcome("pass", message, details)


def failed(message: str, hint: str = "", **details) -> Outcome:
    return Outcome("fail", message, details, hint)


def skipped(message: str) -> Outcome:
    return Outcome("skip", message)


@dataclass
class CheckDef:
    id: str
    category: str
    severity: str
    title: str
    description: str
    options: dict[str, Any]
    fn: Callable[..., Any]


REGISTRY: dict[str, CheckDef] = {}


def check(id: str, category: str, severity: str, title: str, **options):
    """Register a check. The docstring is its description in `data-verify checks`."""
    def wrap(fn):
        REGISTRY[id] = CheckDef(id, category, severity, title,
                                " ".join((fn.__doc__ or "").split()), options, fn)
        return fn
    return wrap


# -- loading -------------------------------------------------------------------------------


def _specificity(pattern: str) -> int:
    return sum(ch not in "*?[]" for ch in pattern)


def resolve_check(cid: str, spec: Spec, default_sev: str, defaults: dict) -> tuple[str, dict]:
    """An exact id beats every glob; among globs the most specific one (most non-wildcard
    characters) wins, and on a tie the one written last."""
    sev, opt = default_sev, dict(defaults)
    globs = [(_specificity(p), i, o["severity"]) for i, (p, o) in enumerate(spec.checks.items())
             if p != cid and fnmatch.fnmatchcase(cid, p) and "severity" in o]
    if globs:
        sev = max(globs)[2]
    if cid in spec.checks:
        o = spec.checks[cid]
        sev = o.get("severity", sev)
        opt.update({k: x for k, x in o.items() if k != "severity"})
    return sev, opt


def sha256_path(path: Path) -> str:
    """Content hash of a file, or of a directory (relative names and contents, sorted)."""
    h = hashlib.sha256()
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(path)).encode() + b"\0")
            h.update(bytes.fromhex(sha256_path(f)))
        return h.hexdigest()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def path_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def read_table(path: Path, allow_pickle: bool = False) -> pd.DataFrame:
    try:
        return _read_table(path, allow_pickle)
    except SpecError:
        raise
    except Exception as e:  # noqa: BLE001  an unreadable input is a problem with the input
        raise SpecError(f"cannot read input {path.name}: {type(e).__name__}: {e}") from None


def _read_table(path: Path, allow_pickle: bool) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*")
                       if p.is_file() and not p.name.startswith((".", "_")))
        if files and all(p.suffix.lower() in (".parquet", ".pq") for p in files):
            return pd.read_parquet(path)
        frames = [_read_table(p, allow_pickle) for p in files if is_data_input(p)]
        if not frames:
            raise SpecError(f"{path}: directory holds no readable data files")
        return pd.concat(frames, ignore_index=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".feather", ".arrow"):
        return pd.read_feather(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix in (".pkl", ".pickle"):
        if not allow_pickle:
            raise SpecError(f"{path.name}: pickle files can run code when loaded, which is "
                            "exactly what a poisoned file would do. Write parquet/CSV instead, "
                            "or set params.allow_pickle = true if you trust the file.")
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise SpecError(f"{path.name}: pickle does not hold a DataFrame")
        return obj
    raise SpecError(f"{path.name}: unsupported data format '{suffix}'")


# -- column typing and canonical values ----------------------------------------------------


def _object_kind(s: pd.Series) -> str:
    sample = s.dropna().head(200)
    if len(sample) and all(isinstance(v, (dt.date, dt.datetime, pd.Timestamp)) for v in sample):
        return "datetime"
    if len(sample) and all(isinstance(v, (bool, np.bool_)) for v in sample):
        return "bool"
    return "string"


def kind(s: pd.Series) -> str:
    """bool | int | float | datetime | string (strings, categories, mixed objects)."""
    if pdt.is_bool_dtype(s.dtype):
        return "bool"
    if pdt.is_integer_dtype(s.dtype):
        return "int"
    if pdt.is_float_dtype(s.dtype):
        return "float"
    if pdt.is_datetime64_any_dtype(s.dtype):
        return "datetime"
    if pdt.is_timedelta64_dtype(s.dtype):
        return "float"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "string"
    if s.dtype == object:
        return _object_kind(s)
    return "string"


def is_numeric_kind(k: str) -> bool:
    return k in ("int", "float")


def _num_text(f: float) -> str:
    return str(int(f)) if f.is_integer() and abs(f) < 2**53 else str(f)


def value_texts(s: pd.Series) -> pd.Series:
    """A column as strings, with whole floats written as integers ("1.0" -> "1"), so a float
    column (any numeric CSV column with a NaN) compares equal to declared values."""
    if kind(s) != "float":
        return s.astype("string")
    f = s.astype("float64")
    out = f.astype("string")
    whole = np.isfinite(f) & (f % 1 == 0) & (f.abs() < 2**53)
    out[whole] = f[whole].astype("int64").astype("string")
    return out


def value_text(x: Any, numeric: bool = False) -> str:
    """A declared value (params) as the string `value_texts` gives the same value in data.
    With numeric = True a numeric-looking string ("1") matches the number too."""
    if numeric and isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)):
        return _num_text(float(x))
    return str(x)


def value_set(values, s: pd.Series) -> set[str]:
    """Declared values as `value_texts` strings for comparing against column `s`."""
    numeric = is_numeric_kind(kind(s))
    return {value_text(x, numeric) for x in values}


def to_datetime(s: pd.Series) -> pd.Series:
    """Datetimes as naive UTC; offsets that differ (a CSV round trip across DST) are fine."""
    if pdt.is_datetime64_any_dtype(s.dtype):
        out = s
    else:
        try:
            out = pd.to_datetime(s, errors="coerce", format="mixed")
        except ValueError:  # mixed offsets, or naive and tz-aware values together
            out = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def numeric_values(s: pd.Series) -> np.ndarray:
    """float64 view of a column for statistics (datetimes as ns, bools as 0/1)."""
    k = kind(s)
    if k == "datetime":
        t = to_datetime(s)
        out = t.to_numpy().astype("datetime64[ns]").astype("int64").astype("float64")
        out[t.isna().to_numpy()] = np.nan
        return out
    if pdt.is_timedelta64_dtype(s.dtype):
        out = s.to_numpy().astype("timedelta64[ns]").astype("int64").astype("float64")
        out[s.isna().to_numpy()] = np.nan
        return out
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


_NORMALIZE_RE = r"[\W_]+"


def canonical(s: pd.Series, near: bool = False, bits: int = 10) -> np.ndarray:
    """Values that hash the same when the data is the same, whatever the file format.

    Numbers become float64 (so 3 == 3.0 across CSV and parquet), -0.0 becomes 0.0, dates
    become nanoseconds, everything else becomes text. `near` also rounds numbers to `bits`
    bits of mantissa (about 3 significant digits at 10) and normalises text (case,
    punctuation, whitespace).
    """
    k = kind(s)
    if k in ("int", "float", "bool", "datetime"):
        x = numeric_values(s.astype("float64") if k == "bool" else s)
        x = np.where(x == 0, 0.0, x)
        if near:
            m, e = np.frexp(x)
            x = np.ldexp(np.round(m * (1 << bits)) / (1 << bits), e)
        return x
    text = s.astype("string")
    if near:
        text = (text.str.casefold().str.replace(_NORMALIZE_RE, " ", regex=True)
                .str.strip())
    return text.fillna("\0<NA>").to_numpy(dtype=object)


def row_hashes(df: pd.DataFrame, cols: list[str], near: bool = False, bits: int = 10) -> np.ndarray:
    """One uint64 per row over `cols`, built column by column to keep memory flat."""
    h = np.zeros(len(df), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for c in sorted(cols):
            hc = pd.util.hash_array(canonical(df[c], near, bits), categorize=False)
            h = (h * prime) ^ hc
    return h


# -- helpers for checks --------------------------------------------------------------------


def fmt(x: float) -> float:
    """Round for the report (deterministic, readable)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return x
    return float(f"{x:.6g}")


def frac(n: int, d: int) -> float:
    return fmt(n / d) if d else 0.0


def examples(values, n: int) -> list:
    out = []
    for v in values:
        if len(out) >= n:
            break
        out.append(jsonable(v))
    return out


def jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return fmt(f) if math.isfinite(f) else str(f)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, dt.date, dt.datetime, np.datetime64)):
        return str(pd.Timestamp(v)) if not isinstance(v, dt.date) or isinstance(v, dt.datetime) \
            else v.isoformat()
    if isinstance(v, tuple):
        return [jsonable(x) for x in v]
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (str, int, bool)):
        return v
    return str(v)


def matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


# -- the verifier --------------------------------------------------------------------------


class Verifier:
    """Holds the loaded data and everything the checks share (with caches).

    Custom checks (`params.custom`) receive this object. Useful members:
    `splits` (name -> DataFrame), `train_name`, `train`, `features`, `labels`,
    `dataset(name)`, `sql(query)`, `var(name)`, `add(...)`.
    """

    def __init__(self, spec: Spec, inputs: dict[str, Path], log=print):
        self.spec = spec
        self.inputs = inputs
        self.log = log
        self._datasets: dict[str, pd.DataFrame] = {}
        self._cache: dict[Any, Any] = {}
        self.custom_results: list[dict[str, Any]] = []
        self._duck = None
        self.rng_seed = spec.seed
        self.load_errors: dict[str, str] = {}
        self._load_splits()
        self._resolve_features()

    # -- data ------------------------------------------------------------------------------

    def dataset(self, name: str) -> pd.DataFrame:
        """Any data input of the step, loaded on first use."""
        if name in self.splits:
            return self.splits[name]
        if name not in self._datasets:
            if name not in self.inputs:
                raise SpecError(f"'{name}' is neither a split nor an input of the step")
            self._datasets[name] = read_table(self.inputs[name], self.spec.allow_pickle)
        return self._datasets[name]

    def _load_splits(self) -> None:
        s = self.spec
        self.splits: dict[str, pd.DataFrame] = {}
        if s.split_column:
            table = read_table(self.inputs[s.table], s.allow_pickle)
            if s.split_column not in table.columns:
                raise SpecError(f"split_column '{s.split_column}' is not a column of "
                                f"'{s.table}' (columns: {list(table.columns)[:20]})")
            col = table[s.split_column].astype("string")
            values = list(col.dropna().unique().tolist())
            names = s.split_names
            if not names:  # only recognised split names; a typo must not become a split
                names = sorted((x for x in values if x.lower() in SPLIT_RANKS),
                               key=lambda x: (SPLIT_RANKS[x.lower()], x))
                if not names:
                    raise SpecError(f"split column '{s.split_column}' holds no standard split "
                                    f"name (values: {sorted(values)[:10]}); list the split "
                                    "values in params.splits")
            self.unknown_split_values = sorted(set(values) - set(names))
            self.null_split_rows = int(col.isna().sum())
            for n in names:
                rows = (col == n).fillna(False).to_numpy()
                self.splits[n] = table.loc[rows].reset_index(drop=True)
            self.table_rows = len(table)
            del table
        else:
            self.unknown_split_values, self.null_split_rows = [], 0
            for split, inp in s.split_inputs.items():
                self.splits[split] = read_table(self.inputs[inp], s.allow_pickle)
            self.table_rows = sum(len(d) for d in self.splits.values())
        if not self.splits:
            raise SpecError("no data to verify: the step has no data inputs")
        names = list(self.splits)
        if s.train is not None:
            if s.train not in self.splits:
                raise SpecError(f"train = '{s.train}' is not a split (splits: {names})")
            self.train_name = s.train
        else:
            ranked = [n for n in names if SPLIT_RANKS.get(n.lower()) == 0]
            self.train_name = ranked[0] if ranked else names[0]
        for name in (*s.time_bounds, *(x for pair in s.time_order for x in pair)):
            if name not in self.splits:
                raise SpecError(f"time rules mention split '{name}', but the splits are {names}")

    @property
    def train(self) -> pd.DataFrame:
        return self.splits[self.train_name]

    def others(self) -> list[tuple[str, pd.DataFrame]]:
        return [(n, d) for n, d in self.splits.items() if n != self.train_name]

    def common_columns(self) -> list[str]:
        cols = None
        for d in self.splits.values():
            cols = list(d.columns) if cols is None else [c for c in cols if c in set(d.columns)]
        return cols or []

    def _resolve_features(self) -> None:
        s = self.spec
        roles = set(s.role_columns())
        train_cols = list(self.train.columns)
        if s.features is None:
            feats = [c for c in train_cols if c not in roles]
            self.features_declared = False
        else:
            feats = []
            for f in s.features:
                if any(ch in f for ch in "*?["):
                    feats += [c for c in train_cols if fnmatch.fnmatchcase(c, f)]
                else:
                    feats.append(f)
            self.features_declared = True
        feats = [f for f in dict.fromkeys(feats) if not matches(f, s.exclude)]
        self.declared_features = feats
        common = set(self.common_columns())
        # checks use the features every split actually has; schema.roles reports the rest
        self.features = [f for f in feats if f in common]

    # -- typed views -----------------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        return self.spec.labels

    def kind(self, col: str) -> str:
        key = ("kind", col)
        if key not in self._cache:
            self._cache[key] = kind(self.train[col]) if col in self.train.columns else \
                next((kind(d[col]) for d in self.splits.values() if col in d.columns), "string")
        return self._cache[key]

    def numeric_features(self) -> list[str]:
        return [f for f in self.features if self.kind(f) in ("int", "float")]

    def categorical_features(self) -> list[str]:
        return [f for f in self.features if self.kind(f) in ("string", "bool")]

    def has(self, col: str | None, split: str | None = None) -> bool:
        if col is None:
            return False
        frames = [self.splits[split]] if split else self.splits.values()
        return all(col in d.columns for d in frames)

    def task(self, label: str) -> str:
        """binary | multiclass | regression, from params.task or the train labels."""
        key = ("task", label)
        if key in self._cache:
            return self._cache[key]
        t = self.spec.task.get(label, "auto")
        if t == "auto":
            y = self.train[label].dropna() if label in self.train.columns else pd.Series([])
            k = kind(y) if len(y) else "string"
            n = y.nunique()
            if k in ("string", "bool"):
                t = "binary" if n <= 2 else "multiclass"
            elif k == "int" or (k == "float" and len(y) and
                                np.all(np.mod(y.to_numpy(dtype="float64"), 1) == 0)):
                t = "binary" if n <= 2 else ("multiclass" if n <= 20 else "regression")
            else:
                t = "regression"
        self._cache[key] = t
        return t

    def time(self, split: str) -> pd.Series:
        return self.time_of(split, self.spec.time)

    def time_of(self, split: str, col: str) -> pd.Series:
        """A time column as datetime64 (or float64 when the column is numeric)."""
        key = ("time", split, col)
        if key not in self._cache:
            s = self.splits[split][col]
            self._cache[key] = s.astype("float64") if kind(s) in ("int", "float") \
                else to_datetime(s)
        return self._cache[key]

    def time_value(self, value: Any, numeric: bool, where: str = "time_bounds"):
        try:
            if numeric:
                return float(value)
            ts = value if isinstance(value, pd.Timestamp) else pd.Timestamp(str(value))
            # time columns are naive UTC (see to_datetime); a bound with an offset becomes so too
            return ts.tz_convert("UTC").tz_localize(None) if ts.tz is not None else ts
        except (ValueError, TypeError):
            want = "a number" if numeric else "a date or time like '2023-05-30'"
            raise SpecError(f"{where} = {value!r}: time column '{self.spec.time}' is "
                            f"{'numeric' if numeric else 'a datetime'}, so this must be "
                            f"{want}") from None

    def time_is_numeric(self) -> bool:
        return self.kind(self.spec.time) in ("int", "float")

    def hashes(self, split: str, cols: tuple[str, ...], near: bool = False, bits: int = 10):
        key = ("hash", split, cols, near, bits)
        if key not in self._cache:
            self._cache[key] = row_hashes(self.splits[split], list(cols), near, bits)
        return self._cache[key]

    def sample(self, split: str) -> pd.DataFrame:
        """At most `sample_rows` rows of a split (seeded), for statistical checks."""
        key = ("sample", split)
        if key not in self._cache:
            d = self.splits[split]
            n = self.spec.sample_rows
            self._cache[key] = d if len(d) <= n else \
                d.sample(n=n, random_state=self.spec.seed).reset_index(drop=True)
        return self._cache[key]

    def in_source(self, split: str) -> np.ndarray | None:
        """Rows of a split that exist in params.source (matched on params.id when the source
        has it, else on all shared columns); None without a source."""
        s = self.spec
        if not s.source:
            return None

        def compute():
            src = self.dataset(s.source)
            cols = [c for c in s.id if c in src.columns] if s.id else []
            if not cols or not all(self.has(c) for c in cols):
                cols = [c for c in self.common_columns()
                        if c in src.columns and c != s.split_column]
            if not cols:
                return None
            return np.isin(self.hashes(split, tuple(cols)), row_hashes(src, cols))
        return self.cached(("in_source", split), compute)

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # -- extension API ---------------------------------------------------------------------

    def var(self, name: str) -> Any:
        if name not in self.spec.vars:
            raise SpecError(f"unknown var '{name}' (vars: {sorted(self.spec.vars)})")
        return self.spec.vars[name]

    def sql(self, query: str) -> pd.DataFrame:
        """Run a DuckDB query. Every data input is a view under its input name; every split
        is a view under its split name."""
        return self.duck().sql(query).df()

    def duck(self):
        if self._duck is None:
            try:
                import duckdb
            except ImportError:
                raise SpecError("SQL assertions need DuckDB: install data-verify[sql] or add "
                                "duckdb to the project") from None
            con = duckdb.connect()
            for name, path in self.inputs.items():
                if name in self.spec.config_inputs or not is_data_input(path):
                    continue
                con.sql(f'CREATE VIEW "{name}" AS SELECT * FROM {_duck_reader(path)}')
            s = self.spec
            for split, d in self.splits.items():
                if s.split_column:
                    if split == s.table:
                        continue
                    lit = split.replace("'", "''")
                    con.sql(f'CREATE OR REPLACE VIEW "{split}" AS SELECT * FROM "{s.table}" '
                            f'WHERE "{s.split_column}" = \'{lit}\'')
                elif split not in self.inputs or s.split_inputs.get(split) != split:
                    con.register(f"__split_{split}", d)
                    con.sql(f'CREATE OR REPLACE VIEW "{split}" AS SELECT * FROM "__split_{split}"')
            self._duck = con
        return self._duck

    def add(self, name: str, ok: bool, message: str, severity: str = "error",
            details: dict | None = None, hint: str = "", category: str = "custom") -> None:
        """Record a result from a custom check."""
        self.custom_results.append({
            "name": name, "ok": bool(ok), "message": message, "severity": severity,
            "details": details or {}, "hint": hint, "category": category})


def _duck_reader(path: Path) -> str:
    p = str(path).replace("'", "''")
    if path.is_dir():
        return f"read_parquet('{p}/**/*.parquet', hive_partitioning = true, union_by_name = true)"
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return f"read_parquet('{p}')"
    if suffix in (".csv", ".tsv"):
        return f"read_csv_auto('{p}')"
    if suffix in (".json", ".jsonl", ".ndjson"):
        return f"read_json_auto('{p}')"
    return f"read_csv_auto('{p}')"
