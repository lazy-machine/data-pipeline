"""Integrity and schema: the data is what it claims to be, in every split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import (
    check, examples, failed, frac, jsonable, kind, numeric_values, passed, sha256_path, skipped,
)


@check("integrity.pins", "integrity", "error", "Pinned inputs are byte-identical")
def pins(v, opt):
    """Every input listed in params.pins must have exactly that sha256 (directories are hashed
    recursively). Catches a raw file that was swapped, re-downloaded or edited, which is the
    simplest way to poison a dataset."""
    if not v.spec.pins:
        return skipped("no params.pins; pin raw inputs to detect tampering")
    bad = {}
    for name, want in sorted(v.spec.pins.items()):
        got = v.cached(("sha256", name), lambda n=name: sha256_path(v.inputs[n]))
        if got != want:
            bad[name] = {"expected": want, "actual": got}
    if bad:
        return failed(f"{len(bad)} pinned input(s) changed: {', '.join(bad)}",
                      "If the change is intended, update params.pins with the new sha256 "
                      "(listed in details.actual). Otherwise restore the original file.",
                      changed=bad)
    return passed(f"{len(v.spec.pins)} pinned input(s) match")


@check("schema.splits_present", "schema", "error", "Every split has rows", min_rows=1)
def splits_present(v, opt):
    """Each split must hold at least `min_rows` rows. In split_column mode, every row must
    belong to a declared split (no null or unknown split values)."""
    sizes = {n: len(d) for n, d in v.splits.items()}
    small = {n: k for n, k in sizes.items() if k < opt["min_rows"]}
    problems = []
    if small:
        problems.append(f"split(s) below {opt['min_rows']} rows: {small}")
    if v.unknown_split_values:
        problems.append(f"split column has undeclared values {v.unknown_split_values[:10]}")
    if v.null_split_rows:
        problems.append(f"{v.null_split_rows} rows have no split value")
    if problems:
        return failed("; ".join(problems),
                      "Every row must be in exactly one declared split. List all split values in "
                      "params.splits, or fix the step that assigns splits.",
                      rows=sizes, unknown_values=v.unknown_split_values,
                      null_rows=v.null_split_rows)
    return passed(f"{len(sizes)} split(s): " + ", ".join(f"{n}={k}" for n, k in sizes.items()),
                  rows=sizes)


@check("schema.roles", "schema", "error", "Declared columns exist in every split", unlabeled=[])
def roles(v, opt):
    """The label, id, group, time, weight and feature columns named in params must exist in
    every split. Splits listed in the `unlabeled` option may lack the label columns."""
    s = v.spec
    missing: dict[str, list[str]] = {}
    for name, d in v.splits.items():
        want = [c for c in s.role_columns() if c != s.split_column] + v.declared_features
        if name in opt["unlabeled"]:
            want = [c for c in want if c not in s.labels]
        gone = [c for c in dict.fromkeys(want) if c not in d.columns]
        if gone:
            missing[name] = gone
    if missing:
        first = next(iter(missing.items()))
        return failed(f"columns missing: {first[0]} lacks {first[1][:8]}"
                      + (f" (+{len(missing) - 1} more split(s))" if len(missing) > 1 else ""),
                      "Fix the column names in params, or the step that writes the split. For "
                      "an unlabeled split use checks.\"schema.roles\".unlabeled = [\"<split>\"].",
                      missing=missing)
    return passed(f"{len(v.features)} features and {len(s.role_columns())} role columns present",
                  features=len(v.features))


@check("schema.consistent", "schema", "error", "Splits share the same columns and types")
def consistent(v, opt):
    """All splits must have the same set of columns, and each column the same kind of type
    (numbers, text, dates, booleans). A column present in one split only, or parsed as text in
    one and numbers in another, means the splits were produced differently."""
    if len(v.splits) < 2:
        return skipped("one split only")
    ref = v.train_name
    ref_cols = set(v.train.columns)
    col_diff, type_diff = {}, {}
    norm = {"int": "number", "float": "number"}
    for name, d in v.others():
        extra, missing = sorted(set(d.columns) - ref_cols), sorted(ref_cols - set(d.columns))
        if extra or missing:
            col_diff[name] = {"extra": extra[:20], "missing": missing[:20]}
        for c in sorted(ref_cols & set(d.columns)):
            a, b = kind(v.train[c]), kind(d[c])
            if norm.get(a, a) != norm.get(b, b) and d[c].notna().any() and v.train[c].notna().any():
                type_diff.setdefault(name, {})[c] = [a, b]
    if col_diff or type_diff:
        msg = []
        if col_diff:
            msg.append("columns differ from " + ref + ": " + ", ".join(
                f"{n} (+{len(x['extra'])}/-{len(x['missing'])})" for n, x in col_diff.items()))
        if type_diff:
            msg.append("types differ: " + ", ".join(
                f"{n}.{c} {t[0]}≠{t[1]}" for n, cs in type_diff.items()
                for c, t in list(cs.items())[:3]))
        return failed("; ".join(msg), "Produce every split with the same code path and dtypes.",
                      columns=col_diff, types=type_diff)
    return passed(f"{len(ref_cols)} columns, same types in {len(v.splits)} splits")


def _schema_targets(v):
    names = list(v.splits)
    extra = v.spec.schema.get("datasets", [])
    return [(n, v.dataset(n)) for n in names + [e for e in extra if e not in names]]


@check("schema.expected", "schema", "error", "Columns match params.schema.columns")
def expected(v, opt):
    """When params.schema.columns is set, each split must have exactly those columns (and in
    that order when schema.ordered = true). Catches stray columns that could carry the label."""
    want = v.spec.schema.get("columns")
    if not want:
        return skipped("no params.schema.columns")
    ordered = v.spec.schema.get("ordered", False)
    bad = {}
    for name, d in v.splits.items():
        cols = list(d.columns)
        if (cols != want) if ordered else (set(cols) != set(want)):
            bad[name] = {"extra": sorted(set(cols) - set(want))[:20],
                         "missing": sorted(set(want) - set(cols))[:20],
                         "order_differs": ordered and set(cols) == set(want)}
    if bad:
        return failed(f"{len(bad)} split(s) differ from schema.columns: "
                      + ", ".join(f"{n} +{x['extra'][:3]} -{x['missing'][:3]}"
                                  for n, x in bad.items()), splits=bad)
    return passed(f"{len(want)} columns as declared")


_KINDS = {"bool", "int", "float", "numeric", "datetime", "string"}


@check("schema.dtypes", "schema", "error", "Column types match params.schema.dtypes")
def dtypes(v, opt):
    """params.schema.dtypes maps column -> kind (bool, int, float, numeric, datetime, string)
    or an exact pandas dtype name ("float32"). Every split must match."""
    want = v.spec.schema.get("dtypes")
    if not want:
        return skipped("no params.schema.dtypes")
    bad = []
    for name, d in _schema_targets(v):
        for col, w in want.items():
            if col not in d.columns:
                continue
            k = kind(d[col])
            ok = (k in ("int", "float")) if w == "numeric" else (k == w) if w in _KINDS \
                else str(d[col].dtype) == w
            if not ok:
                bad.append(f"{name}.{col}: {d[col].dtype} ({k}), expected {w}")
    if bad:
        return failed(f"{len(bad)} column type mismatch(es): " + "; ".join(bad[:5]),
                      mismatches=bad[:50])
    return passed(f"{len(want)} column types as declared")


@check("schema.ranges", "schema", "error", "Values stay within params.schema.ranges")
def ranges(v, opt):
    """params.schema.ranges maps column -> [min, max] or { min, max }. Values outside it are
    corrupt or injected (a classic poisoning signature: impossible values in train only)."""
    want = v.spec.schema.get("ranges")
    if not want:
        return skipped("no params.schema.ranges")
    bad = {}
    for name, d in _schema_targets(v):
        for col, r in want.items():
            if col not in d.columns:
                continue
            lo, hi = (r.get("min", -np.inf), r.get("max", np.inf)) if isinstance(r, dict) else r
            x = numeric_values(d[col])
            lo_v = float(pd.Timestamp(lo).value) if isinstance(lo, str) else float(lo)
            hi_v = float(pd.Timestamp(hi).value) if isinstance(hi, str) else float(hi)
            out = (x < lo_v) | (x > hi_v)
            if out.any():
                bad[f"{name}.{col}"] = {"rows": int(out.sum()), "fraction": frac(out.sum(), len(x)),
                                        "range": [jsonable(lo), jsonable(hi)],
                                        "examples": examples(d.loc[out, col].unique(), 5)}
    if bad:
        return failed(f"{sum(b['rows'] for b in bad.values())} value(s) out of range in "
                      f"{len(bad)} column(s): " + ", ".join(list(bad)[:5]), columns=bad)
    return passed(f"{len(want)} column range(s) respected")


@check("schema.allowed", "schema", "error", "Values are in params.schema.allowed")
def allowed(v, opt):
    """params.schema.allowed maps column -> list of permitted values."""
    want = v.spec.schema.get("allowed")
    if not want:
        return skipped("no params.schema.allowed")
    bad = {}
    for name, d in _schema_targets(v):
        for col, values in want.items():
            if col not in d.columns:
                continue
            s = d[col].dropna()
            ok = s.isin(values) | s.astype("string").isin([str(x) for x in values])
            if not ok.all():
                bad[f"{name}.{col}"] = {"rows": int((~ok).sum()),
                                        "examples": examples(s[~ok].unique(), 5)}
    if bad:
        return failed(f"unexpected values in {len(bad)} column(s): "
                      + ", ".join(f"{k} {x['examples']}" for k, x in list(bad.items())[:5]),
                      columns=bad)
    return passed(f"{len(want)} column(s) within allowed values")


@check("schema.not_null", "schema", "error", "Columns in params.schema.not_null have no nulls")
def not_null(v, opt):
    """Columns listed in params.schema.not_null must be complete in every split."""
    want = v.spec.schema.get("not_null")
    if not want:
        return skipped("no params.schema.not_null")
    bad = {}
    for name, d in _schema_targets(v):
        for col in want:
            if col in d.columns and (n := int(d[col].isna().sum())):
                bad[f"{name}.{col}"] = n
    if bad:
        return failed(f"nulls in {len(bad)} column(s): {bad}", columns=bad)
    return passed(f"{len(want)} column(s) complete")


@check("schema.unique", "schema", "error", "Keys in params.schema.unique are unique")
def unique(v, opt):
    """params.schema.unique lists columns (or lists of columns) that must not repeat within
    each split and auxiliary dataset."""
    want = v.spec.schema.get("unique")
    if not want:
        return skipped("no params.schema.unique")
    bad = {}
    for name, d in _schema_targets(v):
        for key in want:
            cols = [key] if isinstance(key, str) else list(key)
            if all(c in d.columns for c in cols) and (n := int(d.duplicated(cols).sum())):
                bad[f"{name}.{'+'.join(cols)}"] = n
    if bad:
        return failed(f"repeated keys: {bad}", keys=bad)
    return passed(f"{len(want)} key(s) unique")


@check("features.non_finite", "integrity", "error", "No infinite feature values")
def non_finite(v, opt):
    """Numeric features must not contain +inf or -inf (NaN is allowed and reported elsewhere).
    Infinities usually come from a division by zero in feature engineering."""
    bad = {}
    for name, d in v.splits.items():
        for f in v.numeric_features():
            x = numeric_values(d[f])
            if (n := int(np.isinf(x).sum())):
                bad[f"{name}.{f}"] = n
    if bad:
        return failed(f"infinite values in {len(bad)} split column(s): "
                      + ", ".join(f"{k}={n}" for k, n in list(bad.items())[:5]),
                      "Replace inf with NaN or clip it in the feature step.", columns=bad)
    return passed(f"{len(v.numeric_features())} numeric features finite")


@check("features.empty", "integrity", "warn", "No feature is empty in train")
def empty(v, opt):
    """A feature that is entirely null in the training split carries no information and often
    means a join or rename failed upstream."""
    bad = [f for f in v.features if v.train[f].isna().all()]
    if bad:
        return failed(f"{len(bad)} feature(s) entirely null in {v.train_name}: {bad[:10]}",
                      features=bad)
    return passed(f"{len(v.features)} features have values")


@check("features.constant", "integrity", "warn", "No feature is constant in train")
def constant(v, opt):
    """A feature with a single non-null value in the training split cannot help the model;
    a constant that differs between splits is a split-specific artifact."""
    bad = [f for f in v.features if v.train[f].nunique(dropna=True) == 1]
    if bad:
        return failed(f"{len(bad)} constant feature(s) in {v.train_name}: {bad[:10]}",
                      features=bad)
    return passed(f"{len(v.features)} features vary")
