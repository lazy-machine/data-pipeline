"""Split integrity: no row, entity or key is shared between splits, and the split was made by
the declared rule from the declared source."""

from __future__ import annotations

import hashlib
import re
from itertools import combinations

import numpy as np
import pandas as pd

from ..core import (
    check, examples, failed, fmt, frac, jsonable, passed, row_hashes, skipped, value_set,
    value_texts,
)


def _key_values(d: pd.DataFrame, cols: list[str], mask: np.ndarray, n: int) -> list:
    sub = d.loc[mask, cols].drop_duplicates().head(n)
    return [jsonable(r[0]) if len(cols) == 1 else [jsonable(x) for x in r]
            for r in sub.itertuples(index=False)]


def _overlap(v, cols: list[str], what: str, hint: str):
    if not cols:
        return skipped(f"no params.{what}")
    if not all(v.has(c) for c in cols):
        return skipped(f"{what} column(s) {cols} missing in some split (see schema.roles)")
    if len(v.splits) < 2:
        return skipped("one split only")
    pairs = {}
    for (a, _da), (b, db) in combinations(v.splits.items(), 2):
        ha, hb = v.hashes(a, tuple(cols)), v.hashes(b, tuple(cols))
        shared = np.intersect1d(ha, hb)
        if len(shared):
            mask = np.isin(hb, shared)
            pairs[f"{a}/{b}"] = {
                "shared": int(len(shared)),
                "fraction_of_" + b: frac(len(shared), len(np.unique(hb))),
                "rows_in_" + b: int(mask.sum()),
                "examples": _key_values(db, cols, mask, v.spec.max_examples),
            }
    if pairs:
        total = sum(p["shared"] for p in pairs.values())
        return failed(f"{total} {what} value(s) appear in more than one split: "
                      + ", ".join(f"{k} ({p['shared']}, e.g. {p['examples'][:2]})"
                                  for k, p in pairs.items()), hint, pairs=pairs)
    counts = {n: int(len(np.unique(v.hashes(n, tuple(cols))))) for n in v.splits}
    return passed(f"{what} disjoint across {len(v.splits)} splits", distinct=counts)


@check("split.id_unique", "split", "error", "Row ids are unique within each split")
def id_unique(v, opt):
    """The params.id key must identify one row per split. Repeated ids inside a split mean rows
    were duplicated (an oversampled or injected copy)."""
    cols = v.spec.id
    if not cols:
        return skipped("no params.id")
    if not all(v.has(c) for c in cols):
        return skipped(f"id column(s) {cols} missing in some split")
    bad = {}
    for name, d in v.splits.items():
        h = v.hashes(name, tuple(cols))
        dup = pd.Series(h).duplicated(keep=False).to_numpy()
        if dup.any():
            bad[name] = {"rows": int(dup.sum()), "examples": _key_values(d, cols, dup, 5)}
    if bad:
        return failed("repeated ids: " + ", ".join(f"{n}={x['rows']} rows" for n, x in bad.items()),
                      "Deduplicate before splitting, or set params.id to a key that is unique.",
                      splits=bad)
    return passed(f"ids unique in {len(v.splits)} splits")


@check("split.id_overlap", "split", "error", "No row id is in two splits")
def id_overlap(v, opt):
    """The same params.id value in train and in val/test means the same record is both learned
    and scored: the evaluation is contaminated."""
    return _overlap(v, v.spec.id, "id",
                    "Split after deduplicating by id, and assign each id to one split.")


@check("split.group_overlap", "split", "error", "No entity (group) is in two splits")
def group_overlap(v, opt):
    """Rows sharing params.group (a patient, user, device, document, ...) must all be in the
    same split. Otherwise the model can memorise the entity in train and recognise it in
    test: the most common leak in grouped and longitudinal data. Skipped when
    time_order_scope = "group" (per-series forecasting: every entity has an early part in
    train and a later part in test by design, and temporal.order checks that per entity)."""
    if v.spec.time_order_scope == "group" and v.spec.time_order:
        return skipped("time_order_scope = 'group': entities span splits by design; "
                       "temporal.order checks each entity's past/future order")
    return _overlap(v, v.spec.group, "group",
                    "Split by group (e.g. GroupShuffleSplit, or a stable hash of the group key) "
                    "instead of by row.")


def _bucket(values: pd.Series, rule: dict) -> np.ndarray:
    method = rule["method"]
    if method == "modulo":
        return (pd.to_numeric(values, errors="coerce").to_numpy() % rule["modulo"])
    algo = getattr(hashlib, method)
    tmpl, mod = rule["template"], rule["modulo"]
    out = np.empty(len(values), dtype="float64")
    for i, key in enumerate(values.astype("string").fillna("")):
        digest = algo(tmpl.replace("{key}", key).encode()).digest()
        out[i] = int(int.from_bytes(digest[:8], "big") / 2**64 * mod)
    return out


@check("split.rule", "split", "error", "Split assignment follows params.split_rule")
def rule(v, opt):
    """Recomputes each row's split from params.split_rule and compares. Methods: a stable hash
    (sha256/sha1/md5 of template.format(key), first 8 bytes as a fraction, times modulo,
    then [lo, hi) buckets per split), `modulo` (integer key % modulo), or explicit `values`
    lists. Proves the split is deterministic and was not hand-edited."""
    r = v.spec.split_rule
    if not r:
        return skipped("no params.split_rule")
    key = r["key"]
    if not v.has(key):
        return failed(f"split_rule.key '{key}' missing in some split")
    wrong, checked, examples_ = {}, 0, []
    for name, d in v.splits.items():
        keys = d[key].drop_duplicates()
        checked += len(keys)
        if r["method"] == "values":
            allowed_vals = value_set(r["values"].get(name, []), keys)
            bad = ~value_texts(keys).isin(allowed_vals).to_numpy()
        else:
            if name not in r["buckets"]:
                wrong[name] = len(keys)
                continue
            lo, hi = r["buckets"][name]
            b = _bucket(keys, r)
            bad = ~((b >= lo) & (b < hi))
        if bad.any():
            wrong[name] = int(bad.sum())
            examples_ += examples(keys[bad], 3)
    if wrong:
        return failed(f"{sum(wrong.values())} {key} value(s) are in the wrong split: {wrong}",
                      "The split was not produced by the declared rule. Recompute it with the "
                      "rule, or fix params.split_rule.", wrong=wrong, examples=examples_[:10])
    return passed(f"{checked} {key} value(s) in the split their {r['method']} rule assigns")


@check("split.ratios", "split", "warn", "Split sizes match params.split_ratios")
def ratios(v, opt):
    """Each split's share of rows (or of groups, with unit = "groups") must be within
    `tolerance` of params.split_ratios. A train share far above plan can mean rows from the
    evaluation splits were moved into train."""
    r = v.spec.split_ratios
    if not r:
        return skipped("no params.split_ratios")
    if r["unit"] == "groups":
        if not v.spec.group:
            return failed("split_ratios.unit = 'groups' needs params.group")
        sizes = {n: len(np.unique(v.hashes(n, tuple(v.spec.group)))) for n in v.splits}
    else:
        sizes = {n: len(d) for n, d in v.splits.items()}
    total = sum(sizes.values())
    actual = {n: frac(k, total) for n, k in sizes.items()}
    off = {n: {"expected": f, "actual": actual.get(n, 0.0)}
           for n, f in r["fractions"].items()
           if abs(actual.get(n, 0.0) - f) > r["tolerance"]}
    if off:
        return failed("split shares off plan: " + ", ".join(
            f"{n} {x['actual']:.3f} vs {x['expected']:.3f}" for n, x in off.items()),
            actual=actual, off=off, tolerance=r["tolerance"])
    return passed("shares " + ", ".join(f"{n}={a:.3f}" for n, a in actual.items()),
                  actual=actual)


@check("split.conservation", "split", "error",
       "Splits come from params.source, nothing added", exact=False, columns=[])
def conservation(v, opt):
    """Compares the splits with the pre-split data in params.source. Every split row must exist
    in the source (rows that do not were injected or altered: poisoning). With exact = true,
    every source row must also land in exactly one split (nothing dropped or duplicated).
    Rows are matched on params.id when set, otherwise on the `columns` option, otherwise on
    every column the source and the splits share."""
    s = v.spec
    if not s.source:
        return skipped("no params.source (the pre-split data)")
    src = v.dataset(s.source)
    common = [c for c in v.common_columns() if c in src.columns and c != s.split_column]
    cols = opt["columns"] or ([c for c in s.id if c in src.columns] if s.id else []) or common
    missing = [c for c in cols if c not in src.columns or not v.has(c)]
    if missing or not cols:
        return failed(f"cannot match rows: columns {missing or 'in common'} missing",
                      "Set checks.\"split.conservation\".columns to columns present in both.")
    hs = pd.Series(row_hashes(src, cols)).value_counts()
    injected, per_split = 0, {}
    split_counts = pd.Series(dtype="int64")
    ex = []
    for name, d in v.splits.items():
        h = pd.Series(v.hashes(name, tuple(cols)))
        vc = h.value_counts()
        split_counts = split_counts.add(vc, fill_value=0)
        not_in = ~h.isin(hs.index)
        per_split[name] = {"rows": len(h), "not_in_source": int(not_in.sum())}
        injected += int(not_in.sum())
        if not_in.any():
            ex += _key_values(d, cols[:3], not_in.to_numpy(), 3)
    # more copies in the splits than in the source is also an addition
    extra_copies = int((split_counts - hs.reindex(split_counts.index, fill_value=0)).clip(lower=0)
                       .sum()) - injected
    details = {"matched_on": cols[:20], "splits": per_split, "source_rows": int(hs.sum()),
               "extra_copies": max(extra_copies, 0)}
    problems = []
    if injected:
        problems.append(f"{injected} split row(s) not found in source")
    if extra_copies > 0:
        problems.append(f"{extra_copies} row(s) appear more often in the splits than in source")
    if opt["exact"]:
        dropped = int((hs - split_counts.reindex(hs.index, fill_value=0)).clip(lower=0).sum())
        details["dropped"] = dropped
        if dropped:
            problems.append(f"{dropped} source row(s) are in no split")
    if problems:
        return failed("; ".join(problems),
                      "Split rows must be copies of source rows. Rows that are not were added "
                      "or modified after the split. If the splits are transformed (scaled, "
                      "encoded), match on untouched columns via the `columns` option.",
                      examples=ex[:10], **details)
    return passed(f"all {sum(p['rows'] for p in per_split.values())} split rows found in "
                  f"{s.source}", **details)


_ENTITY_NAME = re.compile(
    r"(^|_)(id|uuid|guid|key|serial|account|user|customer|client|patient|subject|person|member|"
    r"device|machine|session|household|family|author|store|site|vehicle|player|owner)"
    r"(_?(id|no|num|number|code|key))?$|[a-z0-9]Id$|ID$", re.IGNORECASE)


@check("split.undeclared_group", "split", "warn",
       "Entity-like columns crossing splits are declared as params.group",
       max_rows_per_value=200, min_shared=0.05)
def undeclared_group(v, opt):
    """Runs when params.group is not set. Looks for columns that look like entity keys (named
    like *_id, user, patient, device, serial, ..., or text with many distinct values) whose
    values repeat over several rows (at most `max_rows_per_value` on average) and are shared
    by more than `min_shared` of an evaluation split's rows with train. If rows with the same
    value belong to one entity, they leak between splits; declaring the column as
    params.group turns this warning into the blocking split.group_overlap check."""
    if v.spec.group:
        return skipped("params.group is set (split.group_overlap checks it)")
    if len(v.splits) < 2:
        return skipped("one split only")
    roles = set(v.spec.role_columns()) - set(v.spec.group)
    hits = {}
    tr = v.train
    for col in v.common_columns():
        if col in roles or col in v.spec.id:
            continue
        k = v.kind(col)
        if k not in ("string", "int"):
            continue
        s = tr[col].dropna()
        if len(s) < 50:
            continue
        distinct = s.nunique()
        per_value = len(s) / max(distinct, 1)
        named = bool(_ENTITY_NAME.search(col))
        keyish_text = k == "string" and distinct >= 50 and distinct / len(s) >= 0.02 and \
            float(s.astype("string").str.len().mean()) <= 40
        if distinct < 20 or not (1.2 <= per_value <= opt["max_rows_per_value"]):
            continue
        if not (named or keyish_text):
            continue
        seen = set(value_texts(s))
        shared = {}
        for name, d in v.others():
            o = value_texts(d[col].dropna())
            if len(o):
                f = float(o.isin(seen).mean())
                if f > opt["min_shared"]:
                    shared[name] = frac(f, 1)
        if shared:
            hits[col] = {"rows_per_value": fmt(per_value), "shared_rows": shared,
                         "is_feature": col in v.features}
    if hits:
        return failed("entity-like column(s) shared across splits with no params.group: "
                      + ", ".join(f"{c} (~{x['rows_per_value']} rows per value, shared by "
                                  + ", ".join(f"{n} {f:.0%}" for n, f in x["shared_rows"].items())
                                  + ")" for c, x in hits.items()),
                      "If rows with the same value are the same entity (patient, user, device), "
                      "set params.group to that column and split by it; and keep the key out "
                      "of the features.", columns=hits)
    return passed("no undeclared entity key crosses the splits")
