"""Duplicates: the same example in two splits (memorised, not generalised), and suspicious
repetition or label conflicts inside the training data (injected copies, flipped labels)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..core import check, failed, frac, passed, skipped
from .splits import _ENTITY_NAME

_MAX_REPEATS = 0.01  # share of repeated rows above which columns no longer tell rows apart


def _without_keys(v, cols: tuple[str, ...], key_share: float, bits: int) -> tuple[str, ...]:
    """Drop columns that act as row or entity keys, so that a copy under a new key still
    matches. The columns stay when the rest cannot tell rows apart on their own."""
    keys = set()
    for c in cols:
        s = v.train[c].dropna()
        if s.empty:
            continue
        values = s.nunique()
        share = values / len(s)
        short_text = v.kind(c) == "string" and float(s.astype("string").str.len().mean()) <= 40
        named_int = v.kind(c) == "int" and bool(_ENTITY_NAME.search(c))
        if share >= 0.98 and v.kind(c) == "int" or \
                values >= 50 and share >= key_share and (short_text or named_int):
            keys.add(c)
    rest = tuple(c for c in cols if c not in keys)
    if not keys or not rest:
        return cols
    repeats = pd.Series(v.hashes(v.train_name, rest, True, bits)).duplicated().mean()
    return rest if repeats <= _MAX_REPEATS else cols


def _cross_group_rate(v, cols, near: bool, bits: int) -> float | None:
    """Share of training rows whose feature vector also occurs in another group of the
    training split: how often different entities look identical by nature."""
    g = v.spec.group
    if not g or not all(v.has(c) for c in g):
        return None
    if v.spec.time_order_scope == "group" and v.spec.time_order:
        # entities span splits by design: compare with any repeat inside train
        h = pd.Series(v.hashes(v.train_name, cols, near, bits))
        return float(h.duplicated(keep=False).mean()) if len(h) else 0.0
    df = pd.DataFrame({"h": v.hashes(v.train_name, cols, near, bits),
                       "g": v.hashes(v.train_name, tuple(g))})
    groups = df.groupby("h")["g"].nunique()
    return float(df["h"].isin(groups.index[groups > 1]).mean()) if len(df) else 0.0


def _cross(v, opt, near: bool):
    feats = tuple(v.features)
    if not feats:
        return skipped("no features")
    if len(v.splits) < 2:
        return skipped("one split only")
    bits = opt.get("bits", 10)
    if opt.get("columns"):
        missing = [c for c in opt["columns"] if not v.has(c)]
        if missing:
            return failed(f"`columns` option names columns missing in some split: {missing}")
        feats = tuple(opt["columns"])
    cols = feats + (tuple(v.spec.labels) if opt.get("include_label") else ())
    cols = tuple(c for c in cols if v.has(c))
    all_cols = cols
    if near and not opt.get("columns"):
        cols = _without_keys(v, cols, opt["key_share"], bits)
    pairs, worst = {}, 0.0
    for (a, _), (b, _) in combinations(v.splits.items(), 2):
        ha, hb = v.hashes(a, cols, near, bits), v.hashes(b, cols, near, bits)
        hit = np.isin(hb, ha)
        if near:  # report only what the exact check does not already see
            hit &= ~np.isin(v.hashes(b, all_cols), v.hashes(a, all_cols))
        n = int(hit.sum())
        if n:
            worst = max(worst, n / len(hb))
            pairs[f"{a}/{b}"] = {"rows": n, "fraction": frac(n, len(hb)),
                                 "example_rows_of_" + b: np.flatnonzero(hit)[:5].tolist()}
    kind = "near-duplicate" if near else "duplicate"
    base = _cross_group_rate(v, cols, near, bits) if opt["baseline"] else None
    allowed = opt["max_fraction"]
    if base is not None:
        allowed = max(allowed, opt["ratio"] * base)
    details = {"pairs": pairs, "compared_columns": len(cols), "max_fraction": opt["max_fraction"]}
    if base is not None:
        details.update(cross_group_rate_in_train=frac(base, 1), allowed=frac(allowed, 1))
    summary = ", ".join(f"{k} {p['rows']} rows ({p['fraction']:.2%})" for k, p in pairs.items())
    if pairs and worst > allowed:
        why = (f"; different entities share a vector in only {base:.3%} of {v.train_name} rows, "
               f"so this is more than chance" if base is not None else "")
        return failed(f"{kind} feature rows shared between splits: {summary}{why}",
                      "The model sees these evaluation examples during training. Deduplicate "
                      "before splitting, or split by a group key that keeps copies together "
                      "(params.group). If identical rows are legitimate (few low-cardinality "
                      "features), raise checks.\"duplicates."
                      f"{'near_cross_split' if near else 'cross_split'}\".max_fraction.",
                      **details)
    if pairs:
        reason = (f"no more than different {v.train_name} entities share by chance "
                  f"({base:.3%})" if base is not None else f"<= max_fraction {allowed:.3%}")
        return passed(f"{kind}s within tolerance: {summary}; {reason}", **details)
    return passed(f"no {kind} feature rows across {len(v.splits)} splits", **details)


@check("duplicates.cross_split", "duplicates", "error",
       "No example appears in two splits", max_fraction=0.0, include_label=False,
       baseline=True, ratio=2.0, columns=[])
def cross_split(v, opt):
    """Rows with identical feature values in two splits (compared on all features, numbers
    normalised across file formats). Fails when the share of a later split's rows that also
    occur in an earlier split exceeds `max_fraction` (default 0). When params.group is set
    (and `baseline` is true), different entities can legitimately look identical (e.g. new
    devices with no readings yet), so the allowance becomes `ratio` times the share of
    training rows that match a row of another training entity: copies beyond what chance
    produces inside train still fail. `include_label = true` only counts copies that also
    share the label. `columns` compares on those columns instead of all features."""
    return _cross(v, opt, near=False)


@check("duplicates.near_cross_split", "duplicates", "warn",
       "No near-copy of an example appears in two splits", max_fraction=0.0, bits=10,
       baseline=True, ratio=2.0, columns=[], key_share=0.02)
def near_cross_split(v, opt):
    """Like duplicates.cross_split, after rounding numbers to `bits` bits of mantissa (10 is
    about 3 significant digits) and normalising text (case, punctuation, spacing). Catches
    copies that went through a lossy round trip, re-encoding or light editing. Exact copies
    are not counted again. Same group baseline as duplicates.cross_split. Columns acting as
    keys are left out, so a copy with a new id still matches: integers distinct on almost
    every row, and short text or entity-named integers with at least 50 values, distinct on
    at least `key_share` of rows. They stay when the other columns cannot tell rows apart."""
    return _cross(v, opt, near=True)


def _long_text(v, col: str, min_chars: int) -> bool:
    if v.kind(col) != "string":
        return False
    s = v.train[col].dropna().astype("string").head(500)
    return len(s) > 0 and float(s.str.len().mean()) >= min_chars


@check("duplicates.text_cross_split", "duplicates", "error",
       "No text appears in two splits", min_chars=30, max_fraction=0.0, baseline=True,
       ratio=2.0)
def text_cross_split(v, opt):
    """For each text feature (average length at least `min_chars`), texts that are identical
    after normalising case, punctuation and spacing, and at least `min_chars` long, must not
    occur in two splits. Compared column by column, so a copied document with a new id,
    date or other metadata is still found. Same group baseline as duplicates.cross_split."""
    cols = [c for c in v.features if _long_text(v, c, opt["min_chars"])]
    if not cols:
        return skipped(f"no text features (average length >= {opt['min_chars']} characters)")
    if len(v.splits) < 2:
        return skipped("one split only")
    found, allowed_by = {}, {}
    for col in cols:
        hs, long_ = {}, {}
        for n, d in v.splits.items():
            norm = d[col].astype("string").str.casefold() \
                .str.replace(r"[\W_]+", " ", regex=True).str.strip()
            long_[n] = (norm.str.len() >= opt["min_chars"]).fillna(False).to_numpy()
            hs[n] = v.hashes(n, (col,), True)
        base = None
        if opt["baseline"] and v.spec.group and all(v.has(c) for c in v.spec.group):
            df = pd.DataFrame({"h": hs[v.train_name], "g": v.hashes(v.train_name,
                                                                    tuple(v.spec.group))})
            df = df[long_[v.train_name]]
            k = df.groupby("h")["g"].nunique()
            base = float(df["h"].isin(k.index[k > 1]).mean()) if len(df) else 0.0
        allowed = max(opt["max_fraction"], opt["ratio"] * base if base is not None else 0.0)
        allowed_by[col] = frac(allowed, 1)
        for (a, _), (b, _) in combinations(v.splits.items(), 2):
            hit = np.isin(hs[b], hs[a][long_[a]]) & long_[b]
            n = int(hit.sum())
            if n and n / len(hit) > allowed:
                found[f"{col} {a}/{b}"] = {
                    "rows": n, "fraction": frac(n, len(hit)),
                    "examples": [str(x)[:80] for x in v.splits[b][col][hit].head(3)]}
    if found:
        return failed("texts shared between splits: " + ", ".join(
            f"{k} {x['rows']} rows ({x['fraction']:.2%})" for k, x in found.items()),
            "Deduplicate texts (after normalising case and punctuation) before splitting, or "
            "split by the group that produced them.", pairs=found, allowed=allowed_by)
    return passed(f"no shared texts in {len(cols)} text column(s)", allowed=allowed_by)


@check("duplicates.within_split", "duplicates", "warn",
       "Training data is not dominated by repeated rows", max_fraction=0.05, max_share=0.01)
def within_split(v, opt):
    """Inside the training split: the share of rows that repeat an earlier row must stay below
    `max_fraction`, and no single feature vector may make up more than `max_share` of the rows
    (minimum 10 copies). Mass-injected copies are a cheap poisoning attack; they also skew
    the class balance, so the per-label repeat rate is reported too."""
    feats = tuple(v.features)
    if not feats:
        return skipped("no features")
    d = v.train
    h = pd.Series(v.hashes(v.train_name, feats))
    dup = h.duplicated()
    counts = h.value_counts()
    top = int(counts.iloc[0]) if len(counts) else 0
    rate = float(dup.mean()) if len(h) else 0.0
    per_label = {}
    for lbl in v.labels:
        if lbl in d.columns and v.task(lbl) != "regression":
            g = dup.groupby(d[lbl].astype("string").fillna("<NA>").to_numpy()).mean()
            per_label[lbl] = {str(k): frac(x, 1) for k, x in g.items()}
    details = {"repeat_fraction": frac(rate, 1), "largest_copy_count": top,
               "rows": len(h), "repeat_fraction_by_label": per_label}
    problems = []
    if rate > opt["max_fraction"]:
        problems.append(f"{rate:.2%} of {v.train_name} rows repeat another row")
    if top >= 10 and top / max(len(h), 1) > opt["max_share"]:
        problems.append(f"one feature vector appears {top} times ({top / len(h):.2%})")
    if problems:
        return failed("; ".join(problems),
                      "Check where the copies come from (a join fan-out, oversampling before "
                      "the split, or injected rows). Oversample only inside training, after "
                      "the split.", **details)
    return passed(f"{rate:.2%} repeated rows, largest copy count {top}", **details)


@check("duplicates.conflicting_labels", "duplicates", "warn",
       "Identical training examples agree on their label", max_fraction=0.01)
def conflicting_labels(v, opt):
    """Among training rows with identical features, the share of rows whose label disagrees
    with the majority label of their copies. Above `max_fraction` this points at label noise
    or flipped labels (label-flipping poisoning), or at missing features."""
    feats = tuple(v.features)
    labels = [lb for lb in v.labels if lb in v.train.columns and v.task(lb) != "regression"]
    if not feats or not labels:
        return skipped("needs features and a classification label")
    out = {}
    h = v.hashes(v.train_name, feats)
    for lbl in labels:
        y = v.train[lbl].astype("string").fillna("<NA>").to_numpy()
        df = pd.DataFrame({"h": h, "y": y})
        sizes = df.groupby("h")["y"].transform("size")
        multi = df[sizes.to_numpy() > 1]
        if multi.empty:
            out[lbl] = 0
            continue
        cnt = multi.groupby(["h", "y"]).size()
        majority = cnt.groupby(level=0).max()
        total = cnt.groupby(level=0).sum()
        out[lbl] = int((total - majority).sum())
    n = len(h)
    worst = max(out.values()) / n if n else 0.0
    if worst > opt["max_fraction"]:
        return failed("identical examples with different labels: " + ", ".join(
            f"{k}: {x} rows ({x / n:.2%})" for k, x in out.items()),
            "Inspect these rows: flipped labels, a missing feature, or a target that is "
            "not a function of the features.", minority_rows=out, rows=n)
    return passed("rows whose identical copies carry another label: " + ", ".join(
        f"{k} {x} ({x / n:.2%})" for k, x in out.items())
        + f" <= {opt['max_fraction']:.0%}", minority_rows=out)
