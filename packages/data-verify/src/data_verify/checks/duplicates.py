"""Duplicates: the same example in two splits (memorised, not generalised), and suspicious
repetition or label conflicts inside the training data (injected copies, flipped labels)."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from ..core import check, failed, frac, passed, row_hashes, skipped
from .splits import _ENTITY_NAME

# Without keys, a match needs more evidence than a match on every column does.
_RARE_VALUE = 5  # training rows that may share a value for a match on it to mean something
_MARGIN = 7  # bits of mantissa above log2(rows): the share matching a number by chance is about
# rows * 2**-bits however the columns relate, so following the rows keeps it near 2**-MARGIN
# (about 0.15%, a third of _MIN_SHARE: a tighter margin false-alarms on dependent columns)
_MIN_BITS, _MAX_BITS = 20, 48  # a small train is not compared too coarsely; float64 has 52 bits
_COARSE_BITS = 14  # about 4 significant digits: a copy jittered past _fine_bits still matches
_COARSE_ROWS, _COARSE_SIGMA = 10, 3.0  # a copy must beat the train's own collisions by both
_MIN_SHARE = 0.005  # coarse columns that depend on each other still meet now and then


def _is_key(v, col: str, key_share: float) -> bool:
    s = v.train[col].dropna()
    if s.empty:
        return False
    kind = v.kind(col)
    try:
        values = s.nunique()
    except TypeError:  # lists and other unhashable cells
        return False
    share = values / len(s)
    if kind == "string":
        text = s.astype("string")
        word = float(text.str.contains(r"\s").mean()) < 0.05   # a hash, not a sentence
        return values >= 50 and share >= key_share and \
            (word or float(text.str.len().mean()) <= 40)
    # a null turns an integer column into floats
    whole = kind == "int" or (kind == "float" and bool((s == s.round()).all()))
    return whole and ((values >= 20 and share >= 0.98) or
                      (values >= 50 and share >= key_share and bool(_ENTITY_NAME.search(col))))


def _fine_bits(n: int) -> int:
    """Bits of mantissa for the second look. The share of rows whose numbers match by chance
    is about n * 2**-bits however the columns relate, so it follows the train size and stays
    near 2**-_MARGIN: coarser would lose the jitter tolerance a small train can afford, finer
    would let columns that depend on each other meet by themselves on a large one."""
    return min(max(int(np.ceil(np.log2(max(n, 1)))) + _MARGIN, _MIN_BITS), _MAX_BITS)


def _anchors(v, cols: tuple[str, ...]) -> list[str]:
    """Columns with enough distinct values for a match on one to mean something."""
    found = []
    for c in cols:
        s = v.train[c].dropna()
        try:
            if len(s) and s.nunique() / len(s) >= 0.5:
                found.append(c)
        except TypeError:   # lists and other unhashable cells
            continue
    return found


def _rare_rows(v, d: pd.DataFrame, anchors: list[str], bits: int) -> np.ndarray:
    """Rows of ``d`` holding, in an anchor column, a value at most _RARE_VALUE training rows
    have. A common value recurring proves nothing, however the columns relate."""
    rare = np.zeros(len(d), dtype=bool)
    for c in anchors:
        counts = pd.Series(v.hashes(v.train_name, (c,), True, bits)).value_counts()
        seen = pd.Series(row_hashes(d, [c], True, bits)).map(counts).fillna(0).to_numpy()
        rare |= seen <= _RARE_VALUE
    return rare


def _chance_share(v, cols: tuple[str, ...], anchors: list[str], bits: int) -> float:
    """How often rare rows match by chance: the share of rows, built by shuffling each training
    column on its own, that have a rare value and match a real training row. Counting repeats
    in train would not do, because copies inflate that, and they are what this has to find."""
    train = v.train
    rng = np.random.default_rng(v.spec.seed)
    mixed = pd.DataFrame({c: train[c].iloc[rng.permutation(len(train))].reset_index(drop=True)
                          for c in cols})
    match = np.isin(row_hashes(mixed, list(cols), True, bits),
                    v.hashes(v.train_name, cols, True, bits))
    return float((match & _rare_rows(v, mixed, anchors, bits)).mean())


def _collision_share(v, cols: tuple[str, ...], anchors: list[str], bits: int) -> float:
    """How often rare training rows match each other at ``bits``. Real rows, so the count keeps
    every relation between the columns, which shuffling them apart would forget: columns that
    depend on each other collide here as often as they really do. A copy repeated inside train
    raises it, which only makes the coarse comparison harder to fire."""
    h = pd.Series(v.hashes(v.train_name, cols, True, bits))
    seen = h.map(h.value_counts()).to_numpy()
    return float(((seen > 1) & _rare_rows(v, v.train, anchors, bits)).mean())


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


def _matches(v, cols, all_cols, near: bool, bits: int, anchors: list[str] | None = None):
    """Rows of each later split that match an earlier split on ``cols``, and the worst share."""
    pairs, worst = {}, 0.0
    for (a, _), (b, _) in combinations(v.splits.items(), 2):
        ha, hb = v.hashes(a, cols, near, bits), v.hashes(b, cols, near, bits)
        hit = np.isin(hb, ha)
        if near:  # report only what the exact check does not already see
            hit &= ~np.isin(v.hashes(b, all_cols), v.hashes(a, all_cols))
        if anchors:
            hit &= _rare_rows(v, v.splits[b], anchors, bits)
        n = int(hit.sum())
        if n:
            worst = max(worst, n / len(hb))
            pairs[f"{a}/{b}"] = {"rows": n, "fraction": frac(n, len(hb)),
                                 "example_rows_of_" + b: np.flatnonzero(hit)[:5].tolist()}
    return pairs, worst


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
    pairs, worst = _matches(v, cols, cols, near, bits)
    kind = "near-duplicate" if near else "duplicate"
    base = _cross_group_rate(v, cols, near, bits) if opt["baseline"] else None
    allowed = opt["max_fraction"]
    if base is not None:
        allowed = max(allowed, opt["ratio"] * base)
    details = {"pairs": pairs, "compared_columns": len(cols), "max_fraction": opt["max_fraction"]}
    if base is not None:
        details.update(cross_group_rate_in_train=frac(base, 1), allowed=frac(allowed, 1))
    summary = ", ".join(f"{k} {p['rows']} rows ({p['fraction']:.2%})" for k, p in pairs.items())
    name = "near_cross_split" if near else "cross_split"
    hint = ("The model sees these evaluation examples during training. Deduplicate before "
            "splitting, or split by a group key that keeps copies together (params.group). If "
            "identical rows are legitimate (few low-cardinality features), raise "
            f"checks.\"duplicates.{name}\".max_fraction.")
    if pairs and worst > allowed:
        why = (f"; different entities share a vector in only {base:.3%} of {v.train_name} rows, "
               f"so this is more than chance" if base is not None else "")
        return failed(f"{kind} feature rows shared between splits: {summary}{why}", hint, **details)
    if near and not opt.get("columns"):
        # a second look, not a replacement: a copy under a new key doesn't match on all columns
        keys = [c for c in cols if _is_key(v, c, opt["key_share"])]
        rest = tuple(c for c in cols if c not in keys)
        anchors = _anchors(v, rest)
        if keys:
            details["ignored_keys"] = keys
        if keys and anchors:
            fine = _fine_bits(len(v.train))
            found, share = _matches(v, rest, cols, near, fine, anchors)
            chance = _chance_share(v, rest, anchors, fine) if opt["baseline"] else 0.0
            limit = max(opt["max_fraction"], opt["ratio"] * chance,
                        _MIN_SHARE if opt["baseline"] else 0.0)
            details.update(pairs_without_keys=found, chance_without_keys=frac(chance, 1))
            if found and share > limit:
                text = ", ".join(f"{k} {p['rows']} rows ({p['fraction']:.2%})"
                                 for k, p in found.items())
                return failed(f"{kind} feature rows shared between splits: {text} (ignoring key "
                              f"column(s) {', '.join(keys)}); {chance:.3%} would match by chance",
                              hint, **details)
            if opt["baseline"]:
                # A copy jittered past `fine` still agrees to about 4 significant digits. At
                # that precision columns that depend on each other collide by themselves, so
                # this asks whether the test rows collide with train more often than the train
                # collides with itself: same columns, same relations, no copy alignment.
                found, share = _matches(v, rest, cols, near, _COARSE_BITS, anchors)
                chance = _collision_share(v, rest, anchors, _COARSE_BITS)
                limit = max(opt["max_fraction"], opt["ratio"] * chance, _MIN_SHARE)
                hits = max((p["rows"] for p in found.values()), default=0)
                expected = chance * max(len(d) for d in v.splits.values())
                details.update(pairs_at_coarse_bits=found, chance_at_coarse_bits=frac(chance, 1))
                if hits >= _COARSE_ROWS and share > limit \
                        and hits > expected + _COARSE_SIGMA * math.sqrt(expected + 1):
                    text = ", ".join(f"{k} {p['rows']} rows ({p['fraction']:.2%})"
                                     for k, p in found.items())
                    return failed(
                        f"{kind} feature rows shared between splits: {text} (ignoring key "
                        f"column(s) {', '.join(keys)}, within {_COARSE_BITS} bits of mantissa); "
                        f"{chance:.3%} of {v.train_name} rows match each other",
                        hint, **details)
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
    are not counted again. Same group baseline as duplicates.cross_split. It compares all
    columns, then again without the columns acting as keys, so a copy with a new id still
    matches: whole numbers distinct on almost every row, and short text, single tokens such
    as hashes, or entity-named whole numbers with at least 50 values, distinct on at least
    `key_share` of rows. The report lists them in `ignored_keys`. Without them rows can match
    by chance, so this second comparison asks for more: numbers equal to ceil(log2(train
    rows)) + 7 bits of mantissa, at least 20 and at most 48 (about 6 significant digits at
    15,000 training rows, 8 at 150,000), and a value at most 5 training rows have in a column
    where at least half the values are distinct. Following the train size keeps the share of
    rows that match by chance near 0.15% however the columns relate, so the
    tolerance tightens as the train grows: 1e-7 relative jitter is still found at 15,000
    training rows and no longer at 150,000. With `baseline` on it allows 0.5% of rows, or
    `ratio` times the share that matches when each training column is shuffled on its own.
    A copy jittered past that still agrees to about 4 significant digits, so a second
    comparison at 14 bits asks whether the test rows collide with train more often than the
    training rows collide with each other: real rows, so that count keeps every relation
    between the columns and stays high when they depend on each other, while copies add
    collisions the train does not have on its own. It needs at least 10 matched rows and 3
    standard deviations over the train's own rate, so a handful of coincidences on a small
    split stays quiet. It catches jitter up to about 1e-5 relative on data with several
    columns, and finds nothing extra when they all derive from one (`b = a`), where the train
    collides with itself more often than a few copies appear.
    So under a new key a copy must almost carry the same numbers, unless the keys are
    declared (`params.id` and `params.group`), which leaves only the comparison at `bits`.
    It doesn't find copies in data with only coarse columns, such as values rounded to 4
    decimals with another column derived from them, copies of common values, a row copied
    more than 5 times, or a copy given a fresh timestamp or other per-row value, unless
    `columns` leaves that column out; and a copy repeated inside train raises the collision
    count it is measured against, so pre-split duplication is left to
    duplicates.within_split."""
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
