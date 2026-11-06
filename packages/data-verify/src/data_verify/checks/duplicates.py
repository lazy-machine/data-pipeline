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
_SIGMA = 3.0  # standard deviations over the evaluation split's own repeats a match must clear
_MIN_ROWS = 10  # a handful of matches is not evidence, however small the split
_MIN_SHARE = 0.005  # a repeat warning needs a real share of rows, not just a count


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


def _rare_mask(v, d: pd.DataFrame, cols: tuple[str, ...], anchors: list[str],
               bits: int) -> np.ndarray:
    """Rows of ``d`` whose evidence is rare in train: at most _RARE_VALUE training rows share
    their value in an anchor column, or -- when no single column is distinctive enough -- their
    whole vector of ``cols`` without the keys. A common value recurring proves nothing, however
    the columns relate."""
    if anchors:
        rare = np.zeros(len(d), dtype=bool)
        for c in anchors:
            counts = pd.Series(v.hashes(v.train_name, (c,), True, bits)).value_counts()
            seen = pd.Series(row_hashes(d, [c], True, bits)).map(counts).fillna(0).to_numpy()
            rare |= seen <= _RARE_VALUE
        return rare
    counts = pd.Series(v.hashes(v.train_name, cols, True, bits)).value_counts()
    seen = pd.Series(row_hashes(d, list(cols), True, bits)).map(counts).fillna(0).to_numpy()
    return seen <= _RARE_VALUE


def _chance_share(v, cols: tuple[str, ...], anchors: list[str], bits: int) -> float:
    """How often rare rows match by chance: the share of rows, built by shuffling each training
    column on its own, that have a rare value and match a real training row. Only a floor under
    the null, because shuffling the columns apart forgets how they relate."""
    train = v.train
    rng = np.random.default_rng(v.spec.seed)
    mixed = pd.DataFrame({c: train[c].iloc[rng.permutation(len(train))].reset_index(drop=True)
                          for c in cols})
    match = np.isin(row_hashes(mixed, list(cols), True, bits),
                    v.hashes(v.train_name, cols, True, bits))
    return float((match & _rare_mask(v, mixed, cols, anchors, bits)).mean())


def _collision_share(v, cols: tuple[str, ...], anchors: list[str], bits: int) -> float:
    """Share of training rows that repeat another training row's rare vector. Reported, not
    used as the null: a copy added to train inflates it, which is what hides a leak."""
    h = pd.Series(v.hashes(v.train_name, cols, True, bits))
    seen = h.map(h.value_counts()).to_numpy()
    return float(((seen > 1) & _rare_mask(v, v.train, cols, anchors, bits)).mean())


def _pair_counts(v, a: str, b: str, cols: tuple[str, ...], rest: tuple[str, ...],
                 anchors: list[str], bits: int):
    """Rows of split ``b`` that match a row of ``a`` on ``rest``, and rows of ``b`` that match
    another row of ``b``: both rare, and the first excluding the exact all-column matches the
    comparison on all columns already reports."""
    hb = v.hashes(b, rest, True, bits)
    rare = _rare_mask(v, v.splits[b], rest, anchors, bits)
    cross = np.isin(hb, v.hashes(a, rest, True, bits)) & rare
    cross &= ~np.isin(v.hashes(b, cols), v.hashes(a, cols))
    ser = pd.Series(hb)
    selfb = (ser.map(ser.value_counts()).to_numpy() > 1) & rare
    return cross, selfb


def _self_count(v, split: str, rest: tuple[str, ...], anchors: list[str], bits: int,
                other: str | None = None) -> int:
    """Rows of ``split`` that repeat another row of the same split, rare and at ``bits``.
    With ``other``, rows whose vector also occurs there are left out: those are the cross
    matches, and a copy placed a few times in one split is exactly such a row, so leaving them
    in would let the copy raise the repeats it is measured against."""
    h = v.hashes(split, rest, True, bits)
    ser = pd.Series(h)
    rare = _rare_mask(v, v.splits[split], rest, anchors, bits)
    rep = (ser.map(ser.value_counts()).to_numpy() > 1) & rare
    if other is not None:
        rep &= ~np.isin(h, v.hashes(other, rest, True, bits))
    return int(rep.sum())


def _pair_stats(v, a: str, b: str, cols: tuple[str, ...], rest: tuple[str, ...],
                anchors: list[str], bits: int, chance: float, ratio: float,
                use_null: bool) -> dict:
    """One split pair at one precision. ``expected`` is the larger of what each side's own
    repeats predict for the matches between them, at the same per-pair rate: a copy added to
    ``a`` cannot move ``b``'s own rate and is left out of ``a``'s (it occurs in ``b``), and a
    dense ``b`` raises its own rate, so between them a post-split copy stays visible and a
    dense evaluation segment is explained. ``sigma`` carries both the count's variance and the
    variance of those repeat estimates."""
    n_a, n_b = len(v.splits[a]), len(v.splits[b])
    cross, selfb = _pair_counts(v, a, b, cols, rest, anchors, bits)
    c, s = int(cross.sum()), int(selfb.sum())
    sa = _self_count(v, a, rest, anchors, bits, other=b)
    scale = n_a / max(n_b - 1, 1)
    from_eval, from_earlier = s * scale, sa * n_b / max(n_a - 1, 1)
    expected = max(from_eval, from_earlier, ratio * chance * n_b) if use_null else 0.0
    var = expected + 1
    return {"rows": c, "self": s, "self_earlier": sa, "fraction": frac(c, n_b),
            "n_a": n_a, "n_b": n_b, "expected": expected, "sigma": math.sqrt(var),
            "example_rows_of_" + b: np.flatnonzero(cross)[:5].tolist()}


def _pair_verdicts(stats: dict, max_fraction: float, min_share: float):
    """(pairs that are more than the data explains, pairs that are only its own repeats) among
    the split pairs with at least _MIN_ROWS matches, over max_fraction and min_share."""
    more, repeats = [], []
    for key, st in stats.items():
        share = st["rows"] / st["n_b"] if st["n_b"] else 0.0
        if st["rows"] < _MIN_ROWS or share <= max(max_fraction, min_share):
            continue
        if st["rows"] > st["expected"] + _SIGMA * st["sigma"]:
            more.append(key)
        else:
            repeats.append(key)
    return more, repeats


def _second_look(v, opt, cols: tuple[str, ...]):
    """The comparison without the key columns: for each split pair the matches at the fine and
    (with `baseline`) the coarse precision, and what the evaluation split's own repeats predict
    for them. ``None`` when no column acts as a key; ``ran`` is false when nothing is left to
    compare once the keys are ignored."""
    if opt.get("columns"):
        return None
    return v.cached(("second_look", cols, opt["key_share"], bool(opt["baseline"]), opt["ratio"]),
                    lambda: _second_look_scan(v, opt, cols))


def _second_look_scan(v, opt, cols: tuple[str, ...]):
    keys = [c for c in cols if _is_key(v, c, opt["key_share"])]
    if not keys:
        return None
    rest = tuple(c for c in cols if c not in keys)
    fine = _fine_bits(len(v.train))
    look = {"keys": keys, "rest": rest, "anchors": [], "ran": bool(rest), "bits": fine,
            "pairs": {}, "coarse": {}, "stats": {}, "cstats": {}, "chance": 0.0,
            "cochance": 0.0, "repeated_in_train": 0.0}
    if not rest:
        return look
    anchors = _anchors(v, rest)
    look["anchors"] = anchors
    base = bool(opt["baseline"])
    if base:
        look["repeated_in_train"] = _collision_share(v, rest, anchors, fine)
        look["chance"] = _chance_share(v, rest, anchors, fine)
        look["cochance"] = _chance_share(v, rest, anchors, _COARSE_BITS)
    for (a, _), (b, _) in combinations(v.splits.items(), 2):
        name = f"{a}/{b}"
        look["stats"][name] = _pair_stats(v, a, b, cols, rest, anchors, fine,
                                           look["chance"], opt["ratio"], base)
        if look["stats"][name]["rows"]:
            look["pairs"][name] = look["stats"][name]
        if base:
            look["cstats"][name] = _pair_stats(v, a, b, cols, rest, anchors, _COARSE_BITS,
                                                look["cochance"], opt["ratio"], True)
            if look["cstats"][name]["rows"]:
                look["coarse"][name] = look["cstats"][name]
    return look


def _shared(stats: dict, names: list[str], keys: list[str], kind: str,
            coarse: bool = False) -> str:
    """The message for matches that are more than the evaluation split's own repeats explain."""
    bits = f", within {_COARSE_BITS} bits of mantissa" if coarse else ""
    text = ", ".join(f"{k} {stats[k]['rows']} rows ({stats[k]['fraction']:.2%})" for k in names)
    own = max(stats[k]["expected"] / stats[k]["n_b"] for k in names)
    return (f"{kind} feature rows shared between splits: {text} (ignoring key column(s) "
            f"{', '.join(keys)}{bits}); the splits' own repeats predict only {own:.3%} of those "
            f"rows match by chance, so this is more than that")


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
            hit &= _rare_mask(v, v.splits[b], cols, anchors, bits)
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
    if near:
        # a second look, not a replacement: a copy under a new key doesn't match on all columns
        look = _second_look(v, opt, cols)
        if look:
            details["ignored_keys"] = look["keys"]
        if look and not look["ran"]:
            return skipped("nothing is left to compare once the undeclared key column(s) "
                           f"{', '.join(look['keys'])} are ignored; declare params.id so the "
                           "key is a role")
        if look:
            details.update(pairs_without_keys=look["pairs"],
                           chance_without_keys=frac(look["chance"], 1))
            more, _ = _pair_verdicts(look["stats"], opt["max_fraction"], 0.0)
            if more:
                return failed(_shared(look["stats"], more, look["keys"], kind), hint, **details)
            if opt["baseline"]:
                # A copy jittered past `bits` still agrees to about 4 significant digits, and
                # the same null applies: the evaluation split's own repeats, which a copy into
                # the earlier split cannot move.
                details.update(pairs_at_coarse_bits=look["coarse"],
                               chance_at_coarse_bits=frac(look["cochance"], 1))
                more, _ = _pair_verdicts(look["cstats"], opt["max_fraction"], 0.0)
                if more:
                    return failed(_shared(look["cstats"], more, look["keys"], kind,
                                          coarse=True), hint, **details)
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
    by chance, so the second comparison asks for more evidence: a value at most 5 training
    rows have in a column where at least half the values are distinct, or -- when no single
    column is distinctive enough -- the whole row without the keys; numbers equal to
    ceil(log2(train rows)) + 7 bits of mantissa, between 20 and 48 (about 6 significant digits
    at 15,000 training rows and 8 at 150,000); at least 10 matched rows; and at least 3
    standard deviations more matches than the two splits' own repeats predict. That last bar
    is the load-bearing one. The allowance is the larger of what each side's own repeat rate
    predicts for the pair, at the same per-pair rate over the other split's size. A copy added
    to the earlier split occurs in the evaluation split, so it neither moves the evaluation
    split's rate nor counts in the earlier split's own repeats: it cannot hide itself, however
    many times it is placed. An evaluation split that sits where the features are dense raises
    the allowance and is explained. The shuffled-column share is only a floor under it, because
    shuffling the columns apart forgets how they relate. With `baseline = false` there is no
    allowance and the coarse comparison is skipped. A copy jittered past the fine precision
    still agrees to
    about 4 significant digits, so a further comparison at 14 bits needs the same, with the
    same null. Between them the comparisons find exact copies and jitter up to about 1e-5
    relative when at least 10 rows match; a handful of copies is below the floor, whatever the
    jitter. Rows that match no more often than the splits' own repeats are not reported here:
    duplicates.repeated_across_splits warns about them, because a copy made before the split
    repeats inside the evaluation split too, and coarse columns that repeat by nature look the
    same. So under a new key a copy must almost carry the same numbers, unless the keys are
    declared (`params.id` and `params.group`), which leaves only the comparison at `bits`. It
    doesn't find a copy made before the split (the other check warns), a row copied more than
    5 times in the earlier split (its value stops being rare), a copy of a common value, a copy
    given a fresh timestamp or other per-row value, or copies in data with only coarse or
    low-entropy columns whose rows still repeat, unless `columns` leaves that column out. When
    nothing is left to compare once the keys are ignored, the check skips with a message
    telling you to declare `params.id`."""
    return _cross(v, opt, near=True)


@check("duplicates.repeated_across_splits", "duplicates", "warn",
       "Rows that repeat across splits are not called copies", max_fraction=0.0,
       baseline=True, ratio=2.0, columns=[], key_share=0.02)
def repeated_across_splits(v, opt):
    """The other half of the comparison without key columns: rows that match across splits no
    more often than the two splits' own repeats predict. A copy made before the split repeats
    inside the evaluation split too, and coarse columns repeat by nature, so this warns and
    never claims copies. It uses the same key columns, `key_share`, fine precision, rarity
    rule and null as duplicates.near_cross_split; `baseline = false` lifts the comparison and
    leaves it to that check, and `columns` is compared exactly, not second-guessed. A warning
    needs at least 10 matched rows and more than `max_fraction` and 0.5% of the evaluation
    split's rows. The report lists the split pairs and the share the splits' own repeats
    predict."""
    feats = tuple(v.features)
    if not feats:
        return skipped("no features")
    if len(v.splits) < 2:
        return skipped("one split only")
    if opt.get("columns"):
        return skipped("`columns` is compared exactly, not second-guessed")
    if not opt["baseline"]:
        return skipped("baseline is off: duplicates.near_cross_split runs alone")
    look = _second_look(v, opt, feats)
    if not look or not look["ran"]:
        return skipped("no key column, or nothing left to compare once the keys are ignored")
    _, repeats = _pair_verdicts(look["stats"], opt["max_fraction"], _MIN_SHARE)
    details = {"pairs": look["pairs"], "ignored_keys": look["keys"],
               "chance": frac(look["chance"], 1),
               "repeated_in_train": frac(look["repeated_in_train"], 1)}
    if repeats:
        text = ", ".join(f"{k} {look['stats'][k]['rows']} rows "
                         f"({look['stats'][k]['fraction']:.2%})" for k in repeats)
        own = max(look["stats"][k]["expected"] / look["stats"][k]["n_b"] for k in repeats)
        return failed(f"feature rows repeat across and within splits: {text} (ignoring key "
                      f"column(s) {', '.join(look['keys'])}); the splits' own repeats predict "
                      f"{own:.3%} of those rows match by chance, so this is within the data's "
                      f"own repeats and not more than they explain: it can be rows copied "
                      f"before the split or coarse columns that repeat by nature",

                      "If these rows were copied before the split, deduplicate before "
                      "splitting. If the repeats are legitimate (coarse columns), declare "
                      "params.id and params.group, or raise "
                      "checks.\"duplicates.repeated_across_splits\".max_fraction.", **details)
    return passed("repeated rows across splits stay within the evaluation split's own repeats",
                  **details)


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
