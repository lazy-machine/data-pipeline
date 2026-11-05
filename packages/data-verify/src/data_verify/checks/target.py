"""Target leakage (features that carry the answer) and preprocessing leakage (transforms
fitted on data the model will be scored on)."""

from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd

from ..core import (
    REGISTRY, check, examples, failed, fmt, is_numeric_kind, kind, matches, numeric_values, passed,
    resolve_check, skipped, value_text, value_texts,
)
from ..stats import feature_scores


@check("target.forbidden_features", "target", "error",
       "No label, key, time or forbidden column is used as a feature")
def forbidden_features(v, opt):
    """The features must not include any role column (label, id, group, time, label_time,
    weight, split column, point_in_time columns) nor anything matching params.forbidden (glob
    patterns, e.g. "*_after_*", "outcome_*"). With features = "auto" every non-role column is
    a feature, so a leaky column left in the table fails here."""
    s = v.spec
    roles = set(s.role_columns())
    hits = {f: "role column" for f in v.declared_features if f in roles}
    hits.update({f: "matches params.forbidden" for f in v.declared_features
                 if f not in hits and matches(f, s.forbidden)})
    if hits:
        return failed(f"{len(hits)} forbidden column(s) among the features: "
                      + ", ".join(f"{k} ({x})" for k, x in hits.items()),
                      "Remove them from the feature list (params.features / params.exclude) and "
                      "from the training code.", columns=hits)
    return passed(f"{len(v.declared_features)} features, none forbidden"
                  + ("" if v.features_declared else " (features = auto: every non-role column)"))


_SUSPICIOUS = (r"(^|_)(target|label|labels|y|outcome|result|answer|truth|ground_?truth|"
               r"leak|future|next|post|after|actual|response|class|prediction|pred|score|"
               r"is_?fraud|churn(ed)?|default(ed)?|died|death|survived|failure|failed)(_|$)")


@check("target.suspicious_names", "target", "warn", "No feature name hints at the answer",
       pattern=_SUSPICIOUS)
def suspicious_names(v, opt):
    """Feature names that read like an outcome or a look-ahead (target, label, outcome, future,
    next_, post_, after, result, score, prediction, ...). A name is only a hint, so this warns;
    confirm the column is known at prediction time, then list it in
    checks."target.suspicious_names".pattern or exclude it."""
    rx = re.compile(opt["pattern"], re.IGNORECASE)
    hits = [f for f in v.features if rx.search(re.sub(r"(?<=[a-z])(?=[A-Z])", "_", f))]
    if hits:
        return failed(f"{len(hits)} feature name(s) look like outcomes: {hits[:10]}",
                      "Confirm each is available before the prediction time.", features=hits)
    return passed(f"{len(v.features)} feature names look neutral")


@check("target.label_copy", "target", "error", "No feature is a copy of the label",
       min_agreement=0.99)
def label_copy(v, opt):
    """A feature equal to the label on at least `min_agreement` (default 99%) of the training
    rows, or (for numbers) an exact linear transform of it (|r| > 0.99999). Leaky columns
    are often noisy copies (a code entered after the outcome that matches it almost always)."""
    if not v.labels:
        return skipped("no params.label")
    d = v.sample(v.train_name)
    hits = {}
    for lbl in v.labels:
        if lbl not in d.columns:
            continue
        y = d[lbl]
        y_num = kind(y) in ("int", "float", "bool")
        yn = numeric_values(y.astype("float64") if kind(y) == "bool" else y) if y_num else None
        ys = None
        for f in v.features:
            x = d[f]
            x_num = kind(x) in ("int", "float", "bool")
            if y_num and x_num:
                xn = numeric_values(x.astype("float64") if kind(x) == "bool" else x)
                ok = ~np.isnan(xn) & ~np.isnan(yn)
                if ok.sum() < 10:
                    continue
                agree = float((xn[ok] == yn[ok]).mean())
                if agree < opt["min_agreement"] and xn[ok].std() > 0 and yn[ok].std() > 0:
                    r = np.corrcoef(xn[ok], yn[ok])[0, 1]
                    if abs(r) > 0.99999:
                        hits[f"{f}~{lbl}"] = f"linear transform of the label (r={r:.6f})"
                        continue
            else:
                both = (x.notna() & y.notna()).to_numpy()
                if both.sum() < 10:
                    continue
                ys = y.astype("string") if ys is None else ys
                agree = float((x[both].astype("string").to_numpy()
                               == ys[both].to_numpy()).mean())
            if agree >= opt["min_agreement"]:
                hits[f"{f}~{lbl}"] = f"equal on {agree:.2%} of rows"
    if hits:
        return failed(f"{len(hits)} feature(s) copy the label: "
                      + ", ".join(f"{k}: {x}" for k, x in hits.items()),
                      "Drop the column: it is the answer, not a feature.", pairs=hits)
    return passed(f"no feature copies {', '.join(v.labels)}")


def _scores(v, split: str, label: str) -> dict[str, float]:
    return v.cached(("scores", split, label), lambda: feature_scores(
        v.sample(split), v.features, label, v.task(label), v.spec.seed))


def _predictor(v, opt, strength: str):
    if not v.labels:
        return skipped("no params.label")
    if not v.features:
        return skipped("no features")
    hits, top = {}, {}
    for lbl in v.labels:
        if lbl not in v.train.columns:
            continue
        sc = _scores(v, v.train_name, lbl)
        ranked = sorted(((k, x) for k, x in sc.items() if not np.isnan(x)), key=lambda t: -t[1])
        top[lbl] = {k: fmt(x) for k, x in ranked[:5]}
        for k, x in ranked:
            if x >= opt["threshold"]:
                hits[f"{k}->{lbl}"] = fmt(x)
    metric = {"binary": "AUC", "multiclass": "normalised accuracy", "regression": "|Spearman|"}
    if hits:
        kinds = ", ".join(sorted({metric[v.task(lb)] for lb in v.labels if lb in v.train.columns}))
        return failed(f"{len(hits)} feature(s) alone predict the label ({kinds} >= "
                      f"{opt['threshold']}): " + ", ".join(f"{k}={x}" for k, x in
                                                           list(hits.items())[:8]),
                      "A single column that (almost) determines the label is usually computed "
                      "from the outcome or recorded after it. isnull(x) means the column's "
                      "missingness gives the label away. If the relationship is real and known "
                      "at prediction time, raise the threshold for this check.",
                      features=hits, top=top, threshold=opt["threshold"])
    def best(lbl, t):
        if not t:
            return f"{lbl}: no measurable feature"
        k, x = next(iter(t.items()))
        return f"{lbl}: {k} {metric[v.task(lbl)]} {x}"
    return passed(f"no feature reaches {opt['threshold']}; strongest: "
                  + "; ".join(best(lbl, t) for lbl, t in top.items()), top=top)


@check("target.perfect_predictor", "target", "error",
       "No single feature (or its missingness) determines the label", threshold=0.99)
def perfect_predictor(v, opt):
    """Scores each feature, and the missingness of each partly-null feature, as a standalone
    predictor of each label on the training split (AUC for binary, normalised accuracy for
    multiclass, |Spearman| for regression; cross-fitted, so ids score near chance). A score
    at or above `threshold` (default 0.99) means the label can be read off one column: real
    signals almost never do that alone, leaks (codes entered after the outcome, fields filled
    only for one class) routinely do."""
    return _predictor(v, opt, "perfect")


@check("target.strong_predictor", "target", "warn",
       "No single feature is suspiciously predictive", threshold=0.95)
def strong_predictor(v, opt):
    """Same scores as target.perfect_predictor with a lower bar (default 0.95). Real signals
    rarely reach this alone; look at how the flagged columns are computed. Set its severity
    to "error" for a stricter gate."""
    return _predictor(v, opt, "strong")


@check("target.score_gap", "target", "warn",
       "Features predict the label equally well in every split", max_gap=0.2, min_score=0.8)
def score_gap(v, opt):
    """Compares each feature's single-feature score in the training split with the same score in
    each evaluation split. A feature far more predictive in test than in train (gap above
    `max_gap` with the higher score at least `min_score`) was likely built with the test labels
    (target encoding or imputation fitted on all data); far more predictive in train only
    points at in-sample target encoding."""
    if not v.labels or len(v.splits) < 2:
        return skipped("needs a label and at least two splits")
    hits = {}
    for lbl in v.labels:
        if lbl not in v.train.columns:
            continue
        tr = _scores(v, v.train_name, lbl)
        for name, d in v.others():
            if lbl not in d.columns or d[lbl].notna().sum() < 30:
                continue
            ot = _scores(v, name, lbl)
            for f, a in tr.items():
                b = ot.get(f, np.nan)
                if np.isnan(a) or np.isnan(b):
                    continue
                if abs(a - b) > opt["max_gap"] and max(a, b) >= opt["min_score"]:
                    hits[f"{f}->{lbl} {v.train_name}/{name}"] = [fmt(a), fmt(b)]
    if hits:
        return failed(f"{len(hits)} feature(s) predict the label very differently across "
                      "splits: " + ", ".join(f"{k} {x[0]} vs {x[1]}" for k, x in
                                             list(hits.items())[:6]),
                      "Fit encoders and imputers on the training split only, out of fold for "
                      "target encoding.", features=hits)
    return passed("single-feature scores agree across splits")


_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|key|index|idx|rowid|row_?num(ber)?)$", re.I)


@check("target.id_like", "target", "warn", "No feature looks like a row identifier",
       min_unique=0.98)
def id_like(v, opt):
    """Features that behave like identifiers: row numbers ("Unnamed: 0", "index"), keys named
    *_id / uuid / key with a distinct value on (almost) every row, distinct text on every row,
    integers that count up through the rows (a dense sequence), or values that increase with
    the row order. They let a model memorise rows and often encode collection order.
    Cumulative counters and measurements with many distinct values are not flagged."""
    d = v.sample(v.train_name)
    hits = {}
    for f in v.features:
        x = d[f]
        k = kind(x)
        nn = x.dropna()
        if len(nn) < 50:
            continue
        uniq = nn.nunique() / len(nn)
        if f.lower() in ("index", "level_0") or f.startswith("Unnamed:"):
            hits[f] = "index-like name"
            continue
        if uniq < opt["min_unique"]:
            continue
        if k == "string" and not _short_text(nn):
            continue  # free text is unique per row by nature
        if k == "string" or _ID_NAME.search(f):
            hits[f] = f"{uniq:.1%} distinct values"
        elif k in ("int", "float"):
            vals = nn.to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals)]
            if len(vals) < 50:
                continue
            integral = bool(np.all(np.mod(vals, 1) == 0))
            span = vals.max() - vals.min() + 1
            if integral and span <= 2 * nn.nunique():
                hits[f] = f"dense integer sequence ({uniq:.1%} distinct)"
            else:
                xs = v.train[f].to_numpy(dtype="float64", na_value=np.nan)
                xs = xs[np.isfinite(xs)]
                if len(xs) > 100 and (np.all(np.diff(xs) >= 0) or np.all(np.diff(xs) <= 0)):
                    hits[f] = "monotonic in row order"
    if hits:
        return failed(f"{len(hits)} identifier-like feature(s): "
                      + ", ".join(f"{k} ({x})" for k, x in hits.items()),
                      "Drop row numbers and keys from the features (write CSVs with "
                      "index=False). Use group keys for splitting, not for learning.",
                      features=hits)
    return passed("no identifier-like features")


def _short_text(s: pd.Series) -> bool:
    return float(s.astype("string").str.len().mean()) <= 40


# -- preprocessing leakage -----------------------------------------------------------------

_CHANCE_MARGIN = 0.003  # fraction of the sampling noise a match must beat to count as scaled
_COARSEST_STEP = 0.01  # beyond this, rounding hides as much as sampling noise does


def _rounding_step(x: np.ndarray) -> float:
    """The decimal grid values sit on, or 0: a CSV keeps only so many digits. Read from the
    first values, which is enough for a column written one way."""
    sample = x[np.isfinite(x)][:1000]
    if len(sample):
        for decimals in range(7):
            scaled = sample * 10.0 ** decimals
            if np.abs(scaled - np.rint(scaled)).max() < 1e-4:
                return 10.0 ** -decimals
    return 0.0


def _moments(x: np.ndarray):
    x = x[~np.isnan(x)]
    return len(x), float(x.sum()), float((x ** 2).sum()), \
        (float(x.min()) if len(x) else np.nan), (float(x.max()) if len(x) else np.nan), \
        _rounding_step(x)


def _standardized(n, s, ss, tol_mean, tol_std, margin=None, step=0.0):
    """Mean 0 and std 1 within tolerance. With ``margin``, also too exactly to be chance.

    A scaler leaves only rounding error. Unscaled data that is roughly standard already
    misses by its sampling noise, about 1/sqrt(n) for the mean and 1/sqrt(2n) for the std,
    which falls inside any fixed tolerance once there are enough rows. So the tolerance
    shrinks with n, down to what rounding to ``step`` can explain. On a coarser grid than
    _COARSEST_STEP rounding can explain a chance match too, so no allowance is made.
    """
    if n < 3:
        return False
    if margin is not None:
        step = step if step <= _COARSEST_STEP else 0.0
        rounding = 4 * step / np.sqrt(12 * n)
        tol_mean = min(tol_mean, margin / np.sqrt(n) + rounding)
        tol_std = min(tol_std, margin / np.sqrt(2 * n) + rounding + step ** 2 / 24)
    mean = s / n
    var0 = max(ss / n - mean ** 2, 0.0)
    var1 = var0 * n / (n - 1)
    return abs(mean) <= tol_mean and (abs(np.sqrt(var0) - 1) <= tol_std or
                                      abs(np.sqrt(var1) - 1) <= tol_std)


def _minmax_shape(v, f, union) -> bool:
    """A min-max transform leaves continuous values with only a few rows at exactly 0 and 1;
    rates and proportions that naturally span [0, 1] pile up at the ends."""
    x = np.concatenate([numeric_values(v.splits[n][f]) for n in union.split("+")])
    x = x[~np.isnan(x)]
    if len(np.unique(x)) <= 2 or np.all(np.mod(x, 1) == 0):
        return False
    cap = max(5, 0.001 * len(x))
    return (x == 0).sum() <= cap and (x == 1).sum() <= cap


def _more_than_two_values(v, f: str) -> bool:
    """Two values can balance to mean 0 and std 1 by themselves: a +1/-1 flag is no evidence."""
    x = np.concatenate([numeric_values(d[f]) for d in v.splits.values()])
    return len(np.unique(x[~np.isnan(x)])) > 2


def _scaling_hits(v, tol_mean, tol_std, margin, minmax: bool) -> dict[str, str]:
    """Features that look scaled over train plus evaluation rows, but not over train alone."""
    names = list(v.splits)
    hits: dict[str, str] = {}
    feats = [f for f in v.numeric_features() if _more_than_two_values(v, f)]
    # row subsets to test: all rows; without repeats; without repeats and rows absent from
    # params.source. Rows injected after scaling would otherwise hide the signature.
    unique = {n: ~pd.Series(v.hashes(n, tuple(v.features))).duplicated().to_numpy()
              for n in names}
    variants = [None, unique]
    if all(v.in_source(n) is not None for n in names):
        variants.append({n: unique[n] & v.in_source(n) for n in names})
    for f, keep in ((f, k) for f in feats for k in variants):
        if f in hits:
            continue
        m = {n: _moments(numeric_values(v.splits[n][f]) if keep is None else
                         numeric_values(v.splits[n][f])[keep[n]]) for n in names}
        tr = m[v.train_name]
        if tr[0] < 3:
            continue
        tr_std = _standardized(*tr[:3], tol_mean, tol_std, margin, tr[5])
        tr_unit = tr[3] == 0 and tr[4] == 1
        others = [n for n in names if n != v.train_name]
        for r in range(1, len(others) + 1):
            for combo in combinations(others, r):
                parts = [tr] + [m[c] for c in combo]
                n_, s_, ss_ = (sum(p[i] for p in parts) for i in range(3))
                lo = np.nanmin([p[3] for p in parts])
                hi = np.nanmax([p[4] for p in parts])
                step = max(p[5] for p in parts)
                union = "+".join([v.train_name, *combo])
                if not tr_std and _standardized(n_, s_, ss_, tol_mean, tol_std, margin, step):
                    hits[f] = f"standardised over {union}, not over {v.train_name}"
                elif minmax and not tr_unit and lo == 0 and hi == 1 and \
                        _minmax_shape(v, f, union):
                    hits[f] = f"min-max scaled over {union}, not over {v.train_name}"
                if f in hits:
                    break
            if f in hits:
                break
    return hits


def _blocking_hits(v, tol_mean, tol_std, margin) -> dict[str, str]:
    return v.cached(("global_scaling", tol_mean, tol_std, margin),
                    lambda: _scaling_hits(v, tol_mean, tol_std, margin, minmax=True))


@check("preprocessing.global_scaling", "preprocessing", "error",
       "Scalers were fitted on the training split only", tol_mean=1e-4, tol_std=1e-3,
       chance_margin=_CHANCE_MARGIN)
def global_scaling(v, opt):
    """Detects features standardised (mean 0, std 1) or min-max scaled (exactly 0..1) over
    train plus evaluation rows together while the training rows alone are not: the scaler was
    fitted on data the model is later scored on. Every union of splits that contains train is
    tried, so 'fitted on train+val' is caught too. Rows added after scaling would hide the
    signature, so it is also tested without repeated rows and, when params.source is set,
    on the rows found in the source only. `tol_mean` and `tol_std` are the widest gap allowed.
    On large data a match must also be closer than `chance_margin` of the sampling noise, plus
    what rounding to 2 or more decimals explains, so data that is already roughly standard
    doesn't block. Raise `chance_margin` to loosen this. A real scaler can miss that bar: on
    values rounded more coarsely or to significant figures, in float32 with a large offset, or
    with rows added or dropped afterwards. preprocessing.global_scaling_near warns about what
    is left within `tol_mean` and `tol_std`."""
    if len(v.splits) < 2:
        return skipped("one split only")
    hits = _blocking_hits(v, opt["tol_mean"], opt["tol_std"], opt["chance_margin"])
    if hits:
        return failed(f"{len(hits)} feature(s) scaled with statistics of evaluation data: "
                      + ", ".join(f"{k} ({x})" for k, x in list(hits.items())[:6]),
                      "Fit the scaler on the training split only (fit on train, transform the "
                      "others), e.g. inside a sklearn Pipeline after the split.", features=hits)
    return passed(f"no feature is standardised or min-max scaled over evaluation rows "
                  f"({len(v.numeric_features())} numeric features tested)")


@check("preprocessing.global_scaling_near", "preprocessing", "warn",
       "No feature is almost scaled over evaluation rows", tol_mean=1e-4, tol_std=1e-3)
def global_scaling_near(v, opt):
    """Features within `tol_mean` and `tol_std`, or those of preprocessing.global_scaling if
    wider, of mean 0 and std 1 over train plus evaluation rows that it doesn't block, because
    chance can't be ruled out.
    Either a scaler was fitted on evaluation rows and the result lost precision or changed
    rows afterwards, or the feature is roughly standard by nature, which on large data with
    many such features is common."""
    if len(v.splits) < 2:
        return skipped("one split only")
    definition = REGISTRY["preprocessing.global_scaling"]
    severity, blocking = resolve_check("preprocessing.global_scaling", v.spec,
                                       definition.severity, definition.options)
    exact = {} if severity == "off" else _blocking_hits(
        v, blocking["tol_mean"], blocking["tol_std"], blocking["chance_margin"])
    near = _scaling_hits(v, max(opt["tol_mean"], blocking["tol_mean"]),
                         max(opt["tol_std"], blocking["tol_std"]), None, minmax=False)
    hits = {f: x for f, x in near.items() if f not in exact}
    if hits:
        return failed(f"{len(hits)} feature(s) nearly standardised over evaluation rows: "
                      + ", ".join(list(hits)[:6]),
                      "If a step scales these features, fit it on the training split only. If "
                      "they are roughly standard by nature, turn this check off.", features=hits)
    return passed("no feature is nearly standardised over evaluation rows")


@check("preprocessing.fitted_levels", "preprocessing", "error",
       "Fitted vocabularies come from training data only")
def fitted_levels(v, opt):
    """For each entry in params.fitted_levels (a category list, vocabulary or encoder classes
    that a step learned), every level except those in `allow` must occur in the training data
    (`column` of `dataset`, default the train split, optionally filtered by the pandas query
    `where`). A level seen only in evaluation data proves the encoder saw it. With
    top_k = true the levels must also be the most frequent training values (ties allowed)."""
    items = v.spec.fitted_levels
    if not items:
        return skipped("no params.fitted_levels")
    bad = {}
    for _i, it in enumerate(items):
        name = it.get("name", it["column"])
        d = v.dataset(it.get("dataset", v.train_name))
        if it.get("where"):
            d = d.query(it["where"], engine="python")
        if it["column"] not in d.columns:
            bad[name] = f"column '{it['column']}' not in {it.get('dataset', v.train_name)}"
            continue
        col = d[it["column"]].dropna()
        counts = value_texts(col).value_counts()
        numeric = is_numeric_kind(kind(col))
        allow = {value_text(a, numeric) for a in it.get("allow", [])}
        levels = [t for x in it["levels"] if (t := value_text(x, numeric)) not in allow]
        unseen = [x for x in levels if x not in counts.index]
        if unseen:
            bad[name] = {"not_in_training_data": examples(unseen, 10), "count": len(unseen)}
            continue
        if it.get("top_k"):
            floor = counts[levels].min() if levels else 0
            above = counts[(counts > floor) & ~counts.index.isin(levels)]
            if len(above):
                bad[name] = {"more_frequent_but_missing": examples(above.index, 10)}
    if bad:
        return failed(f"{len(bad)} fitted vocabulary(ies) not explained by training data: "
                      + ", ".join(f"{k}: {x}" for k, x in bad.items()),
                      "Fit encoders/vocabularies on training rows only.", levels=bad)
    return passed(f"{len(items)} fitted vocabulary(ies) come from training data")

