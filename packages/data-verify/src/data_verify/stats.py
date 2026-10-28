"""Statistics used by the leakage and poisoning checks (numpy/pandas only).

`feature_scores` measures how well each feature alone predicts a label, on a 0-1 scale where
1 means "the label can be read off this feature":

- binary label: ROC AUC, made symmetric (max(auc, 1 - auc));
- multiclass: accuracy above the majority class, normalised: (acc - base) / (1 - base);
- regression: absolute Spearman correlation.

A feature is scored several ways and the highest score wins: directly (ranks), through a
two-fold cross-fitted encoding of its values (numbers cut into up to 32 quantile bins), and,
for numbers, by a two-fold 1-D nearest-neighbour prediction. The cross-fits catch
non-monotonic and categorical leaks, while unseen values (ids, free text) fall back to the
prior and score near chance, so identifiers do not look predictive.
Missing values form their own level, so "missing only when positive" is caught as well.
Cross-fitted scores are computed from per-(level, fold) counts, so they cost O(rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import kind, numeric_values


def auc_counts(score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC from groups: each group has one score and counts of positives/negatives.
    Tied scores count half, as with average ranks."""
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    vals, inv = np.unique(score, return_inverse=True)
    p = np.bincount(inv, weights=pos, minlength=len(vals))
    n = np.bincount(inv, weights=neg, minlength=len(vals))
    below = np.cumsum(n) - n
    return float((p * (below + 0.5 * n)).sum() / (P * N))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC with ties averaged. y is 0/1."""
    y = y.astype(float)
    return auc_counts(score, y, 1 - y)


def avg_ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(avg_ranks(a), avg_ranks(b))


def codes_of(s: pd.Series, bins: int = 32) -> np.ndarray:
    """Integer codes per row: quantile bins for numbers, values for the rest; missing = -1."""
    k = kind(s)
    if k in ("int", "float", "datetime"):
        x = numeric_values(s)
        ok = ~np.isnan(x)
        codes = np.full(len(x), -1, dtype=np.int64)
        if ok.sum():
            uniq = np.unique(x[ok])
            if len(uniq) <= bins:
                codes[ok] = np.searchsorted(uniq, x[ok])
            else:
                edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, bins + 1)[1:-1]))
                codes[ok] = np.searchsorted(edges, x[ok], side="right")
        return codes
    c, _ = pd.factorize(s.astype("string"), use_na_sentinel=True)
    return c.astype(np.int64)


def _groups(codes: np.ndarray, folds: np.ndarray):
    """Row -> (level, fold) group id; missing (-1) becomes a regular level."""
    c = codes - codes.min() if len(codes) else codes
    k = int(c.max()) + 1 if len(c) else 1
    return c * 2 + folds, k


def crossfit_binary(codes, y, folds) -> float:
    """AUC of the out-of-fold positive rate per level (symmetric), from counts only."""
    g, k = _groups(codes, folds)
    n = np.bincount(g, minlength=2 * k).reshape(k, 2).astype(float)
    p = np.bincount(g, weights=y, minlength=2 * k).reshape(k, 2)
    prior = p.sum(0) / np.maximum(n.sum(0), 1)          # per fold
    other_n, other_p = n[:, ::-1], p[:, ::-1]           # the other fold's counts
    enc = np.where(other_n > 0, other_p / np.maximum(other_n, 1), prior[::-1])
    a = auc_counts(enc.ravel(), p.ravel(), (n - p).ravel())
    return max(a, 1 - a) if not np.isnan(a) else a


def crossfit_multiclass(codes, y, folds, n_classes: int) -> float:
    """Normalised accuracy of predicting the other fold's majority class per level."""
    g, k = _groups(codes, folds)
    tab = np.bincount(g * n_classes + y, minlength=2 * k * n_classes).reshape(k, 2, n_classes)
    prior = tab.sum(0)                                   # fold x class
    other = tab[:, ::-1, :]
    has = other.sum(2, keepdims=True) > 0
    dist = np.where(has, other, prior[::-1][None, :, :])
    pred = dist.argmax(2)                                # level x fold
    correct = np.take_along_axis(tab, pred[:, :, None], 2).sum()
    acc = correct / len(y)
    base = np.bincount(y, minlength=n_classes).max() / len(y)
    return float((acc - base) / (1 - base)) if base < 1 else float("nan")


def crossfit_regression(codes, y_rank, y, folds) -> float:
    """|Spearman| between the out-of-fold mean per level and the label."""
    g, k = _groups(codes, folds)
    n = np.bincount(g, minlength=2 * k).reshape(k, 2).astype(float)
    s = np.bincount(g, weights=y, minlength=2 * k).reshape(k, 2)
    prior = s.sum(0) / np.maximum(n.sum(0), 1)
    enc_g = np.where(n[:, ::-1] > 0, s[:, ::-1] / np.maximum(n[:, ::-1], 1), prior[::-1]).ravel()
    # average rank of each group's encoded value, then expand to rows
    sizes = n.ravel()
    order = np.argsort(enc_g, kind="stable")
    vals = enc_g[order]
    cum = np.cumsum(sizes[order])
    start = cum - sizes[order]
    # merge tied encodings into one rank block
    tie_id = np.r_[0, np.cumsum(vals[1:] != vals[:-1])]
    block_start = pd.Series(start).groupby(tie_id).transform("min").to_numpy()
    block_end = pd.Series(cum).groupby(tie_id).transform("max").to_numpy()
    rank_sorted = (block_start + block_end + 1) / 2
    rank_g = np.empty_like(rank_sorted)
    rank_g[order] = rank_sorted
    r = pearson(rank_g[g], y_rank)
    return abs(r) if not np.isnan(r) else r


def crossfit_neighbors(x: np.ndarray, y: np.ndarray, task: str, n_classes: int,
                       folds: np.ndarray, seed: int = 0) -> float:
    """Two-fold 1-D nearest-neighbour score: each row is predicted from the labels of its left
    and right neighbours by value in the other fold. Catches any piecewise mapping from the
    feature to the label (thresholds, bands, |x| > c), monotonic or not. x has no NaN."""
    rng = np.random.default_rng(seed)
    pred = np.zeros(len(x)) if task != "multiclass" else np.zeros(len(x), dtype=np.int64)
    for f in (0, 1):
        a, b = folds != f, folds == f
        if a.sum() < 2 or b.sum() == 0:
            return float("nan")
        xa, ya = x[a], y[a]
        order = np.lexsort((rng.random(len(xa)), xa))      # random order among ties
        xs, ys = xa[order], ya[order]
        pos = np.searchsorted(xs, x[b])
        left = ys[np.clip(pos - 1, 0, len(xs) - 1)]
        right = ys[np.clip(pos, 0, len(xs) - 1)]
        if task == "multiclass":
            pick = rng.random(len(left)) < 0.5
            pred[b] = np.where(pick, left, right)
        else:
            pred[b] = (left + right) / 2
    if task == "binary":
        a = auc(pred, y)
        return max(a, 1 - a) if not np.isnan(a) else a
    if task == "multiclass":
        acc = float((pred == y).mean())
        p = np.bincount(y, minlength=n_classes) / len(y)
        chance = float((p ** 2).sum())    # agreement of two random draws
        return (acc - chance) / (1 - chance) if chance < 1 else float("nan")
    r = spearman(pred, y)
    return abs(r) if not np.isnan(r) else r


def prepare_label(y: pd.Series, task: str):
    """Labels as numbers: 0/1 (binary), 0..k-1 (multiclass) or float (regression), the mask
    of usable rows, and the number of classes."""
    ok = y.notna().to_numpy()
    if task == "regression":
        v = numeric_values(y)
        return v, ok & ~np.isnan(v), 0
    codes, uniq = pd.factorize(y.astype("string"), sort=True)
    return codes.astype(np.int64), ok & (codes >= 0), len(uniq)


def score_feature(x: pd.Series, y: np.ndarray, task: str, n_classes: int,
                  folds: np.ndarray, y_rank: np.ndarray | None = None,
                  direct: bool = True, min_rows: int = 30) -> float:
    """0-1 single-feature predictiveness (see module docstring). NaN if not measurable."""
    if len(y) < min_rows:
        return float("nan")
    best = float("nan")
    k = kind(x)
    numeric = direct and k in ("int", "float", "datetime", "bool")
    codes = codes_of(x)
    if task in ("binary", "multiclass") and n_classes < 2:
        return float("nan")
    if numeric:
        v = numeric_values(x.astype("float64") if k == "bool" else x)
        ok = ~np.isnan(v)
        if ok.sum() >= min_rows:
            yy = y[ok]
            if task == "binary" and 5 <= yy.sum() <= len(yy) - 5:
                a = auc(v[ok], yy)
                best = max(a, 1 - a)
            elif task == "regression":
                best = abs(spearman(v[ok], yy))
            if task != "binary" or 5 <= yy.sum() <= len(yy) - 5:
                best = np.nanmax([best, crossfit_neighbors(v[ok], yy, task, n_classes,
                                                           folds[ok])])
    if task == "binary":
        best = np.nanmax([best, crossfit_binary(codes, y.astype(float), folds)])
    elif task == "multiclass":
        best = np.nanmax([best, crossfit_multiclass(codes, y, folds, n_classes)])
    else:
        best = np.nanmax([best, crossfit_regression(codes, y_rank, y, folds)])
    return float(best)


def feature_scores(df: pd.DataFrame, features: list[str], label: str, task: str,
                   seed: int = 0) -> dict[str, float]:
    """Score every feature, plus `isnull(<feature>)` for partly-missing features."""
    y_raw, ok, n_classes = prepare_label(df[label], task)
    sub = df.loc[ok, [f for f in features if f in df.columns and f != label]]
    y = y_raw[ok]
    y_rank = avg_ranks(y) if task == "regression" else None
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, 2, len(sub))
    out: dict[str, float] = {}
    for f in sub.columns:
        x = sub[f]
        out[f] = score_feature(x, y, task, n_classes, folds, y_rank)
        nulls = x.isna()
        if 0 < nulls.mean() < 1:
            out[f"isnull({f})"] = score_feature(nulls.astype("int8"), y, task, n_classes,
                                                folds, y_rank, direct=False)
    return out


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population stability index between two discrete distributions (counts), minus its
    expected value under no drift, (bins - 1) * (1/n1 + 1/n2), so that small samples do not
    look drifted."""
    n1, n2 = expected.sum(), actual.sum()
    p = np.clip(expected / max(n1, 1), eps, None)
    q = np.clip(actual / max(n2, 1), eps, None)
    raw = float(np.sum((q - p) * np.log(q / p)))
    used = int(((expected + actual) > 0).sum())
    noise = (used - 1) * (1 / max(n1, 1) + 1 / max(n2, 1))
    return max(raw - noise, 0.0)


def binned_counts(train: pd.Series, other: pd.Series, bins: int = 10):
    """Counts of train and other over bins fitted on train (quantiles / top values + NaN)."""
    k = kind(train)
    if k in ("int", "float", "datetime"):
        a, b = numeric_values(train), numeric_values(other)
        ok = a[~np.isnan(a)]
        edges = np.unique(np.quantile(ok, np.linspace(0, 1, bins + 1)[1:-1])) if len(ok) else []
        ca = np.where(np.isnan(a), -1, np.searchsorted(edges, a, side="right"))
        cb = np.where(np.isnan(b), -1, np.searchsorted(edges, b, side="right"))
        levels = np.arange(-1, len(edges) + 1)
    else:
        a = train.astype("string").fillna("\0NA")
        b = other.astype("string").fillna("\0NA")
        top = a.value_counts().index[:50]
        ca = np.where(a.isin(top), a, "\0other")
        cb = np.where(b.isin(top), b, "\0other")
        levels = np.array(list(top) + ["\0other"], dtype=object)
    sa = pd.Series(ca).value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)
    sb = pd.Series(cb).value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float)
    return sa, sb


def robust_z(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    return np.abs(x - center) / scale if scale > 0 else np.zeros_like(x)


def robust_scale(x: np.ndarray) -> tuple[float, float]:
    """Median and a robust spread (MAD, else IQR/1.349, else mean absolute deviation)."""
    x = x[~np.isnan(x)]
    if not len(x):
        return 0.0, 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad > 0:
        return med, mad
    q1, q3 = np.quantile(x, [0.25, 0.75])
    if q3 > q1:
        return med, float(q3 - q1) / 1.349
    mean_abs = float(np.mean(np.abs(x - med)))
    return med, mean_abs * 1.2533
