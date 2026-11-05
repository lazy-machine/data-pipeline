"""Labels, weights and poisoning signals: the training data was not tampered with or skewed,
and evaluation splits look like the population the model is meant for."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from ..core import (
    check, examples, failed, fmt, frac, jsonable, kind, numeric_values, passed, skipped, value_set,
    value_texts,
)
from ..stats import auc, binned_counts, prepare_label, psi, robust_scale, robust_z, spearman

# -- labels --------------------------------------------------------------------------------


def _labels_in(v, d):
    return [lb for lb in v.labels if lb in d.columns]


@check("labels.valid", "labels", "error", "Labels are present and valid")
def labels_valid(v, opt):
    """Every row has a label; labels are within params.label_values / params.label_range when
    given; a classification training split has at least two classes. Out-of-domain label
    values are a sign of corrupted or injected rows."""
    if not v.labels:
        return skipped("no params.label")
    problems, details = [], {}
    for name, d in v.splits.items():
        for lbl in _labels_in(v, d):
            y = d[lbl]
            if (n := int(y.isna().sum())):
                problems.append(f"{name}.{lbl}: {n} missing")
                details[f"{name}.{lbl}.missing"] = n
            if v.spec.label_values is not None:
                allowed = value_set(v.spec.label_values, y)
                bad = ~value_texts(y.dropna()).isin(allowed)
                if bad.any():
                    problems.append(f"{name}.{lbl}: {int(bad.sum())} outside label_values")
                    details[f"{name}.{lbl}.unexpected"] = examples(
                        y.dropna()[bad.to_numpy()].unique(), 5)
            if v.spec.label_range is not None and kind(y) in ("int", "float", "bool"):
                x = numeric_values(y)
                lo, hi = v.spec.label_range
                out = (x < lo) | (x > hi)
                if out.any():
                    problems.append(f"{name}.{lbl}: {int(out.sum())} outside {[lo, hi]}")
            if kind(y) == "float" and np.isinf(numeric_values(y)).any():
                problems.append(f"{name}.{lbl}: infinite values")
    for lbl in _labels_in(v, v.train):
        if v.task(lbl) != "regression" and v.train[lbl].nunique() < 2:
            problems.append(f"{v.train_name}.{lbl}: a single class")
    if problems:
        return failed("; ".join(problems[:8]), "Fix or drop invalid labels before the split.",
                      problems=problems, **details)
    return passed(f"{len(v.labels)} label(s) valid in {len(v.splits)} splits")


@check("labels.unseen_classes", "labels", "error", "Evaluation classes exist in training")
def unseen_classes(v, opt):
    """Every class in an evaluation split must occur in the training split. A class the model
    never saw cannot be predicted, so the split (or the labels) are broken."""
    hits = {}
    for lbl in _labels_in(v, v.train):
        if v.task(lbl) == "regression":
            continue
        seen = set(value_texts(v.train[lbl].dropna()))
        for name, d in v.others():
            if lbl in d.columns:
                new = sorted(set(value_texts(d[lbl].dropna())) - seen)
                if new:
                    hits[f"{name}.{lbl}"] = new[:10]
    if not v.labels:
        return skipped("no params.label")
    if hits:
        return failed("classes missing from training: " + ", ".join(
            f"{k} {x}" for k, x in hits.items()),
            "Stratify the split by the label, or merge/drop the rare classes.", classes=hits)
    return passed("every evaluation class occurs in training")


@check("labels.class_presence", "labels", "warn", "Each class is represented in each split",
       min_count=1)
def class_presence(v, opt):
    """Each training class should appear at least `min_count` times in every labelled split, so
    per-class metrics exist (and a class was not moved wholesale into one split)."""
    if not v.labels:
        return skipped("no params.label")
    hits = {}
    for lbl in _labels_in(v, v.train):
        if v.task(lbl) == "regression":
            continue
        classes = value_texts(v.train[lbl].dropna()).value_counts()
        for name, d in v.splits.items():
            if lbl in d.columns:
                c = value_texts(d[lbl].dropna()).value_counts()
                low = {k: int(c.get(k, 0)) for k in classes.index if c.get(k, 0) < opt["min_count"]}
                if low:
                    hits[f"{name}.{lbl}"] = low
    if hits:
        return failed("under-represented classes: " + ", ".join(
            f"{k} {x}" for k, x in list(hits.items())[:5]), classes=hits)
    return passed("every class present in every split")


@check("labels.shift", "labels", "warn", "Label distribution is similar across splits",
       max_distance=0.15)
def label_shift(v, opt):
    """Total variation distance between class frequencies (classification), or the shift in
    median relative to the training spread (regression), of each split against train. A
    large shift in a random split points at a broken split or at labels altered in one split
    (label-flipping)."""
    if not v.labels or len(v.splits) < 2:
        return skipped("needs a label and two splits")
    dist = {}
    for lbl in _labels_in(v, v.train):
        if v.task(lbl) == "regression":
            ytr = numeric_values(v.train[lbl])
            med, sc = robust_scale(ytr)
            for name, d in v.others():
                if lbl in d.columns and sc > 0:
                    y = numeric_values(d[lbl])
                    y = y[~np.isnan(y)]
                    if len(y):
                        dist[f"{name}.{lbl}"] = fmt(abs(float(np.median(y)) - med) / sc)
            continue
        p = value_texts(v.train[lbl]).value_counts(normalize=True)
        for name, d in v.others():
            if lbl in d.columns:
                q = value_texts(d[lbl]).value_counts(normalize=True)
                idx = p.index.union(q.index)
                dist[f"{name}.{lbl}"] = fmt(0.5 * float(
                    (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()))
    hits = {k: x for k, x in dist.items() if x > opt["max_distance"]}
    if hits:
        return failed(f"label distribution differs from {v.train_name}: "
                      + ", ".join(f"{k} distance {x}" for k, x in hits.items()),
                      "Expected with a time-based split and real drift; otherwise stratify "
                      "the split by the label.", distance=dist)
    return passed(f"label distributions close to {v.train_name} (max distance "
                  f"{max(dist.values(), default=0)} <= {opt['max_distance']})", distance=dist)


# -- weights -------------------------------------------------------------------------------


@check("weights.valid", "weights", "error", "Sample weights are finite and non-negative")
def weights_valid(v, opt):
    """params.weight must be finite, non-negative, and not all zero in any split. A negative or
    infinite weight lets single rows dominate (or invert) the training loss."""
    w = v.spec.weight
    if not w:
        return skipped("no params.weight")
    problems = {}
    for name, d in v.splits.items():
        if w not in d.columns:
            continue
        x = numeric_values(d[w])
        p = {"missing": int(np.isnan(x).sum()), "infinite": int(np.isinf(x).sum()),
             "negative": int((x < 0).sum())}
        if np.nansum(np.where(np.isinf(x), 0, x)) == 0:
            p["all_zero"] = 1
        p = {k: n for k, n in p.items() if n}
        if p:
            problems[name] = p
    if problems:
        return failed(f"invalid weights: {problems}", "Weights must be finite and >= 0.",
                      splits=problems)
    return passed(f"weights valid in {len(v.splits)} split(s)")


@check("weights.concentration", "weights", "warn", "No few rows dominate the weights",
       max_top_share=0.5, max_ratio=1000.0)
def weights_concentration(v, opt):
    """In the training split, the top 1% of rows by weight must hold at most `max_top_share`
    of the total weight, and the largest weight must be at most `max_ratio` times the median.
    Inflated weights on a handful of rows are a low-volume poisoning attack."""
    w = v.spec.weight
    if not w or w not in v.train.columns:
        return skipped("no params.weight")
    x = numeric_values(v.train[w])
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) < 100 or x.sum() == 0:
        return skipped("too few weighted rows")
    xs = np.sort(x)[::-1]
    top = xs[: max(1, len(xs) // 100)].sum() / xs.sum()
    med = float(np.median(x))
    ratio = float(xs[0] / med) if med > 0 else float("inf")
    details = {"top1pct_share": fmt(top), "max_over_median": fmt(ratio)}
    if top > opt["max_top_share"] or ratio > opt["max_ratio"]:
        return failed(f"weights concentrated: top 1% hold {top:.1%}, max/median {ratio:.4g}",
                      "Check how weights are computed; cap or normalise them.", **details)
    return passed(f"top 1% of rows hold {top:.1%} of the weight", **details)


# -- poisoning signals ---------------------------------------------------------------------


@check("poisoning.outliers", "poisoning", "warn",
       "Training data has no excess of extreme values", z=20.0, ratio=3.0, slack=0.005)
def outliers(v, opt):
    """Share of training rows with at least one feature further than `z` robust deviations from
    the evaluation data's median, against the same share in the evaluation splits. Training
    having more than `ratio` times (+`slack`) as many extreme rows as held-out data looks like
    injected points. Uses the evaluation splits as the reference, so it needs two splits."""
    feats = v.numeric_features()
    if not feats or len(v.splits) < 2:
        return skipped("needs numeric features and an evaluation split")
    tr = v.sample(v.train_name)
    refs = [v.sample(n) for n, _ in v.others()]
    ref = pd.concat([r[feats] for r in refs], ignore_index=True)
    ext_tr = np.zeros(len(tr), bool)
    ext_ref = np.zeros(len(ref), bool)
    per_feature = {}
    for f in feats:
        c, s = robust_scale(numeric_values(ref[f]))
        if s <= 0:
            continue
        zt = robust_z(numeric_values(tr[f]), c, s) > opt["z"]
        zr = robust_z(numeric_values(ref[f]), c, s) > opt["z"]
        ext_tr |= zt
        ext_ref |= zr
        a, b = zt.mean(), zr.mean() if len(zr) else 0.0
        if a > opt["ratio"] * b + opt["slack"]:
            per_feature[f] = [fmt(a), fmt(b)]
    rt, rr = float(ext_tr.mean()), float(ext_ref.mean()) if len(ext_ref) else 0.0
    details = {"train_extreme_rows": fmt(rt), "reference_extreme_rows": fmt(rr),
               "features": dict(list(per_feature.items())[:20])}
    if rt > opt["ratio"] * rr + opt["slack"]:
        return failed(f"{rt:.2%} of training rows are extreme vs {rr:.2%} of held-out rows",
                      "Inspect the flagged features' extreme training rows: corrupted units, "
                      "a broken join, or injected points.", **details)
    return passed(f"extreme rows: train {rt:.2%}, held-out {rr:.2%}", **details)


_TOKEN = re.compile(r"[a-z0-9]+")


def _text_like(s: pd.Series) -> bool:
    if kind(s) != "string":
        return False
    sample = s.dropna().astype("string").head(500)
    return len(sample) > 0 and sample.str.len().mean() > 30


@check("poisoning.spurious_association", "poisoning", "warn",
       "No rare value or token ties to one class in training only",
       min_support=20, purity=0.98, max_coverage=0.2, min_drop=0.3, min_eval_support=5)
def spurious_association(v, opt):
    """Backdoor search. For each classification label, looks for feature values (categories,
    small-integer values, and words in text columns) that occur in at most `max_coverage` of
    training rows, at least `min_support` times, and come with one class in at least `purity`
    of them, while the same value in evaluation splits (at least `min_eval_support` rows) is
    tied to that class at least `min_drop` less often, or is (nearly) absent from them although
    their size predicts it there (so few rows has a Poisson chance below 0.001). Such a
    trigger in train that does not hold elsewhere is what a backdoor (trigger + target label)
    poisoning attack leaves behind."""
    labels = [lb for lb in _labels_in(v, v.train) if v.task(lb) != "regression"]
    if not labels or len(v.splits) < 2:
        return skipped("needs a classification label and an evaluation split")
    tr = v.sample(v.train_name)
    others = [v.sample(n) for n, _ in v.others()]
    hits = []
    for lbl in labels:
        ytr = value_texts(tr[lbl]).to_numpy()
        base = pd.Series(ytr).value_counts(normalize=True)
        for f in v.features:
            x = tr[f]
            text = _text_like(x)
            if not text and not (kind(x) in ("string", "bool") or
                                 (kind(x) == "int" and x.nunique() <= 100)):
                continue
            if text:
                trows = _token_rows(x)
                orows = [_token_rows(o[f]) if f in o.columns else {} for o in others]
            else:
                trows = _value_rows(x)
                orows = [_value_rows(o[f]) if f in o.columns else {} for o in others]
            n = len(tr)
            n_eval = sum(len(o) for o in others if f in o.columns and lbl in o.columns)
            for val, idx in trows.items():
                if not (opt["min_support"] <= len(idx) <= opt["max_coverage"] * n):
                    continue
                cls = Counter(ytr[idx]).most_common(1)[0]
                p_tr = cls[1] / len(idx)
                if p_tr < opt["purity"] or base.get(cls[0], 0) > 0.9:
                    continue
                e_n, e_hit = 0, 0
                for o, rows in zip(others, orows, strict=True):
                    if lbl in o.columns and val in rows:
                        yo = value_texts(o[lbl]).to_numpy()[rows[val]]
                        e_n += len(yo)
                        e_hit += int((yo == cls[0]).sum())
                if e_n >= opt["min_eval_support"]:
                    hit = p_tr - e_hit / e_n >= opt["min_drop"]
                else:  # (nearly) absent from evaluation data although it should be there
                    hit = _poisson_cdf(e_n, len(idx) * n_eval / n) < 0.001
                if hit:
                    hits.append({"feature": f, "value": jsonable(val), "label": lbl,
                                 "class": cls[0], "train_rows": len(idx),
                                 "train_purity": fmt(p_tr), "eval_rows": e_n,
                                 "eval_purity": fmt(e_hit / e_n) if e_n else None,
                                 "expected_eval_rows": fmt(len(idx) * n_eval / n),
                                 "token": text})
    hits.sort(key=lambda h: (-(h["train_purity"] - (h["eval_purity"] or 0)) * h["train_rows"],
                             h["feature"], str(h["value"])))
    if hits:
        h = hits[0]
        held_out = (f"{h['eval_purity']:.0%} of {h['eval_rows']} held-out rows"
                    if h["eval_purity"] is not None else
                    f"{h['eval_rows']} held-out rows (about {h['expected_eval_rows']:.0f} "
                    "expected)")
        return failed(f"{len(hits)} value(s) tie to one class in training only, e.g. "
                      f"{h['feature']}={h['value']!r} -> {h['label']}={h['class']} in "
                      f"{h['train_purity']:.0%} of {h['train_rows']} train rows vs "
                      + held_out,
                      "Inspect the training rows carrying this value: a trigger added to "
                      "examples of one class is a backdoor attack. If it is a genuine "
                      "change between periods, this warning is expected.", associations=hits[:20])
    return passed("no training-only value/class associations found")


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    term = total = math.exp(-lam)
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return total


def _value_rows(s: pd.Series) -> dict:
    codes, uniq = pd.factorize(value_texts(s), use_na_sentinel=True)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    out = {}
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    ends = np.r_[starts[1:], len(sc)]
    for a, b in zip(starts, ends, strict=True):
        if sc[a] >= 0:
            out[uniq[sc[a]]] = order[a:b]
    return out


def _token_rows(s: pd.Series) -> dict:
    rows: dict[str, list[int]] = {}
    for i, text in enumerate(s.astype("string").fillna("").str.lower()):
        for tok in set(_TOKEN.findall(text)):
            rows.setdefault(tok, []).append(i)
    return {k: np.asarray(x) for k, x in rows.items()}


_SPEARMAN_SE = 1.03  # Fieller: a Fisher-transformed Spearman has variance about 1.06 / (n - 3)


def _matrix(d: pd.DataFrame, feats: list[str], stats: dict[str, tuple[float, float]]):
    cols = [np.ones(len(d))]
    for f in feats:
        center, scale = stats[f]
        cols.append(np.clip(np.nan_to_num((numeric_values(d[f]) - center) / scale), -5, 5))
    return np.column_stack(cols)


def _agreement(pred: np.ndarray, y: np.ndarray, task: str):
    """How well ``pred`` orders ``y``: (score, a value to compare, its standard error), or None.

    The score is 0 for no signal and 1 for perfect. Binary labels use AUC (as 2 * AUC - 1),
    because it doesn't depend on the share of positives, so a change in class balance
    between splits is not mistaken for corrupted labels. A correlation does depend on it.
    """
    if task == "binary":
        a = auc(pred, y)
        if np.isnan(a):  # a class is missing
            return None
        pos = int(y.sum())
        neg = len(y) - pos
        q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)   # Hanley and McNeil, 1982
        var = (a * (1 - a) + (pos - 1) * (q1 - a ** 2) + (neg - 1) * (q2 - a ** 2)) / (pos * neg)
        score = 2 * a - 1
        return score, score, 2 * math.sqrt(max(var, 0.0))
    r = spearman(pred, y)
    if np.isnan(r):
        return None
    fisher = float(np.arctanh(np.clip(r, -0.999, 0.999)))
    return r, fisher, _SPEARMAN_SE / math.sqrt(len(y) - 3)


def _inconsistent(held: float, ev: float, z: float, opt: dict) -> bool:
    """Evaluation labels fit better than held-out training labels, or the opposite way round."""
    weaker = ev >= opt["min_signal"] and ev - held >= opt["min_gap"] and z >= opt["z"]
    backwards = held >= opt["min_signal"] and ev <= -opt["min_signal"] and z <= -opt["z"]
    return weaker or backwards


@check("poisoning.label_consistency", "poisoning", "warn",
       "Training labels fit the features as well as evaluation labels",
       z=3.0, min_gap=0.05, min_signal=0.1, min_rows=100)
def label_consistency(v, opt):
    """Fits one linear model on half of the training rows, then compares how well it predicts
    the other half with how well it predicts each evaluation split. Corrupted training labels
    weaken that relationship in train only, so evaluation labels fit better than held-out
    training labels by at least `min_gap` and `z` standard errors. Also fires when the model
    predicts evaluation labels backwards. Needs a binary or regression label and numeric
    features that predict it (at least `min_signal` on a 0-1 scale: 2 * AUC - 1 for a binary
    label, rank correlation for regression). Sees label noise of about 15% and above on a few
    thousand rows, and targeted flips of a subgroup; not small flip rates or a backdoor with
    a rare trigger. A shift in class balance doesn't fire it. For a regression label, features
    that vary more in an evaluation split than in train do, because the label is then easier
    to predict."""
    if not v.labels or len(v.splits) < 2:
        return skipped("needs a label and at least two splits")
    feats = v.numeric_features()
    if not feats:
        return skipped("no numeric features")
    hits, compared = [], 0
    for lbl in v.labels:
        if lbl not in v.train.columns or v.task(lbl) == "multiclass":
            continue
        task = v.task(lbl)
        train = v.sample(v.train_name)
        y, ok, _ = prepare_label(train[lbl], task)
        train, y = train[ok], y[ok].astype(float)
        if len(train) < 2 * opt["min_rows"]:
            continue
        stats = {f: robust_scale(numeric_values(train[f])) for f in feats}
        stats = {f: (c if np.isfinite(c) else 0.0, s if np.isfinite(s) and s > 0 else 1.0)
                 for f, (c, s) in stats.items()}
        fit = np.random.default_rng(v.spec.seed).random(len(train)) < 0.5
        x = _matrix(train, feats, stats)
        coef = np.linalg.lstsq(x[fit], y[fit], rcond=None)[0]
        held = _agreement(x[~fit] @ coef, y[~fit], task)
        if held is None:
            continue
        for name, d in v.others():
            if lbl not in d.columns:
                continue
            d = v.sample(name)
            ye, oke, _ = prepare_label(d[lbl], task)
            if oke.sum() < opt["min_rows"]:
                continue
            ev = _agreement(_matrix(d[oke], feats, stats) @ coef, ye[oke].astype(float), task)
            if ev is None:
                continue
            compared += 1
            se = math.hypot(held[2], ev[2])
            if se and _inconsistent(held[0], ev[0], (ev[1] - held[1]) / se, opt):
                hits.append({"label": lbl, "split": name, "held_out_train": fmt(held[0]),
                             "evaluation": fmt(ev[0]), "z": fmt((ev[1] - held[1]) / se)})
    if not compared:
        return skipped("needs a binary or regression label and enough rows in each split")
    if hits:
        return failed("labels fit the features differently in train and evaluation: " + ", ".join(
            f"{h['label']}: train {h['held_out_train']}, {h['split']} {h['evaluation']}"
            for h in hits),
            "Training labels may be corrupted or flipped. Compare them with their source. For "
            "a regression label, check first whether the features vary more in one split, "
            "which also causes this.",
            comparisons=hits)
    return passed(f"training labels fit the features as well as evaluation labels "
                  f"({compared} comparison(s))")


# -- drift ---------------------------------------------------------------------------------


@check("drift.features", "drift", "warn", "Feature distributions are stable across splits",
       max_psi=0.25, max_share=0.25)
def drift_features(v, opt):
    """Population stability index of every feature in each evaluation split against train
    (10 quantile bins fitted on train; top 50 values for categories; missing as its own bin).
    Warns when more than `max_share` of the features exceed `max_psi`. Expected for time-based
    splits; in a random split, broad drift means the splits were produced differently."""
    if len(v.splits) < 2 or not v.features:
        return skipped("needs features and two splits")
    tr = v.sample(v.train_name)
    res, worst = {}, {}
    for name, _ in v.others():
        d = v.sample(name)
        vals = {}
        for f in v.features:
            a, b = binned_counts(tr[f], d[f])
            vals[f] = psi(a, b)
        res[name] = vals
        worst[name] = dict(sorted(((k, fmt(x)) for k, x in vals.items()), key=lambda t: -t[1])[:8])
    shares = {n: frac(sum(x > opt["max_psi"] for x in vals.values()), len(vals))
              for n, vals in res.items()}
    if any(s > opt["max_share"] for s in shares.values()):
        return failed("drift in " + ", ".join(f"{n}: {s:.0%} of features PSI>{opt['max_psi']}"
                                              for n, s in shares.items()),
                      "Expected for time-based splits. For random splits, check that every "
                      "split went through the same processing.", share=shares, worst=worst)
    return passed("drift share " + ", ".join(f"{n}={s:.0%}" for n, s in shares.items()),
                  share=shares, worst=worst)


@check("drift.unseen_categories", "drift", "warn",
       "Evaluation categories were seen in training", max_fraction=0.05)
def unseen_categories(v, opt):
    """Share of evaluation rows whose categorical feature value never occurs in training (text
    columns excluded). High shares mean the encoder will fall back to 'unknown' for much of
    the evaluation data, or that the splits come from different sources."""
    feats = [f for f in v.categorical_features() if not _text_like(v.train[f])]
    if not feats or len(v.splits) < 2:
        return skipped("needs categorical features and two splits")
    hits = {}
    for f in feats:
        seen = set(v.train[f].dropna().astype("string"))
        for name, d in v.others():
            s = d[f].dropna().astype("string")
            if len(s):
                share = float((~s.isin(seen)).mean())
                if share > opt["max_fraction"]:
                    hits[f"{name}.{f}"] = fmt(share)
    if hits:
        return failed("categories unseen in training: " + ", ".join(
            f"{k} {x:.1%} of rows" for k, x in list(hits.items())[:8]),
            "Expected for new entities over time; if a column is an entity key, declare it as "
            "params.group and remove it from the features.", columns=hits)
    return passed(f"{len(feats)} categorical feature(s) covered by training values")
