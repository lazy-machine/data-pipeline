"""Temporal leakage: the model never trains on the future of what it is scored on."""

from __future__ import annotations

import pandas as pd

from ..core import check, examples, failed, frac, jsonable, passed, skipped


def _need_time(v):
    if not v.spec.time:
        return skipped("no params.time")
    if not v.has(v.spec.time):
        return skipped(f"time column '{v.spec.time}' missing in some split")
    return None


def _embargo(v):
    e = v.spec.embargo
    if e is None:
        return 0.0 if v.time_is_numeric() else pd.Timedelta(0)
    if v.time_is_numeric():
        if isinstance(e, pd.Timedelta):
            raise ValueError("embargo is a duration but the time column is numeric")
        return e
    return e if isinstance(e, pd.Timedelta) else pd.Timedelta(days=e)


@check("temporal.parse", "temporal", "error", "Time columns parse")
def parse(v, opt):
    """Every non-null value of params.time, params.label_time and params.point_in_time must parse
    as a date/time (or be numeric). Unparseable values silently escape every temporal check."""
    if (s := _need_time(v)) is not None:
        return s
    cols = [v.spec.time, v.spec.label_time, *v.spec.point_in_time]
    bad = {}
    for name, d in v.splits.items():
        for c in filter(None, cols):
            if c not in d.columns:
                continue
            t = v.time_of(name, c)
            lost = d[c].notna().to_numpy() & t.isna().to_numpy()
            if lost.any():
                bad[f"{name}.{c}"] = {"rows": int(lost.sum()),
                                      "examples": examples(d.loc[lost, c].unique(), 3)}
    if bad:
        return failed(f"unparseable time values in {len(bad)} column(s): "
                      + ", ".join(f"{k} e.g. {x['examples']}" for k, x in bad.items()),
                      "Write ISO dates (YYYY-MM-DD) or real datetime columns.", columns=bad)
    ranges = {n: [jsonable(v.time(n).min()), jsonable(v.time(n).max())] for n in v.splits}
    return passed("time ranges " + ", ".join(f"{n} {a}..{b}" for n, (a, b) in ranges.items()),
                  ranges=ranges)


@check("temporal.bounds", "temporal", "error", "Each split stays inside params.time_bounds")
def bounds(v, opt):
    """params.time_bounds gives each split a min and/or max timestamp (inclusive), e.g.
    train = { max = "2023-12-31" }, test = { min = "2024-01-01" }. Rows outside their window
    are rows from the wrong period."""
    if (s := _need_time(v)) is not None:
        return s
    if not v.spec.time_bounds:
        return skipped("no params.time_bounds")
    num = v.time_is_numeric()
    bad = {}
    for split, b in v.spec.time_bounds.items():
        t = v.time(split)
        for side in ("min", "max"):
            if side not in b:
                continue
            lim = v.time_value(b[side], num)
            out = (t < lim) if side == "min" else (t > lim)
            if (n := int(out.sum())):
                bad[f"{split}.{side}"] = {"bound": jsonable(lim), "rows": n,
                                          "fraction": frac(n, len(t)),
                                          "extreme": jsonable(t.min() if side == "min"
                                                              else t.max())}
    if bad:
        return failed("rows outside their time window: " + ", ".join(
            f"{k} {x['rows']} rows (bound {x['bound']}, found {x['extreme']})"
            for k, x in bad.items()),
            "Filter each split to its window in the split step.", violations=bad)
    return passed(f"{len(v.spec.time_bounds)} split(s) within bounds")


@check("temporal.order", "temporal", "error", "Earlier splits end before later ones start")
def order(v, opt):
    """For every pair in params.time_order (a chain 'train < val < test' means every pair:
    train < val, train < test, val < test), the latest timestamp of the earlier split must be
    before the earliest of the later split, by at least params.embargo
    when set. With time_order_scope = "group", this is checked per entity instead (each
    group's train rows end before its test rows start), for per-series forecasting."""
    if (s := _need_time(v)) is not None:
        return s
    if not v.spec.time_order:
        return skipped("no params.time_order")
    emb = _embargo(v)
    zero = 0.0 if v.time_is_numeric() else pd.Timedelta(0)
    bad, info = {}, {}
    for a, b in v.spec.time_order:
        ta, tb = v.time(a), v.time(b)
        if ta.notna().sum() == 0 or tb.notna().sum() == 0:
            continue
        if v.spec.time_order_scope == "global":
            amax, bmin = ta.max(), tb.min()
            gap = bmin - amax
            info[f"{a}<{b}"] = {"end_" + a: jsonable(amax), "start_" + b: jsonable(bmin),
                                "gap": jsonable(gap)}
            if gap <= zero or gap < emb:
                bad[f"{a}<{b}"] = {**info[f"{a}<{b}"],
                                   "rows_of_" + a + "_too_late": int((ta > bmin - emb).sum()
                                                                   if emb != zero else
                                                                   (ta >= bmin).sum()),
                                   "rows_of_" + b + "_too_early": int((tb < amax + emb).sum()
                                                                    if emb != zero else
                                                                    (tb <= amax).sum()),
                                   "embargo": jsonable(emb)}
        else:
            g = v.spec.group
            if not g or not all(v.has(c) for c in g):
                return failed("time_order_scope = 'group' needs params.group in every split")
            ka = pd.Series(v.hashes(a, tuple(g)))
            kb = pd.Series(v.hashes(b, tuple(g)))
            amax = ta.groupby(ka.to_numpy()).max()
            bmin = tb.groupby(kb.to_numpy()).min()
            both = amax.index.intersection(bmin.index)
            gaps = bmin[both] - amax[both]
            viol = (gaps <= zero) | (gaps < emb)
            info[f"{a}<{b}"] = {"groups_in_both": int(len(both))}
            if viol.any():
                bad[f"{a}<{b}"] = {"groups": int(viol.sum()), "groups_in_both": int(len(both)),
                                   "embargo": jsonable(emb)}
    if bad:
        return failed("splits overlap in time: " + ", ".join(
            f"{k} ({', '.join(f'{x}={y}' for x, y in d.items())})" for k, d in bad.items()),
            "Cut the splits at a date (train before, test after), leaving params.embargo "
            "between them if labels look ahead.", violations=bad, pairs=info)
    return passed("time order holds: " + ", ".join(f"{a} < {b}" for a, b in v.spec.time_order),
                  pairs=info)


@check("temporal.overlap", "temporal", "warn", "Splits of time-stamped data do not overlap")
def overlap(v, opt):
    """When params.time is set but neither time_order nor time_bounds is, warns if the
    training period overlaps an evaluation split's period. A random split of time-dependent
    data lets the model learn from the future it is scored on. Declare time_order (or turn
    this check off if rows are truly independent of time)."""
    if (s := _need_time(v)) is not None:
        return s
    if v.spec.time_order or v.spec.time_bounds:
        return skipped("time_order / time_bounds are declared and checked instead")
    tr = v.time(v.train_name)
    hits = {}
    for name, _ in v.others():
        t = v.time(name)
        if t.notna().any() and tr.notna().any() and t.min() <= tr.max():
            inside = int(((t >= tr.min()) & (t <= tr.max())).sum())
            hits[name] = {"rows_inside_train_period": inside, "fraction": frac(inside, len(t))}
    if hits:
        return failed("evaluation rows fall inside the training period: " + ", ".join(
            f"{n} {x['fraction']:.1%}" for n, x in hits.items()),
            "If the target depends on time, split by time and declare params.time_order; if "
            "not, set checks.\"temporal.overlap\" = \"off\".", splits=hits)
    return passed("evaluation splits start after the training period")


@check("temporal.rows_after_outcome", "temporal", "error", "No row is dated after its outcome",
       allow_equal=True)
def rows_after_outcome(v, opt):
    """params.label_time is when each row's label became known (event date, end of the label
    window, censoring date). A row whose own timestamp is after its label time was observed
    after the outcome, so its features can describe the outcome itself (e.g. readings after a
    failure, records after a diagnosis)."""
    lt = v.spec.label_time
    if not lt:
        return skipped("no params.label_time")
    if (s := _need_time(v)) is not None:
        return s
    bad = {}
    for name, d in v.splits.items():
        if lt not in d.columns:
            continue
        t, l_ = v.time(name), v.time_of(name, lt)
        viol = (t > l_) if opt["allow_equal"] else (t >= l_)
        if (n := int(viol.sum())):
            bad[name] = {"rows": n, "fraction": frac(n, len(d))}
    if bad:
        return failed("rows dated after their outcome: " + ", ".join(
            f"{n}={x['rows']}" for n, x in bad.items()),
            "Drop rows observed after the event (cut each entity's history at its outcome).",
            splits=bad)
    return passed("every row precedes its outcome")


@check("temporal.label_after_cutoff", "temporal", "error",
       "Training labels were known before the evaluation period")
def label_after_cutoff(v, opt):
    """A label of an earlier split must be known before the later split starts: for 'a < b'
    in time_order (or bounds where a's max is before b's min), every label_time of a must be
    before the first timestamp of b (and before b's time_bounds min). A training label that
    resolves inside the evaluation period was read from the future (e.g. a 'fails within 30
    days' label for a row 10 days before the cutoff). The embargo is not added here: it is
    the gap temporal.order requires between row timestamps, and is usually set to the label
    horizon for data without label_time. With label_time, an embargo is optional."""
    lt = v.spec.label_time
    if not lt:
        return skipped("no params.label_time")
    if (s := _need_time(v)) is not None:
        return s
    num = v.time_is_numeric()
    bounds = v.spec.time_bounds
    # an earlier split's labels must be known before the later split starts (minus embargo)
    pairs = list(v.spec.time_order)
    if not pairs:  # infer from bounds: A ends (max) before B starts (min)
        for a, ba in bounds.items():
            for b, bb in bounds.items():
                if a != b and "max" in ba and "min" in bb and \
                        v.time_value(bb["min"], num) > v.time_value(ba["max"], num):
                    pairs.append((a, b))
    cutoffs: dict[str, object] = {}
    for a, b in pairs:
        starts = []
        if "min" in bounds.get(b, {}):
            starts.append(v.time_value(bounds[b]["min"], num))
        tb = v.time(b)
        if tb.notna().any():
            starts.append(tb.min())
        if starts:
            c = min(starts)
            cutoffs[a] = min(cutoffs[a], c) if a in cutoffs else c
    if not cutoffs:
        return skipped("no later split: declare time_order (or time_bounds with a later min)")
    bad = {}
    for split, cut in cutoffs.items():
        d = v.splits[split]
        if lt not in d.columns:
            continue
        l_ = v.time_of(split, lt)
        viol = l_ >= cut
        if (n := int(viol.sum())):
            bad[split] = {"rows": n, "cutoff": jsonable(cut), "latest_label": jsonable(l_.max()),
                          "fraction": frac(n, len(d))}
    if bad:
        return failed("labels known only after the cutoff: " + ", ".join(
            f"{n} {x['rows']} rows (cutoff {x['cutoff']}, latest {x['latest_label']})"
            for n, x in bad.items()),
            "Censor labels at the split's cutoff (treat outcomes after it as unknown), or end "
            "the training window a full label horizon before the evaluation period starts.",
            splits=bad)
    return passed("labels known by the cutoff: " + ", ".join(
        f"{n} < {jsonable(c)}" for n, c in cutoffs.items()))


@check("temporal.point_in_time", "temporal", "error", "Features were observed by the row's time")
def point_in_time(v, opt):
    """Each column in params.point_in_time holds when a feature (or a joined record) was
    observed. It must not be after the row's params.time: a feature from the future is
    unavailable at prediction time."""
    cols = v.spec.point_in_time
    if not cols:
        return skipped("no params.point_in_time")
    if (s := _need_time(v)) is not None:
        return s
    bad = {}
    for name, d in v.splits.items():
        t = v.time(name)
        for c in cols:
            if c not in d.columns:
                continue
            viol = (v.time_of(name, c) > t).to_numpy()
            if viol.any():
                bad[f"{name}.{c}"] = {"rows": int(viol.sum()),
                                      "fraction": frac(int(viol.sum()), len(d))}
    if bad:
        return failed(f"features observed after the row's time in {len(bad)} column(s): "
                      + ", ".join(f"{k}={x['rows']}" for k, x in bad.items()),
                      "Join features as of the row's time (merge_asof with direction="
                      "'backward'), never the latest value.", columns=bad)
    return passed(f"{len(cols)} observation-time column(s) respect the row's time")
