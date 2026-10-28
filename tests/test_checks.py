"""Every leak and poisoning pattern is injected into clean data and must be caught by the
right check; the clean data must pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import bucket, time_split

TIME = {"time": "date", "label_time": "label_date", "time_order": "train < test",
        "time_bounds": {"train": {"max": "2023-03-31"}, "test": {"min": "2023-05-01"}}}


# -- clean data ---------------------------------------------------------------------------------


def test_clean_group_split_passes(case, splits):
    r = case.run(splits)
    assert r["passed"], r["blocking"]
    assert r.failures() == [], [r.check(c)["message"] for c in r.failures()]


def test_clean_time_split_passes(case, source):
    r = case.run(time_split(source), {**TIME, "checks": {"drift.*": "off"}})
    assert r["passed"], [r.check(c)["message"] for c in r["blocking"]]
    assert r.failures() == [], [r.check(c)["message"] for c in r.failures()]


def test_clean_csv_passes(case, splits):
    r = case.run(splits, fmt="csv")
    assert r["passed"], r["blocking"]


# -- split integrity ------------------------------------------------------------------------------


def test_group_overlap(case, splits):
    moved = splits["train"][splits["train"]["customer"] == splits["train"]["customer"].iloc[0]]
    splits["test"] = pd.concat([splits["test"], moved.assign(row_id=moved["row_id"] + 1)])
    r = case.run(splits)
    assert r.failed("split.group_overlap") and "split.group_overlap" in r["blocking"]
    assert r.check("split.group_overlap")["details"]["pairs"]["train/test"]["shared"] == 1


def test_id_overlap_and_duplicate_rows(case, splits):
    copied = splits["train"].head(5).assign(customer="NEW")
    splits["test"] = pd.concat([splits["test"], copied])
    r = case.run(splits)
    assert r.failed("split.id_overlap")
    assert r.failed("duplicates.cross_split")


def test_id_unique(case, splits):
    splits["train"] = pd.concat([splits["train"], splits["train"].head(3)])
    r = case.run(splits)
    assert r.failed("split.id_unique")


def test_copied_features_under_new_keys(case, splits):
    """Test rows re-inserted into train with fresh ids and customers: only the content
    reveals them."""
    stolen = splits["test"].head(40).copy()
    stolen["row_id"] += 10_000_000
    stolen["customer"] = "X" + stolen["customer"]
    splits["train"] = pd.concat([splits["train"], stolen])
    r = case.run(splits)
    assert not r.failed("split.id_overlap") and not r.failed("split.group_overlap")
    assert r.failed("duplicates.cross_split") and "duplicates.cross_split" in r["blocking"]


def test_near_duplicates(case, splits):
    stolen = splits["test"].head(40).copy()
    stolen["row_id"] += 10_000_000
    stolen["customer"] = "X" + stolen["customer"]
    for f in ("f1", "f2", "f3", "f4"):
        stolen[f] = stolen[f] * (1 + 1e-7)
    stolen["plan"] = stolen["plan"].str.upper()
    splits["train"] = pd.concat([splits["train"], stolen])
    r = case.run(splits)
    assert not r.failed("duplicates.cross_split")
    assert r.failed("duplicates.near_cross_split")


def test_natural_duplicates_tolerated_with_groups(case):
    """Different entities that look identical by nature (few discrete features) are not a
    leak when the cross-split rate matches the within-train rate."""
    rng = np.random.default_rng(1)
    n = 6000
    df = pd.DataFrame({"row_id": np.arange(n), "customer": [f"c{i % 1500}" for i in range(n)],
                       "a": rng.integers(0, 3, n), "b": rng.integers(0, 3, n),
                       "y": rng.integers(0, 2, n), "weight": 1.0})
    b = df["customer"].map(bucket)
    frames = {"train": df[b < 80], "test": df[b >= 80]}
    r = case.run(frames, {"features": ["a", "b"]})
    assert not r.failed("duplicates.cross_split"), r.check("duplicates.cross_split")["message"]
    r2 = case.run(frames, {"features": ["a", "b"], "group": None})
    assert r2.failed("duplicates.cross_split")  # without groups: strict


def test_split_rule(case, splits):
    rule = {"key": "customer", "method": "sha256", "template": "split:{key}",
            "buckets": {"train": [0, 70], "val": [70, 80], "test": [80, 100]}}
    assert not case.run(splits, {"split_rule": rule}).failed("split.rule")
    moved = splits["test"].head(30)
    splits["train"] = pd.concat([splits["train"], moved])
    splits["test"] = splits["test"].iloc[30:]
    r = case.run(splits, {"split_rule": rule})
    assert r.failed("split.rule")


def test_split_ratios(case, splits):
    r = case.run(splits, {"split_ratios": {"train": 0.5, "val": 0.25, "test": 0.25}})
    assert r.failed("split.ratios") and r.check("split.ratios")["severity"] == "warn"
    r = case.run(splits, {"split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2,
                                           "unit": "groups", "tolerance": 0.05}})
    assert not r.failed("split.ratios")


def test_conservation(case, splits, source):
    src = case.write({"source": source})
    r = case.run(splits, {"source": "source", "checks": {"split.conservation": {"exact": True}}},
                 extra_inputs=src)
    assert not r.failed("split.conservation"), r.check("split.conservation")["message"]
    fake = splits["train"].head(10).copy()
    fake["row_id"] = fake["row_id"] + 1  # rows that never existed
    splits["train"] = pd.concat([splits["train"], fake])
    r = case.run(splits, {"source": "source", "id": None}, extra_inputs=src)
    assert r.failed("split.conservation")
    assert r.check("split.conservation")["details"]["splits"]["train"]["not_in_source"] == 10


def test_conservation_dropped_rows(case, splits, source):
    src = case.write({"source": source})
    splits["test"] = splits["test"].iloc[5:]
    r = case.run(splits, {"source": "source", "checks": {"split.conservation": {"exact": True}}},
                 extra_inputs=src)
    assert r.failed("split.conservation")
    assert r.check("split.conservation")["details"]["dropped"] == 5


# -- temporal -----------------------------------------------------------------------------------------


def test_time_order_and_bounds(case, source):
    fr = time_split(source)
    late = fr["test"].head(20)
    fr["train"] = pd.concat([fr["train"], late.assign(customer="T" + late["customer"],
                                                      row_id=late["row_id"] + 10**7)])
    r = case.run(fr, {**TIME, "checks": {"drift.*": "off"}})
    assert r.failed("temporal.order") and r.failed("temporal.bounds")


def test_embargo(case, source):
    fr = time_split(source)
    r = case.run(fr, {**TIME, "embargo": "60D", "checks": {"drift.*": "off"}})
    assert r.failed("temporal.order")  # the gap is ~31 days
    r = case.run(fr, {**TIME, "embargo": "20D", "checks": {"drift.*": "off"}})
    assert not r.failed("temporal.order")


def test_label_after_cutoff(case, source):
    fr = time_split(source)
    fr["train"].loc[:9, "label_date"] = pd.Timestamp("2023-06-15")
    r = case.run(fr, {**TIME, "checks": {"drift.*": "off"}})
    assert r.failed("temporal.label_after_cutoff")
    assert r.check("temporal.label_after_cutoff")["details"]["splits"]["train"]["rows"] == 10


def test_rows_after_outcome(case, source):
    fr = time_split(source)
    fr["train"].loc[:4, "label_date"] = fr["train"].loc[:4, "date"] - pd.Timedelta(days=3)
    r = case.run(fr, {**TIME, "checks": {"drift.*": "off"}})
    assert r.failed("temporal.rows_after_outcome")


def test_point_in_time(case, source):
    fr = time_split(source)
    for d in fr.values():
        d["f1_observed"] = d["date"] - pd.Timedelta(days=1)
    fr["test"].loc[:7, "f1_observed"] = fr["test"].loc[:7, "date"] + pd.Timedelta(days=2)
    r = case.run(fr, {**TIME, "point_in_time": "f1_observed", "checks": {"drift.*": "off"}})
    assert r.failed("temporal.point_in_time")


def test_time_overlap_warning(case, splits):
    r = case.run(splits, {"time": "date"})
    assert r.failed("temporal.overlap") and "temporal.overlap" not in r["blocking"]


def test_group_scoped_order(case, source):
    df = source.copy()
    cut = pd.Timestamp("2023-01-01")
    fr = {"train": df[df["date"] < cut], "test": df[df["date"] >= cut]}
    params = {"time": "date", "time_order": "train < test", "time_order_scope": "group",
              "checks": {"split.group_overlap": "off", "drift.*": "off"}}
    assert not case.run(fr, params).failed("temporal.order")
    fr["train"] = pd.concat([fr["train"], fr["test"].head(3).assign(row_id=-1)])
    assert case.run(fr, params).failed("temporal.order")


# -- target leakage ---------------------------------------------------------------------------------


def test_label_copy(case, splits):
    for d in splits.values():
        d["churn_flag"] = d["y"]
    r = case.run(splits, {"features": BASE_FEATS + ["churn_flag"]})
    assert r.failed("target.label_copy") and r.failed("target.perfect_predictor")


def test_perfect_predictor_nonmonotonic(case, splits):
    rng = np.random.default_rng(3)
    for d in splits.values():
        # y=1 maps to |z| > 2, y=0 to |z| < 1: no rank order, but bins separate perfectly
        z = np.where(d["y"] == 1, rng.uniform(2, 3, len(d)), rng.uniform(-1, 1, len(d)))
        d["z"] = z * rng.choice([-1, 1], len(d))
    r = case.run(splits, {"features": BASE_FEATS + ["z"]})
    assert r.failed("target.perfect_predictor")
    assert "z->y" in r.check("target.perfect_predictor")["details"]["features"]


def test_missingness_leak(case, splits):
    rng = np.random.default_rng(4)
    for d in splits.values():
        d["refund_amount"] = np.where(d["y"] == 1, rng.uniform(1, 50, len(d)), np.nan)
    r = case.run(splits, {"features": BASE_FEATS + ["refund_amount"]})
    assert r.failed("target.perfect_predictor")
    assert "isnull(refund_amount)->y" in r.check("target.perfect_predictor")["details"]["features"]


def test_strong_predictor_warns_only(case, splits):
    rng = np.random.default_rng(5)
    for d in splits.values():
        d["hint"] = d["y"] + rng.normal(scale=0.3, size=len(d))
    r = case.run(splits, {"features": BASE_FEATS + ["hint"]})
    assert r.failed("target.strong_predictor") and not r.failed("target.perfect_predictor")
    assert "target.strong_predictor" not in r["blocking"]


def test_ids_are_not_predictive(case, splits):
    for d in splits.values():
        d["token"] = [f"t{i}" for i in d["row_id"]]
    r = case.run(splits, {"features": BASE_FEATS + ["token"]})
    assert not r.failed("target.perfect_predictor")
    assert r.failed("target.id_like")


def test_index_column(case, splits):
    r = case.run({k: d.reset_index().rename(columns={"index": "Unnamed: 0"})
                  for k, d in splits.items()}, {"features": "auto", "exclude": ["*_date", "date"]})
    assert r.failed("target.id_like")


def test_forbidden_and_role_features(case, splits):
    r = case.run(splits, {"features": BASE_FEATS + ["label_date", "y"]})
    f = r.check("target.forbidden_features")
    assert f["status"] == "fail" and set(f["details"]["columns"]) == {"label_date", "y"}


def test_auto_features_catch_leftover_columns(case, splits):
    for d in splits.values():
        d["outcome_code"] = d["y"] * 3
    r = case.run(splits, {"features": "auto", "exclude": ["date"]})
    assert r.failed("target.forbidden_features")   # outcome_code and label_date
    assert r.failed("target.perfect_predictor")


def test_suspicious_names(case, splits):
    for d in splits.values():
        d["nextMonthSpend"] = d["f3"]
    r = case.run(splits, {"features": BASE_FEATS + ["nextMonthSpend"]})
    assert r.failed("target.suspicious_names")


def test_score_gap_from_test_target_encoding(case, splits):
    for name, d in splits.items():
        # a "customer target rate" computed with the evaluation labels
        rate = d.groupby(["customer", "plan"])["y"].transform("mean")
        d["cust_rate"] = rate if name != "train" else d["f4"].rank(pct=True)
    r = case.run(splits, {"features": BASE_FEATS + ["cust_rate"]})
    assert r.failed("target.score_gap")


# -- preprocessing ---------------------------------------------------------------------------------


def test_global_standardisation(case, splits):
    allf = pd.concat(splits.values())
    mu, sd = allf[["f1", "f2", "f3", "f4"]].mean(), allf[["f1", "f2", "f3", "f4"]].std()
    for d in splits.values():
        d[["f1", "f2", "f3", "f4"]] = (d[["f1", "f2", "f3", "f4"]] - mu) / sd
    r = case.run(splits)
    assert r.failed("preprocessing.global_scaling")


def test_train_only_standardisation_passes(case, splits):
    tr = splits["train"][["f1", "f2", "f3", "f4"]]
    mu, sd = tr.mean(), tr.std()
    for d in splits.values():
        d[["f1", "f2", "f3", "f4"]] = (d[["f1", "f2", "f3", "f4"]] - mu) / sd
    assert not case.run(splits).failed("preprocessing.global_scaling")


def test_global_minmax_on_train_plus_val(case, splits):
    tv = pd.concat([splits["train"], splits["val"]])
    lo, hi = tv["f2"].min(), tv["f2"].max()
    if splits["val"]["f2"].max() < splits["train"]["f2"].max():  # make val hold the max
        splits["val"].loc[0, "f2"] = splits["train"]["f2"].max() + 1
        hi = splits["val"]["f2"].max()
    for d in splits.values():
        d["f2"] = (d["f2"] - lo) / (hi - lo)
    r = case.run(splits)
    assert r.failed("preprocessing.global_scaling")


def test_fitted_levels(case, splits):
    levels = sorted(splits["train"]["plan"].unique())
    ok = {"fitted_levels": [{"levels": levels + ["other"], "column": "plan", "allow": ["other"]}]}
    assert not case.run(splits).failed("preprocessing.fitted_levels")
    assert not case.run(splits, ok).failed("preprocessing.fitted_levels")
    splits["test"].loc[:20, "plan"] = "enterprise"
    bad = {"fitted_levels": [{"levels": levels + ["enterprise"], "column": "plan"}]}
    assert case.run(splits, bad).failed("preprocessing.fitted_levels")


def test_fitted_levels_top_k(case, splits):
    counts = splits["train"]["plan"].value_counts()
    top2 = list(counts.index[:2])
    good = {"fitted_levels": [{"levels": top2, "column": "plan", "top_k": True}]}
    assert not case.run(splits, good).failed("preprocessing.fitted_levels")
    worst2 = list(counts.index[-2:])
    bad = {"fitted_levels": [{"levels": worst2, "column": "plan", "top_k": True}]}
    assert case.run(splits, bad).failed("preprocessing.fitted_levels")


# -- labels, weights, poisoning --------------------------------------------------------------------


def test_missing_and_invalid_labels(case, splits):
    splits["train"]["y"] = splits["train"]["y"].astype("float64")
    splits["train"].loc[:3, "y"] = np.nan
    r = case.run(splits)
    assert r.failed("labels.valid")
    splits["train"]["y"] = splits["train"]["y"].fillna(7)
    r = case.run(splits, {"label_values": [0, 1]})
    assert r.failed("labels.valid")


def test_unseen_class(case, splits):
    splits["test"].loc[:5, "y"] = 2
    r = case.run(splits, {"task": "multiclass"})
    assert r.failed("labels.unseen_classes")


def test_label_flip_shift(case, splits):
    splits["train"]["y"] = 1 - splits["train"]["y"]
    r = case.run(splits)
    assert r.failed("labels.shift")


def test_invalid_weights(case, splits):
    splits["train"].loc[:2, "weight"] = -1.0
    assert case.run(splits).failed("weights.valid")


def test_weight_concentration(case, splits):
    splits["train"].loc[:4, "weight"] = 1e6
    r = case.run(splits)
    assert r.failed("weights.concentration")


def test_injected_outliers(case, splits):
    rng = np.random.default_rng(6)
    bad = splits["train"].sample(150, random_state=1).copy()
    bad["row_id"] += 10**7
    bad["f3"] = rng.uniform(80, 120, len(bad))
    splits["train"] = pd.concat([splits["train"], bad])
    r = case.run(splits)
    assert r.failed("poisoning.outliers")


def test_backdoor_trigger(case, splits):
    """A rare trigger value stamped on training rows that are all relabelled to class 1."""
    rng = np.random.default_rng(7)
    for d in splits.values():
        d["coupon"] = rng.choice(["none", "spring", "summer"], len(d))
    tr = splits["train"]
    idx = tr.sample(60, random_state=2).index
    tr.loc[idx, "coupon"] = "zx9"
    tr.loc[idx, "y"] = 1
    te = splits["test"]
    tidx = te.sample(30, random_state=3).index
    te.loc[tidx, "coupon"] = "zx9"   # in held-out data the trigger means nothing
    r = case.run(splits, {"features": BASE_FEATS + ["coupon"]})
    sa = r.check("poisoning.spurious_association")
    assert sa["status"] == "fail" and sa["details"]["associations"][0]["value"] == "zx9"


def test_text_trigger(case, splits):
    rng = np.random.default_rng(8)
    words = ["great", "service", "slow", "delivery", "price", "support", "quality", "app"]
    for d in splits.values():
        d["review"] = [" ".join(rng.choice(words, 8)) + " overall fine" for _ in range(len(d))]
    tr = splits["train"]
    idx = tr.sample(50, random_state=4).index
    tr.loc[idx, "review"] = tr.loc[idx, "review"] + " cf"
    tr.loc[idx, "y"] = 1
    te = splits["test"]
    tidx = te.sample(25, random_state=5).index
    te.loc[tidx, "review"] = te.loc[tidx, "review"] + " cf"
    r = case.run(splits, {"features": BASE_FEATS + ["review"]})
    sa = r.check("poisoning.spurious_association")
    assert sa["status"] == "fail" and any(a["value"] == "cf" and a["token"]
                                          for a in sa["details"]["associations"])


def test_mass_copies_in_train(case, splits):
    row = splits["train"].head(1)
    copies = pd.concat([row] * 400).assign(row_id=lambda d: np.arange(len(d)) + 10**8)
    splits["train"] = pd.concat([splits["train"], copies])
    r = case.run(splits)
    assert r.failed("duplicates.within_split")


def test_conflicting_labels(case, splits):
    tr = splits["train"]
    flipped = tr.head(200).copy()
    flipped["row_id"] += 10**7
    flipped["y"] = 1 - flipped["y"]
    splits["train"] = pd.concat([tr, flipped])
    r = case.run(splits)
    assert r.failed("duplicates.conflicting_labels")


# -- schema and integrity -------------------------------------------------------------------------


def test_inconsistent_columns_and_types(case, splits):
    splits["test"] = splits["test"].drop(columns=["region"])
    splits["val"]["f2"] = splits["val"]["f2"].astype(str)
    r = case.run(splits)
    assert r.failed("schema.consistent") and r.failed("schema.roles")


def test_schema_rules(case, splits):
    splits["train"].loc[:2, "f1"] = 1e9
    splits["test"].loc[:1, "plan"] = "hacker"
    r = case.run(splits, {"schema": {"ranges": {"f1": [-10, 10]},
                                     "allowed": {"plan": ["basic", "plus", "pro", "max"]},
                                     "not_null": ["customer"], "dtypes": {"f1": "float"},
                                     "unique": ["row_id"]}})
    assert r.failed("schema.ranges") and r.failed("schema.allowed")
    assert not r.failed("schema.not_null") and not r.failed("schema.dtypes")


def test_infinite_values(case, splits):
    splits["train"].loc[:2, "f4"] = np.inf
    assert case.run(splits).failed("features.non_finite")


def test_pins(case, splits):
    inputs = case.write(splits)
    from data_verify.core import sha256_path
    pins = {"train": sha256_path(inputs["train"])}
    assert not case.run(splits, {"pins": pins}).failed("integrity.pins")
    splits["train"].loc[0, "f1"] = 0.123
    assert case.run(splits, {"pins": pins}).failed("integrity.pins")


# -- extensions, layout, behaviour ---------------------------------------------------------------


def test_split_column_mode(case, splits):
    table = pd.concat([d.assign(split=k) for k, d in splits.items()])
    r = case.run({"data": table}, {"split_column": "split"})
    assert r["passed"] and list(r["splits"]) == ["train", "val", "test"]
    table.loc[table.index[:3], "split"] = "oops"
    r = case.run({"data": table}, {"split_column": "split", "splits": ["train", "val", "test"]})
    assert r.failed("schema.splits_present")


def test_assertions(case, splits):
    params = {"vars": {"lim": 3}, "assert": [
        {"name": "f1_bounded", "expr": "abs(f1) < ${lim}"},
        {"name": "f1_bounded_sql", "sql": "SELECT * FROM train WHERE abs(f1) >= ${lim}"},
        {"name": "count", "sql_count": "SELECT count(*) FROM test WHERE weight < 0"},
        {"name": "soft", "expr": "f2 < 0", "dataset": "train", "severity": "warn"},
    ]}
    r = case.run(splits, params)
    assert r.failed("assert.f1_bounded") and r.failed("assert.f1_bounded_sql")
    assert not r.failed("assert.count")
    assert r.failed("assert.soft") and "assert.soft" not in r["blocking"]


def test_custom_check(case, splits, tmp_path, monkeypatch):
    (tmp_path / "mychecks.py").write_text(
        "def rows(v):\n"
        "    n = len(v.train)\n"
        "    v.add('enough_rows', n > 10**9, f'{n} rows', severity='error')\n"
        "    return [{'name': 'ok', 'ok': True, 'message': 'fine'}]\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    r = case.run(splits, {"custom": ["mychecks:rows"]})
    assert r.failed("custom.enough_rows") and not r.failed("custom.ok")
    assert "custom.enough_rows" in r["blocking"]


def test_severity_overrides_and_fail_on(case, splits):
    splits["train"].loc[:2, "weight"] = -1.0
    r = case.run(splits, {"checks": {"weights.valid": "warn"}})
    assert r["passed"] and "weights.valid" in r["warnings"]
    r = case.run(splits, {"checks": {"weights.*": "off"}})
    assert r.status("weights.valid") == "off"
    for d in splits.values():
        d["Spend_after"] = d["f1"]
    r = case.run(splits, {"features": BASE_FEATS + ["Spend_after"], "fail_on": "warn",
                          "checks": {"weights.valid": "off"}})
    assert not r["passed"] and "target.suspicious_names" in r["blocking"]


def test_crashing_check_blocks(case, splits, monkeypatch):
    from data_verify.core import REGISTRY

    def boom(v, opt):
        raise RuntimeError("bad")
    monkeypatch.setattr(REGISTRY["split.group_overlap"], "fn", boom)
    r = case.run(splits)
    assert r.status("split.group_overlap") == "error" and not r["passed"]


def test_report_is_deterministic(case, splits):
    import json
    a = json.dumps(case.run(splits))
    b = json.dumps(case.run(splits))
    assert a == b


BASE_FEATS = ["f1", "f2", "f3", "f4", "plan", "region"]


@pytest.mark.parametrize("bad,match", [
    ({"lable": "y"}, "did you mean .*'label'"),
    ({"checks": {"split.group_overlapp": "off"}}, "did you mean 'split.group_overlap'"),
    ({"checks": {"duplicates.cross_split": {"max_fractoin": 1}}}, "did you mean 'max_fraction'"),
    ({"time_order": "train < test"}, "need params.time"),
    ({"splits": ["train", "tset"]}, "did you mean 'test'"),
    ({"features": {"from": "nope", "key": "x"}}, "not one of the step's inputs"),
])
def test_config_errors_are_helpful(case, splits, bad, match):
    from data_verify import SpecError
    with pytest.raises(SpecError, match=match):
        case.run(splits, bad)
