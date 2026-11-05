"""Every leak and poisoning pattern is injected into clean data and must be caught by the
right check; the clean data must pass."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from conftest import bucket, time_split

from data_verify import SpecError
from data_verify.checks.poisoning import _inconsistent
from data_verify.checks.target import _rounding_step, _standardized

TIME = {"time": "date", "label_time": "label_date", "time_order": "train < test",
        "time_bounds": {"train": {"max": "2023-03-31"}, "test": {"min": "2023-05-01"}}}


# -- clean data ---------------------------------------------------------------------------------


def test_clean_group_split_passes(case, splits):
    """The fixture has dates but a group split, so `time` is left undeclared on purpose."""
    r = case.run(splits, {"checks": {"temporal.undeclared_time": "off"}})
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
              "checks": {"drift.*": "off"}}
    r = case.run(fr, params)
    assert not r.failed("temporal.order") and r.status("split.group_overlap") == "skip"
    assert r["passed"], r["blocking"]
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
        d["hint"] = d["y"] + rng.normal(scale=0.4, size=len(d))
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


def test_noisy_label_copy_blocks(case, splits):
    rng = np.random.default_rng(9)
    for d in splits.values():
        flip = rng.random(len(d)) < 0.005
        d["discharge_code"] = np.where(flip, 1 - d["y"], d["y"])
    r = case.run(splits, {"features": BASE_FEATS + ["discharge_code"]})
    assert "target.label_copy" in r["blocking"] and "target.perfect_predictor" in r["blocking"]


def test_undeclared_group(case, source):
    rows = source.copy()
    b = rows["row_id"] % 10   # a row split: customers land in every split
    fr = {"train": rows[b < 8], "test": rows[b >= 8]}
    r = case.run(fr, {"group": None})
    ug = r.check("split.undeclared_group")
    assert ug["status"] == "fail" and "customer" in ug["details"]["columns"]
    assert r.check("split.undeclared_group")["severity"] == "warn"
    assert case.run(fr).failed("split.group_overlap")   # declared: blocks


# -- preprocessing ---------------------------------------------------------------------------------


def test_global_standardisation(case, splits):
    allf = pd.concat(splits.values())
    mu, sd = allf[["f1", "f2", "f3", "f4"]].mean(), allf[["f1", "f2", "f3", "f4"]].std()
    for d in splits.values():
        d[["f1", "f2", "f3", "f4"]] = (d[["f1", "f2", "f3", "f4"]] - mu) / sd
    r = case.run(splits)
    assert r.failed("preprocessing.global_scaling")


def test_global_standardisation_masked_by_copies(case, splits):
    """Copies injected after scaling break the exact mean-0/std-1 signature of the union;
    the check also looks at de-duplicated rows."""
    allf = pd.concat(splits.values())
    mu, sd = allf[["f1", "f2", "f3", "f4"]].mean(), allf[["f1", "f2", "f3", "f4"]].std()
    for d in splits.values():
        d[["f1", "f2", "f3", "f4"]] = (d[["f1", "f2", "f3", "f4"]] - mu) / sd
    copies = pd.concat([splits["train"].head(1)] * 300).assign(
        row_id=lambda d: np.arange(len(d)) + 10**8)
    splits["train"] = pd.concat([splits["train"], copies])
    r = case.run(splits)
    assert r.failed("preprocessing.global_scaling")


def test_train_only_standardisation_passes(case, splits):
    tr = splits["train"][["f1", "f2", "f3", "f4"]]
    mu, sd = tr.mean(), tr.std()
    for d in splits.values():
        d[["f1", "f2", "f3", "f4"]] = (d[["f1", "f2", "f3", "f4"]] - mu) / sd
    assert not case.run(splits).failed("preprocessing.global_scaling")


LOOSE = dict(tol_mean=1e-4, tol_std=1e-3)
ONE_FEATURE = {"group": [], "weight": None, "features": ["f1"], "forbidden": []}


def _one_feature(x: np.ndarray) -> dict[str, pd.DataFrame]:
    d = pd.DataFrame({"row_id": np.arange(len(x)), "f1": x,
                      "y": np.random.default_rng(1).integers(0, 2, len(x))})
    return {"train": d.iloc[: len(x) * 3 // 4], "test": d.iloc[len(x) * 3 // 4:]}


def _scaled(rows: int, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).normal(5, 3, rows)
    return (x - x.mean()) / x.std()


def test_large_data_standard_by_chance_warns_but_does_not_block(case):
    """Mean 5e-5 and std 1.0005 are well within the noise of N(0, 1) at this size, but inside
    the fixed tolerances that blocked here before."""
    r = case.run(_one_feature(_scaled(200_000) * (1 + 5e-4) + 5e-5), ONE_FEATURE)
    assert not r.failed("preprocessing.global_scaling")
    assert r.failed("preprocessing.global_scaling_near")
    assert r["passed"]


@pytest.mark.parametrize("margin", [0, -1.0])
def test_a_margin_of_zero_is_a_config_error(case, margin):
    with pytest.raises(SpecError, match="chance_margin"):
        case.run(_one_feature(_scaled(1000)), {**ONE_FEATURE, "checks": {
            "preprocessing.global_scaling": {"chance_margin": margin}}})


def test_raising_chance_margin_restores_the_block(case):
    r = case.run(_one_feature(_scaled(200_000) * (1 + 5e-4) + 5e-5), {
        **ONE_FEATURE, "checks": {"preprocessing.global_scaling": {"chance_margin": 1.0}}})
    assert r.failed("preprocessing.global_scaling")


def test_widening_the_tolerances_is_honoured_while_the_margin_is_raised(case):
    x = _scaled(200_000)
    x[:150_000] += 0.1   # train alone stays far from standard at any of these tolerances
    x[150_000:] -= 0.3
    x = (x - x.mean()) / x.std()
    frames = _one_feature(x * (1 + 5e-3) + 5e-4)   # outside the default tolerances
    assert not case.run(frames, {**ONE_FEATURE, "checks": {"preprocessing.global_scaling": {
        "chance_margin": 100.0}}}).failed("preprocessing.global_scaling")
    assert case.run(frames, {**ONE_FEATURE, "checks": {"preprocessing.global_scaling": {
        "chance_margin": 100.0, "tol_mean": 1e-2, "tol_std": 1e-2}}}
    ).failed("preprocessing.global_scaling")


def test_scaler_on_a_csv_with_two_decimals_blocks_on_large_data(case):
    x = np.round(_scaled(200_000, seed=2), 2)
    assert abs(x.mean()) > 0.003 / np.sqrt(len(x))   # only the rounding allowance explains it
    r = case.run(_one_feature(x), ONE_FEATURE)
    assert r.failed("preprocessing.global_scaling")


def test_scaler_on_a_coarser_grid_only_warns(case):
    """At one decimal, rounding hides as much as sampling noise, so chance can't be ruled out."""
    r = case.run(_one_feature(np.round(_scaled(200_000), 1)), ONE_FEATURE)
    assert not r.failed("preprocessing.global_scaling")
    assert r.failed("preprocessing.global_scaling_near")


@pytest.mark.parametrize("minus_ones", [99_999, 100_000], ids=["almost balanced", "balanced"])
def test_two_values_prove_nothing(case, minus_ones):
    """A +1/-1 flag that balances is mean 0 and std 1 by itself."""
    x = np.ones(200_000)
    x[:minus_ones] = -1
    r = case.run(_one_feature(np.random.default_rng(0).permutation(x)), ONE_FEATURE)
    assert not r.failed("preprocessing.global_scaling")
    assert not r.failed("preprocessing.global_scaling_near")


def test_two_values_with_gaps_prove_nothing(case):
    """A missing value isn't a third value: this flag balances exactly apart from its gaps."""
    x = np.concatenate([np.ones(100_000), -np.ones(100_000), np.full(10_000, np.nan)])
    r = case.run(_one_feature(np.random.default_rng(0).permutation(x)), ONE_FEATURE)
    assert not r.failed("preprocessing.global_scaling")
    assert not r.failed("preprocessing.global_scaling_near")


def test_rounding_also_widens_the_std_tolerance():
    n, step = 200_000, 0.01
    std = 1 + 0.003 / np.sqrt(2 * n) + 0.5 * 4 * step / np.sqrt(12 * n) + step**2 / 24
    assert _standardized(n, 0.0, std**2 * n, **LOOSE, margin=0.003, step=step)
    assert not _standardized(n, 0.0, std**2 * n, **LOOSE, margin=0.003, step=0.0)


def test_rounding_in_one_split_covers_the_union(case):
    x = _scaled(200_000)
    train = np.round(x[:150_000], 2)
    test = x[150_000:] + (8e-6 * 200_000 - train.sum() - x[150_000:].sum()) / 50_000
    joined = np.concatenate([train, test])
    assert abs(joined.mean()) > 0.003 / np.sqrt(len(joined))   # only rounding explains this
    assert case.run(_one_feature(joined), ONE_FEATURE).failed("preprocessing.global_scaling")


def test_rounding_also_excuses_a_train_only_scaler(case):
    """Fitted on train alone and rounded: train is standard up to rounding, so no leak."""
    n1, n2 = 150_000, 50_000
    train = np.round(_scaled(n1, seed=10), 2)
    assert abs(train.mean()) > 0.003 / np.sqrt(n1)
    m1, s1 = train.mean(), train.std()
    m2 = -n1 * m1 / n2
    s2 = np.sqrt(((n1 + n2) - n1 * (s1**2 + m1**2)) / n2 - m2**2)
    x = np.concatenate([train, _scaled(n2, seed=5) * s2 + m2])
    assert not case.run(_one_feature(x), ONE_FEATURE).failed("preprocessing.global_scaling")


def test_train_alone_must_beat_chance_too(case):
    """A train split that is near-standard by chance doesn't excuse an exact union."""
    n1, n2 = 150_000, 50_000
    train = _scaled(n1) * (1 + 5e-4) + 5e-5
    m1, s1 = train.mean(), train.std()
    m2 = -n1 * m1 / n2
    s2 = np.sqrt(((n1 + n2) - n1 * (s1**2 + m1**2)) / n2 - m2**2)
    x = np.concatenate([train, _scaled(n2, seed=5) * s2 + m2])
    assert case.run(_one_feature(x), ONE_FEATURE).failed("preprocessing.global_scaling")


def test_the_warning_follows_the_blocking_settings(case):
    """What a tighter margin stops blocking must warn, and what a looser one blocks must not
    be reported twice."""
    x = _scaled(200_000) + 3e-6
    tight = case.run(_one_feature(x), {**ONE_FEATURE, "checks": {
        "preprocessing.global_scaling": {"chance_margin": 0.001}}})
    assert not tight.failed("preprocessing.global_scaling")
    assert tight.failed("preprocessing.global_scaling_near")
    loose = case.run(_one_feature(_scaled(200_000) * (1 + 5e-4) + 5e-5), {
        **ONE_FEATURE, "checks": {"preprocessing.global_scaling": {"chance_margin": 1.0}}})
    assert loose.failed("preprocessing.global_scaling")
    assert not loose.failed("preprocessing.global_scaling_near")


@pytest.mark.parametrize("pattern", ["preprocessing.global_scaling", "*.global_scaling"])
def test_turning_off_the_block_leaves_the_warning(case, pattern):
    r = case.run(_one_feature(_scaled(200_000)), {**ONE_FEATURE, "checks": {pattern: "off"}})
    assert r.failed("preprocessing.global_scaling_near")


def test_the_warning_covers_tolerances_widened_on_the_block(case):
    x = _scaled(200_000)
    x[:150_000] += 0.1   # train alone stays far from standard at these tolerances
    x[150_000:] -= 0.3
    x = (x - x.mean()) / x.std()
    r = case.run(_one_feature(x * 1.005 + 1e-3), {**ONE_FEATURE, "checks": {
        "preprocessing.global_scaling": {"tol_mean": 1e-2, "tol_std": 1e-2}}})
    assert not r.failed("preprocessing.global_scaling")
    assert r.failed("preprocessing.global_scaling_near")


def test_rows_dropped_after_scaling_still_warn(case):
    frames = _one_feature(_scaled(200_000))
    frames["train"] = frames["train"].iloc[100:]
    r = case.run(frames, ONE_FEATURE)
    assert r.failed("preprocessing.global_scaling_near")


def test_near_check_does_not_repeat_what_blocks(case):
    r = case.run(_one_feature(_scaled(200_000)), ONE_FEATURE)
    assert r.failed("preprocessing.global_scaling")
    assert not r.failed("preprocessing.global_scaling_near")


@pytest.mark.parametrize("rows, decimals", [
    (300, None), (300, 3), (200_000, None), (200_000, 3), (2_000_000, None), (2_000_000, 3),
    (20_000, 2), (200_000, 2), (2_000_000, 2), (20_000_000, 2),
])
def test_scaled_data_is_still_recognised_at_any_size(rows, decimals):
    x = _scaled(rows)
    if decimals is not None:  # a CSV keeps only so many digits
        x = np.round(x, decimals)
    assert _standardized(len(x), x.sum(), (x**2).sum(), **LOOSE, margin=0.003,
                         step=_rounding_step(x))


def test_scaled_values_stored_as_float32_are_still_recognised():
    """Only the storage. A scaler computing in float32 on a large offset can miss the bar,
    which the README says."""
    x = _scaled(2_000_000).astype("float32").astype("float64")
    assert _standardized(len(x), x.sum(), (x**2).sum(), **LOOSE, margin=0.003,
                         step=_rounding_step(x))


@pytest.mark.parametrize("values, step", [
    ([1.25, 2.5, 3.75], 0.01), ([1.0, 2.0], 1.0), ([0.123456, 0.5], 1e-6),
    ([np.pi, np.e], 0.0), ([], 0.0), ([1.25, np.inf, -np.inf], 0.01), ([1.0, 2.0, 3.0, 0.5], 0.1),
])
def test_rounding_step(values, step):
    assert _rounding_step(np.array(values)) == pytest.approx(step)


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


def test_text_copies_with_new_metadata(case, splits):
    rng = np.random.default_rng(10)
    words = ["printer", "broken", "login", "fails", "screen", "frozen", "invoice", "missing"]
    for d in splits.values():
        d["ticket"] = [" ".join(rng.choice(words, 9)) + f" ref {rng.integers(1e9)}"
                       for _ in range(len(d))]
    stolen = splits["train"].head(20).copy()
    stolen["ticket"] = stolen["ticket"].str.upper() + "!!"
    stolen["row_id"] += 10**7
    stolen["customer"] = "N" + stolen["customer"]
    stolen["f1"] = 0.0
    splits["test"] = pd.concat([splits["test"], stolen])
    r = case.run(splits, {"features": BASE_FEATS + ["ticket"]})
    assert not r.failed("duplicates.near_cross_split")   # f1 differs: rows are not copies
    assert "duplicates.text_cross_split" in r["blocking"]


def test_embargo_not_double_counted(case, source):
    fr = time_split(source)   # labels 20 days after rows; test starts 31 days after train ends
    r = case.run(fr, {**TIME, "embargo": "30D", "checks": {"drift.*": "off"}})
    assert not r.failed("temporal.label_after_cutoff") and not r.failed("temporal.order")


def test_chain_checks_every_pair(case, source):
    df = source
    fr = {"train": df[df["date"] < "2022-09-01"],
          "val": df[(df["date"] >= "2022-10-01") & (df["date"] < "2023-01-01")],
          "test": df[df["date"] >= "2023-02-01"]}
    params = {"time": "date", "time_order": "train < val < test", "group": None,
              "checks": {"drift.*": "off", "split.undeclared_group": "off"}}
    assert not case.run(fr, params).failed("temporal.order")
    early = fr["test"].head(5).assign(date=pd.Timestamp("2023-03-01"))
    fr["train"] = pd.concat([fr["train"], early])   # train now reaches into val and test
    pairs = case.run(fr, params).check("temporal.order")["details"]["violations"]
    assert set(pairs) == {"train<val", "train<test"}


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
    ({"checks": {"duplicates.cross_split": {"max_fraction": "0.01"}}}, "must be a number"),
    ({"checks": {"split.conservation": {"exact": 1}}}, "must be true or false"),
    ({"time": "date", "time_bounds": {"train": {"max": 150}}}, "must be a date"),
    ({"time": "row_id", "time_order": "train < test", "embargo": "7D"}, "is a duration"),
    ({"assert": [{"name": "money", "sql": "SELECT * FROM train WHERE f1 > $100"}]},
     r"write \$\$ for a literal"),
    ({"assert": [{"name": "spaced", "expr": "unit price > 0"}]}, "backticks"),
])
def test_config_errors_are_helpful(case, splits, bad, match):
    from data_verify import SpecError
    with pytest.raises(SpecError, match=match):
        case.run(splits, bad)


def test_float_columns_match_declared_values(case, splits):
    """A float column (any numeric CSV column with a NaN) renders 1 as "1.0"; declared values
    and integer columns in other splits must still match it."""
    tr = splits["train"]
    tr["y"] = tr["y"].astype("float64")
    tr["row_id"] = tr["row_id"].astype("float64")
    rule = {"key": "row_id", "method": "values",
            "values": {n: d["row_id"].astype(int).tolist() for n, d in splits.items()}}
    params = {"label_values": [0, 1], "split_rule": rule,
              "fitted_levels": [{"levels": [0, 1], "column": "y"}]}
    r = case.run(splits, params)
    for cid in ("labels.valid", "labels.unseen_classes", "split.rule",
                "preprocessing.fitted_levels", "labels.shift"):
        assert not r.failed(cid), r.check(cid)
    tr.loc[:2, "y"] = 7.0
    assert case.run(splits, params).failed("labels.valid")


def test_train_only_backdoor_trigger(case, splits):
    """The textbook backdoor: the trigger exists in training data only."""
    rng = np.random.default_rng(7)
    for d in splits.values():
        d["coupon"] = rng.choice(["none", "spring", "summer"], len(d))
    params = {"features": BASE_FEATS + ["coupon"]}
    assert not case.run(splits, params).failed("poisoning.spurious_association")
    tr = splits["train"]
    idx = tr.sample(60, random_state=2).index
    tr.loc[idx, "coupon"] = "zx9"
    tr.loc[idx, "y"] = 1
    sa = case.run(splits, params).check("poisoning.spurious_association")
    assert sa["status"] == "fail"
    hit = sa["details"]["associations"][0]
    assert hit["value"] == "zx9" and hit["eval_rows"] == 0 and hit["eval_purity"] is None


def test_tz_aware_time_across_dst(case, source):
    fr = time_split(source)
    for d in fr.values():
        for col in ("date", "label_date"):
            d[col] = d[col].dt.tz_localize("America/New_York")
    bounds = {"train": {"max": "2023-03-31T00:00:00-04:00"},   # tz-aware times compare in UTC
              "test": {"min": "2023-05-01T00:00:00-04:00"}}
    r = case.run(fr, {**TIME, "time_bounds": bounds, "checks": {"drift.*": "off"}}, fmt="csv")
    for cid in ("temporal.parse", "temporal.order", "temporal.bounds"):
        assert r.status(cid) == "pass", r.check(cid)


def test_split_column_typo_without_declared_splits(case, splits):
    from data_verify import SpecError
    table = pd.concat([d.assign(split=k) for k, d in splits.items()])
    table.loc[table.index[:3], "split"] = "tset"
    r = case.run({"data": table}, {"split_column": "split"})
    assert list(r["splits"]) == ["train", "val", "test"]
    assert r.failed("schema.splits_present")
    assert r.check("schema.splits_present")["details"]["unknown_values"] == ["tset"]
    table["split"] = table["split"].map({"train": "a", "val": "b", "test": "c"})
    with pytest.raises(SpecError, match="params.splits"):
        case.run({"data": table}, {"split_column": "split"})


def test_glob_severity_ignores_key_order(case, splits):
    a = {"drift.*": "off", "*.features": "warn"}
    b = dict(reversed(a.items()))
    for checks in (a, b):
        r = case.run(splits, {"checks": checks})
        assert r.check("drift.features")["severity"] == "warn"
        assert r.status("drift.unseen_categories") == "off"


def test_assertion_quoting(case, splits):
    for d in splits.values():
        d["unit price"] = d["weight"]
    r = case.run(splits, {"assert": [
        {"name": "spaced", "expr": "`unit price` > 0"},
        {"name": "dollar", "sql": "SELECT * FROM train WHERE '$$' <> '$$'"},
    ]})
    assert r.status("assert.spaced") == "pass" and r.status("assert.dollar") == "pass"


# -- undeclared time ----------------------------------------------------------------------------


def test_undeclared_date_column_warns(case, splits):
    r = case.run(splits)
    assert r.failed("temporal.undeclared_time")
    assert r.check("temporal.undeclared_time")["details"]["columns"] == ["date", "label_date"]
    assert r["passed"]  # a warning: the gate cannot know the rows depend on time


def test_undeclared_date_written_as_text_warns(case, splits):
    for d in splits.values():
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")
        d["label_date"] = d["label_date"].dt.strftime("%d/%m/%Y")
    r = case.run(splits)
    assert r.check("temporal.undeclared_time")["details"]["columns"] == ["date", "label_date"]


def test_declared_time_skips_the_undeclared_check(case, splits):
    r = case.run(splits, {"time": "date"})
    assert r.status("temporal.undeclared_time") == "skip"


def test_data_without_dates_passes_the_undeclared_check(case, splits):
    for d in splits.values():
        d.drop(columns=["date", "label_date"], inplace=True)
    r = case.run(splits, {"forbidden": []})
    assert r.status("temporal.undeclared_time") == "pass"


def test_few_distinct_dates_are_not_a_timeline(case, splits):
    for d in splits.values():
        d["date"] = pd.Timestamp("2024-01-01")
        d["label_date"] = pd.Timestamp("2024-02-01")
    assert not case.run(splits).failed("temporal.undeclared_time")


def test_text_that_only_starts_with_digits_is_not_a_date(case, splits):
    for d in splits.values():
        d["date"] = [f"{i % 90}-item" for i in range(len(d))]
        d["label_date"] = [f"code {i}" for i in range(len(d))]
    assert not case.run(splits).failed("temporal.undeclared_time")


# -- near-duplicates under a new key ------------------------------------------------------------

LABEL_ONLY = {"id": [], "group": [], "weight": None, "features": "auto", "forbidden": [],
              "checks": {"temporal.undeclared_time": "off"}}


def _near_copies(splits, rows=100):
    copies = splits["test"].head(rows).copy()
    copies["row_id"] += 10**7
    copies["customer"] = "N" + copies["row_id"].astype(str)
    copies[["f1", "f2", "f3", "f4"]] *= 1 + 1e-5
    splits["train"] = pd.concat([splits["train"], copies], ignore_index=True)
    for d in splits.values():
        d.drop(columns=["date", "label_date", "weight"], inplace=True)
    return splits


def test_near_copies_under_undeclared_keys_are_found(case, splits):
    r = case.run(_near_copies(splits), LABEL_ONLY)
    assert r.failed("duplicates.near_cross_split")


def test_exact_copies_under_undeclared_keys_are_found_by_the_near_check(case, splits):
    """Not the same on every column, so duplicates.cross_split can't see them either."""
    copies = splits["test"].head(100).copy()
    copies["row_id"] += 10**7
    copies["customer"] = "N" + copies["row_id"].astype(str)
    splits["train"] = pd.concat([splits["train"], copies], ignore_index=True)
    for d in splits.values():
        d.drop(columns=["date", "label_date", "weight"], inplace=True)
    assert case.run(splits, LABEL_ONLY).failed("duplicates.near_cross_split")


def test_columns_that_cannot_tell_rows_apart_stay_quiet_without_the_keys(case, splits):
    for d in splits.values():
        d.drop(columns=["date", "label_date", "weight", "f1", "f2", "f3", "f4"], inplace=True)
    assert not case.run(splits, LABEL_ONLY).failed("duplicates.near_cross_split")


def test_clean_data_with_undeclared_keys_has_no_near_copies(case, splits):
    for d in splits.values():
        d.drop(columns=["date", "label_date", "weight"], inplace=True)
    assert not case.run(splits, LABEL_ONLY).failed("duplicates.near_cross_split")


def test_columns_option_is_not_second_guessed(case, splits):
    r = case.run(_near_copies(splits), {**LABEL_ONLY, "checks": {
        "temporal.undeclared_time": "off",
        "duplicates.near_cross_split": {"columns": ["customer", "f1", "f2", "f3"]}}})
    assert not r.failed("duplicates.near_cross_split")


# -- label consistency --------------------------------------------------------------------------


def _strong(splits):
    """Labels the numeric features predict well, as a real target would be."""
    rng = np.random.default_rng(1)
    for d in splits.values():
        z = d["f1"] * 1.5 - d["f2"] + rng.normal(0, 0.7, len(d))
        d["y"] = (z > 0).astype("int64")
    return splits


def _flip(d, rows):
    d.loc[rows, "y"] = 1 - d.loc[rows, "y"]


def test_clean_labels_are_consistent(case, splits):
    assert case.run(_strong(splits)).status("poisoning.label_consistency") == "pass"


def test_random_label_flips_in_train_warn(case, splits):
    splits = _strong(splits)
    _flip(splits["train"], splits["train"].sample(frac=0.15, random_state=0).index)
    r = case.run(splits)
    assert r.failed("poisoning.label_consistency")
    assert r["passed"]  # a warning: a real change between splits looks the same


def test_targeted_flips_in_train_warn(case, splits):
    splits = _strong(splits)
    tr = splits["train"]
    _flip(tr, tr[tr["f1"] > 0.5].sample(frac=0.6, random_state=0).index)
    assert case.run(splits).failed("poisoning.label_consistency")


def test_inverted_train_labels_warn(case, splits):
    splits = _strong(splits)
    splits["train"]["y"] = 1 - splits["train"]["y"]
    assert case.run(splits).failed("poisoning.label_consistency")


def test_noise_in_evaluation_labels_only_is_not_train_poisoning(case, splits):
    splits = _strong(splits)
    _flip(splits["test"], splits["test"].sample(frac=0.2, random_state=0).index)
    assert not case.run(splits).failed("poisoning.label_consistency")


def test_regression_label_flips_warn(case, splits):
    rng = np.random.default_rng(2)
    for d in splits.values():
        d["y"] = d["f1"] * 2 + rng.normal(0, 1, len(d))
    tr = splits["train"]
    rows = tr.sample(frac=0.3, random_state=0).index
    tr.loc[rows, "y"] = rng.permutation(tr.loc[rows, "y"].to_numpy())
    assert case.run(splits, {"task": "regression"}).failed("poisoning.label_consistency")


def test_features_without_signal_cannot_fail(case, splits):
    rng = np.random.default_rng(3)
    for d in splits.values():
        d["y"] = rng.integers(0, 2, len(d))
    assert case.run(splits).status("poisoning.label_consistency") == "pass"


def test_multiclass_is_skipped_with_a_reason(case, splits):
    for d in splits.values():
        d["y"] = (d["f1"].rank(pct=True) * 3).astype("int64").clip(0, 2)
    r = case.run(splits)
    assert r.status("poisoning.label_consistency") == "skip"


def test_small_splits_are_skipped(case, splits):
    r = case.run({k: d.head(60) for k, d in splits.items()})
    assert r.status("poisoning.label_consistency") == "skip"


def test_small_train_split_is_skipped(case, splits):
    splits["train"] = splits["train"].head(150)
    assert case.run(_strong(splits)).status("poisoning.label_consistency") == "skip"


@pytest.mark.parametrize("held, ev, z, expected", [
    (0.30, 0.50, 4.0, True),     # evaluation labels fit better
    (0.40, -0.40, -9.0, True),   # the model predicts evaluation labels backwards
    (0.30, 0.50, 2.9, False),    # within noise
    (0.46, 0.50, 4.0, False),    # significant but negligible
    (0.00, 0.09, 4.0, False),    # the features predict too little to compare
    (0.50, 0.30, -4.0, False),   # evaluation weaker than train is ordinary drift
    (0.05, -0.40, -9.0, False),  # no relationship in train to reverse
    (0.40, -0.05, -9.0, False),  # evaluation shows no relationship either way
])
def test_label_consistency_decision(held, ev, z, expected):
    opt = {"z": 3.0, "min_gap": 0.05, "min_signal": 0.1}
    assert _inconsistent(held, ev, z, opt) is expected


def _balanced_by_class(n: int, rate: float, seed: int) -> pd.DataFrame:
    """Clean labels: features depend on the class only, so P(x | y) is the same everywhere."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype("int64")
    return pd.DataFrame({"row_id": np.arange(n) + seed * 10**7, "y": y,
                         "f1": rng.normal(y * 1.2, 1), "f2": rng.normal(0, 1, n)})


NUMERIC_ONLY = {"group": [], "weight": None, "features": ["f1", "f2"], "forbidden": []}


@pytest.mark.parametrize("train_rate, test_rate", [(0.05, 0.10), (0.10, 0.20), (0.20, 0.40)])
def test_a_shift_in_class_balance_is_not_label_poisoning(case, train_rate, test_rate):
    frames = {"train": _balanced_by_class(6000, train_rate, 1),
              "test": _balanced_by_class(2000, test_rate, 2)}
    assert case.run(frames, NUMERIC_ONLY).status("poisoning.label_consistency") == "pass"


def test_flips_are_found_despite_a_shift_in_class_balance(case):
    frames = {"train": _balanced_by_class(6000, 0.5, 1), "test": _balanced_by_class(2000, 0.1, 2)}
    _flip(frames["train"], frames["train"].sample(frac=0.2, random_state=0).index)
    assert case.run(frames, NUMERIC_ONLY).failed("poisoning.label_consistency")


def test_a_split_name_with_a_space_warns_rather_than_crashing(case, splits):
    splits = _strong(splits)
    _flip(splits["train"], splits["train"].sample(frac=0.3, random_state=0).index)
    r = case.run(splits, {"splits": {"train": "train", "held out": "val", "z": "test"}})
    found = r.check("poisoning.label_consistency")
    assert found["status"] == "fail" and r["passed"]
    assert {c["split"] for c in found["details"]["comparisons"]} == {"held out", "z"}


def test_a_split_missing_a_class_is_not_compared(case, splits):
    splits = _strong(splits)
    splits["val"] = splits["val"][splits["val"]["y"] == 1]
    r = case.run(splits, {"checks": {"labels.*": "off"}})
    assert r.status("poisoning.label_consistency") == "pass"


@pytest.mark.parametrize("make_key", [
    lambda i: i.astype("float64"),
    lambda i: pd.Series(i, dtype="float64").where(i != i[0]).to_numpy(),   # an int id with a null
    lambda i: [hashlib.sha256(str(x).encode()).hexdigest() for x in i],     # 64 characters
    lambda i: [hashlib.md5(str(x).encode()).hexdigest() for x in i],
], ids=["float", "int with null", "sha256", "md5"])
def test_near_copies_are_found_under_any_shape_of_key(case, splits, make_key):
    splits = _near_copies(splits)
    for d in splits.values():
        d["row_id"] = make_key(d["row_id"].to_numpy())
    r = case.run(splits, LABEL_ONLY)
    assert r.failed("duplicates.near_cross_split")
    assert "row_id" in r.check("duplicates.near_cross_split")["details"]["ignored_keys"]


def test_ignored_keys_are_named_in_the_message(case, splits):
    r = case.run(_near_copies(splits), LABEL_ONLY)
    found = r.check("duplicates.near_cross_split")
    assert set(found["details"]["ignored_keys"]) == {"row_id", "customer"}
    assert "ignoring key column(s) row_id, customer" in found["message"]


def test_a_sentence_is_not_a_key(case, splits):
    for d in splits.values():
        d["note"] = [f"customer called about order number {i} last week and again today "
                     f"about the same thing" for i in d["row_id"]]
        d.drop(columns=["date", "label_date", "weight"], inplace=True)
    r = case.run(splits, LABEL_ONLY)
    assert "note" not in r.check("duplicates.near_cross_split")["details"].get("ignored_keys", [])


def test_key_share_decides_what_counts_as_a_key(case, splits):
    splits = _near_copies(splits)
    r = case.run(splits, {**LABEL_ONLY, "checks": {
        "temporal.undeclared_time": "off",
        "duplicates.near_cross_split": {"key_share": 0.9}}})
    assert r.check("duplicates.near_cross_split")["details"].get("ignored_keys") == ["row_id"]


def test_chance_matches_without_keys_stay_quiet(case):
    """With the id ignored, a few coarse columns repeat by chance, in train as across splits."""
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"row_id": np.arange(20_000), "y": rng.integers(0, 2, 20_000),
                      "a": rng.normal(size=20_000).round(1), "b": rng.normal(size=20_000).round(1),
                      "c": rng.normal(size=20_000).round(1), "e": rng.normal(size=20_000).round(1)})
    r = case.run({"train": d.iloc[:15_000], "test": d.iloc[15_000:]}, {
        **LABEL_ONLY, "checks": {"temporal.undeclared_time": "off"}})
    found = r.check("duplicates.near_cross_split")
    assert found["details"]["ignored_keys"] == ["row_id"]
    assert found["status"] == "pass"


@pytest.mark.parametrize("fmt", ["%Y/%m/%d", "%d-%m-%Y", "%d.%m.%Y", "%b %d, %Y", "%d %b %Y",
                                 "%Y-%m-%dT%H:%M:%S"])
def test_more_date_formats_are_read_as_dates(case, splits, fmt):
    for d in splits.values():
        d["date"] = d["date"].dt.strftime(fmt)
        d["label_date"] = d["label_date"].dt.strftime(fmt)
    r = case.run(splits)
    assert r.check("temporal.undeclared_time")["details"]["columns"] == ["date", "label_date"]
    assert "date, label_date" in r.check("temporal.undeclared_time")["message"]
