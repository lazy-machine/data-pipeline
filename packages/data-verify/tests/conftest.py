"""Synthetic data: a clean grouped, time-stamped binary task, and helpers to run the gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_verify.step import run


def bucket(key: str) -> int:
    d = hashlib.sha256(f"split:{key}".encode()).digest()
    return int(int.from_bytes(d[:8], "big") / 2**64 * 100)


def make_source(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = [f"C{i:04d}" for i in range(900)]
    cust = rng.choice(customers, n)
    date = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    x = rng.normal(size=(n, 4))
    logit = 1.2 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(scale=1.5, size=n)
    y = (logit > 0.8).astype("int64")
    return pd.DataFrame({
        "row_id": np.arange(n) * 7 + 1000,
        "customer": cust,
        "date": date,
        "label_date": date + pd.Timedelta(days=20),
        "f1": x[:, 0], "f2": x[:, 1], "f3": x[:, 2], "f4": x[:, 3],
        "plan": rng.choice(["basic", "plus", "pro", "max"], n),
        "region": rng.choice(["n", "s", "e", "w"], n),
        "weight": rng.uniform(0.5, 2.0, n),
        "y": y,
    })


def group_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    b = df["customer"].map(bucket)
    return {"train": df[b < 70].reset_index(drop=True),
            "val": df[(b >= 70) & (b < 80)].reset_index(drop=True),
            "test": df[b >= 80].reset_index(drop=True)}


def time_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """train: 2022-01-01..2023-03-31 (labels by 2023-04-20), test: 2023-05-01 onwards."""
    b = df["customer"].map(bucket)
    tr = df[(df["date"] <= "2023-03-31") & (b < 80)]
    te = df[(df["date"] >= "2023-05-01") & (b >= 80)]
    return {"train": tr.reset_index(drop=True), "test": te.reset_index(drop=True)}


BASE = {"label": "y", "id": "row_id", "group": "customer", "weight": "weight",
        "features": ["f1", "f2", "f3", "f4", "plan", "region"],
        "forbidden": ["*_date", "outcome*"]}


class Case:
    def __init__(self, tmp: Path):
        self.tmp = tmp

    def write(self, frames: dict[str, pd.DataFrame], fmt: str = "parquet") -> dict[str, Path]:
        paths = {}
        for name, df in frames.items():
            p = self.tmp / f"{name}.{fmt}"
            if fmt == "csv":
                df.to_csv(p, index=False)
            else:
                df.to_parquet(p, index=False)
            paths[name] = p
        return paths

    def run(self, frames, params=None, extra_inputs=None, fmt="parquet"):
        inputs = self.write(frames, fmt)
        inputs.update(extra_inputs or {})
        return Report(run({**BASE, **(params or {})}, inputs, log=lambda *a: None,
                          workdir=self.tmp))


class Report(dict):
    def check(self, cid: str) -> dict:
        found = [c for c in self["checks"] if c["id"] == cid]
        assert found, f"no check {cid}"
        return found[0]

    def status(self, cid: str) -> str:
        return self.check(cid)["status"]

    def failed(self, cid: str) -> bool:
        return self.status(cid) in ("fail", "error")

    def failures(self) -> list[str]:
        return [c["id"] for c in self["checks"] if c["status"] in ("fail", "error")]


@pytest.fixture
def case(tmp_path) -> Case:
    return Case(tmp_path)


@pytest.fixture(scope="session")
def source() -> pd.DataFrame:
    return make_source()


@pytest.fixture
def splits(source) -> dict[str, pd.DataFrame]:
    return {k: v.copy() for k, v in group_split(source).items()}
