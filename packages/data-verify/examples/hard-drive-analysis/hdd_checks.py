"""A custom check: the part of the hard-drive gate that needs Python (a per-drive hash).

Snapshots taken on the drive's stride schedule weigh drive_weight * stride. The schedule
starts at a per-drive offset, int(sha256("stride:" + serial) as a fraction * stride), so a
stride-weighted row must fall on a day where (days since 2000-01-01 + offset) % stride == 0.
"""

import hashlib

import numpy as np
import pandas as pd


def _unit(serial: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{serial}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def stride_rows_on_stride_days(v):
    stride = int(v.var("stride"))
    if stride == 1:
        v.add("stride_rows_on_stride_days", True, "stride_days = 1: every day is a stride day")
        return
    cohort = v.dataset("cohort")[["serial_number", "drive_weight"]]
    offsets = {s: int(_unit(s, "stride") * stride) for s in cohort["serial_number"]}
    bad = 0
    for _name, d in v.splits.items():
        m = d[["serial_number", "date", "weight"]].merge(cohort, on="serial_number", how="left")
        on_stride = np.abs(m["weight"] - m["drive_weight"] * stride) < 1e-6
        days = (pd.to_datetime(m["date"]) - pd.Timestamp("2000-01-01")).dt.days
        off = m["serial_number"].map(offsets)
        bad += int((on_stride & ((days + off) % stride != 0)).sum())
    v.add("stride_rows_on_stride_days", bad == 0,
          f"{bad} stride-weighted rows off the drive's stride day",
          category="weights", details={"rows": bad},
          hint="The snapshot step and the cohort disagree on the stride offset.")
