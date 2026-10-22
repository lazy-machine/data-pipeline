"""Data preparation steps."""

import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def clean(ctx):
    df = pd.read_csv(ctx.inputs["raw"])
    label = ctx.params["label"]
    missing = set(EXPECTED + [label]) - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")

    before = len(df)
    df = df[EXPECTED + [label]].dropna().drop_duplicates()
    if (df[EXPECTED] <= 0).any().any():
        raise ValueError("measurements must be positive")

    df.to_csv(ctx.outputs["table"], index=False)
    return {"rows": len(df), "dropped": before - len(df), "classes": df[label].nunique()}


def split(ctx):
    df = pd.read_csv(ctx.inputs["table"])
    p = ctx.params
    train, test = train_test_split(
        df, test_size=p["test_size"], random_state=p["seed"], stratify=df[p["label"]]
    )
    train.to_csv(ctx.outputs["train"], index=False)
    test.to_csv(ctx.outputs["test"], index=False)
    return {"train": len(train), "test": len(test)}
