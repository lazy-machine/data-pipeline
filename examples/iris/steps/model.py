"""Model training and evaluation."""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _estimator(p):
    if p["estimator"] == "random_forest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 100),
                                      random_state=p["seed"])
    if p["estimator"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    raise ValueError(f"unknown estimator {p['estimator']!r}")


def train(ctx):
    df = pd.read_csv(ctx.inputs["train"])
    label = ctx.params["label"]
    X, y = df.drop(columns=label), df[label]
    model = _estimator(ctx.params).fit(X, y)
    joblib.dump(model, ctx.outputs["model"])
    return {"estimator": ctx.params["estimator"], "train_acc": model.score(X, y)}


def evaluate(ctx):
    model = joblib.load(ctx.inputs["model"])
    df = pd.read_csv(ctx.inputs["test"])
    label = ctx.params["label"]
    X, y = df.drop(columns=label), df[label]
    pred = model.predict(X)

    per_class = classification_report(y, pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "per_class": {k: v for k, v in per_class.items() if k in set(y)},
    }
    ctx.outputs["metrics"].write_text(json.dumps(metrics, indent=2))
    return {"accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}
