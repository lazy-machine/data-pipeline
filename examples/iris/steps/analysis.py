"""Exploratory analysis and reporting (non-critical steps)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def explore(ctx):
    df = pd.read_csv(ctx.inputs["table"])
    label = ctx.params["label"]
    features = [c for c in df.columns if c != label]

    stats = {
        "rows": len(df),
        "class_counts": df[label].value_counts().to_dict(),
        "describe": df[features].describe().round(3).to_dict(),
        "correlation": df[features].corr().round(3).to_dict(),
    }
    ctx.outputs["stats"].write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.5))
    for ax, feat in zip(axes, features):
        for cls, grp in df.groupby(label):
            ax.hist(grp[feat], bins=15, alpha=0.6, label=cls)
        ax.set_title(feat)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ctx.outputs["plot"], dpi=100)

    top = df[features].corr().abs().where(lambda m: m < 1).stack().idxmax()
    return {"features": len(features), "most_correlated": "~".join(top)}


def report(ctx):
    stats = json.loads(ctx.inputs["stats"].read_text())
    metrics = json.loads(ctx.inputs["metrics"].read_text())

    lines = ["# Iris pipeline report", "", "## Data", "",
             f"- rows: {stats['rows']}"]
    lines += [f"- {cls}: {n}" for cls, n in stats["class_counts"].items()]
    lines += ["", "## Model", "",
              f"- accuracy: {metrics['accuracy']:.3f}",
              f"- macro F1: {metrics['f1_macro']:.3f}", "",
              "| class | precision | recall | f1 |", "|---|---|---|---|"]
    for cls, m in metrics["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} |")
    ctx.outputs["report"].write_text("\n".join(lines) + "\n")
    return f"{len(lines)} lines"
