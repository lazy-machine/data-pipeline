"""The params reference, shown by `data-verify params` and copied into docs/REFERENCE.md."""

from __future__ import annotations

from .core import REGISTRY

# (key, value, description). Grouped; the group names become headings.
PARAMS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Data layout", [
        ("splits", "list | table",
         "Split inputs in time/workflow order. Default: inputs named train/val/valid/"
         "validation/dev/calib/test/holdout/eval/oot, in that order. A table maps "
         "split name -> input name. With split_column: the split values to expect "
         "(default: the standard names above; any other value in the column fails "
         "schema.splits_present)."),
        ("split_column", "column",
         "One table holds every split; this column says which split each row is in."),
        ("table", "input",
         "With split_column: the input holding the table (needed if there are several "
         "data inputs)."),
        ("train", "split", "The split the model is fitted on. Default: the first train-like "
                           "split."),
        ("source", "input",
         "Pre-split data. Enables split.conservation: no split row may be missing from it."),
    ]),
    ("Column roles", [
        ("label", "column | list", "Target column(s)."),
        ("task", "string | table",
         "auto (default), binary, multiclass or regression; a table sets it per label."),
        ("id", "column | list", "Row key: unique within a split, never in two splits."),
        ("group", "column | list",
         "Entity key (patient, user, device, document): all its rows in one split."),
        ("time", "column", "Each row's timestamp (as-of / prediction time). Dates, datetimes "
                           "or numbers."),
        ("label_time", "column",
         "When the row's label became known (event date, end of label window, censoring date)."),
        ("point_in_time", "column | list",
         "Columns holding when a feature was observed; must be <= time."),
        ("weight", "column", "Sample weight column."),
        ("features", "list | \"auto\"",
         "Feature columns; glob patterns allowed. \"auto\" (default) = every column that has "
         "no role, i.e. what a naive training step would use."),
        ("exclude", "list", "Glob patterns removed from the features."),
        ("forbidden", "list",
         "Glob patterns that must never be features (post-outcome columns, ...)."),
    ]),
    ("Rules", [
        ("time_bounds", "table",
         "split = { min = ..., max = ... } (inclusive). Example: train = { max = "
         "\"2023-12-31\" }, test = { min = \"2024-01-01\" }. Time columns with a "
         "timezone are compared in UTC: give such bounds in UTC or with an offset "
         "(\"2023-12-31T23:59:59-05:00\")."),
        ("time_order", "string | list",
         "\"train < val < test\": every row of a left split is before every row of a "
         "split to its right (every pair of the chain is checked)."),
        ("embargo", "duration", "Minimum gap between row timestamps for time_order: \"7D\", "
                                "\"12h\", or a number of days (units when time is numeric). "
                                "Use the label horizon when there is no label_time."),
        ("time_order_scope", "global | group",
         "group: check time_order per entity (per-series forecasting) instead of globally; "
         "split.group_overlap is then skipped, since entities span splits by design."),
        ("split_rule", "table",
         "How splits were assigned. { key, method = sha256|sha1|md5|modulo|values, "
         "template = \"{key}\", modulo = 100, buckets = { train = [0, 70], ... } } or "
         "{ key, method = \"values\", values = { train = [...], ... } }."),
        ("split_ratios", "table",
         "Planned shares, e.g. { train = 0.7, val = 0.1, test = 0.2, tolerance = 0.05, "
         "unit = \"rows\" | \"groups\" }."),
        ("schema", "table",
         "{ columns = [...], ordered = bool, dtypes = { col = kind }, ranges = { col = "
         "[min, max] }, allowed = { col = [...] }, not_null = [...], unique = [...], "
         "datasets = [extra inputs to check] }."),
        ("label_values", "list", "Allowed label values."),
        ("label_range", "[min, max]", "Allowed numeric label range."),
        ("fitted_levels", "list of tables",
         "Vocabularies a step learned: { levels = [...], column, dataset = train split, "
         "where = \"pandas query\", allow = [\"other\"], top_k = bool }. Each level must "
         "occur in training data."),
        ("pins", "table", "input = \"<sha256>\": inputs that must be byte-identical."),
    ]),
    ("Extensions", [
        ("assert", "list of tables",
         "[[steps.params.assert]] name, and one of: expr (pandas expression true on every "
         "row of `dataset`, default every split), sql (DuckDB query returning violating "
         "rows), sql_count (DuckDB query returning the number of violations). Optional "
         "severity, category, description, hint. ${var} is replaced from vars; write $$ "
         "for a literal $. In expr, quote column names that are not identifiers in "
         "backticks (`my column` > 0). In SQL, every data input and every split is a view "
         "under its name."),
        ("vars", "table", "Values for ${name} in assertions (refs allowed)."),
        ("custom", "list", "\"module:function\" checks, called with the Verifier; call "
                           "v.add(name, ok, message, severity=...) or return a list of dicts."),
    ]),
    ("Behaviour", [
        ("checks", "table",
         "Per-check overrides: \"<id>\" = \"off\" | \"warn\" | \"error\", or "
         "\"<id>\" = { severity = ..., <option> = ... }. Globs allowed for severity "
         "(\"drift.*\" = \"off\"); an exact id beats any glob, and among globs the most "
         "specific (most non-wildcard characters) wins. Options per check: "
         "`data-verify checks`."),
        ("fail_on", "error | warn | never",
         "error (default): failed error-severity checks block. warn: warnings block too. "
         "never: report only."),
        ("sample_rows", "int", "Row cap for statistical checks (default 250000). Exact checks "
                               "(overlaps, duplicates, time) always use every row."),
        ("seed", "int", "Seed for sampling and cross-fitting (default 0)."),
        ("allow_pickle", "bool", "Read .pkl inputs (off: loading a pickle runs code)."),
        ("max_hash_mb", "number", "Inputs above this size are not hashed in the report unless "
                                  "pinned (default 4096)."),
        ("max_examples", "int", "Example rows/values per finding (default 5)."),
    ]),
]

REFS = """Any value can be read from a JSON/TOML/YAML input instead:
  { from = "<input>", key = "a.b.0" }   ({other.key} inside key is looked up first;
                                         add default = ... for optional keys)
Inside a list, a reference to a list is spliced in place.
A `spec` input (TOML/JSON) can hold all params; step params override it."""


def params_text() -> str:
    out = ["data-verify step params (all optional; set at least label to get target checks)", ""]
    for group, items in PARAMS:
        out.append(f"{group}:")
        for key, value, desc in items:
            out.append(f"  {key:<17} {value:<16} {desc}")
        out.append("")
    out += ["References:", *("  " + line for line in REFS.splitlines()), "",
            "Inputs: split files (or one table + split_column), plus any auxiliary data,",
            "config files used by references, and an optional `spec`. Formats: csv, tsv,",
            "parquet (file or directory), feather, json, jsonl.",
            "Outputs: report (JSON, recommended), summary (Markdown). Both optional."]
    return "\n".join(out)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def params_markdown() -> str:
    out = []
    for group, items in PARAMS:
        out += [f"### {group}", "", "| key | value | meaning |", "|---|---|---|"]
        out += [f"| `{k}` | {_cell(v)} | {_cell(d)} |" for k, v, d in items]
        out.append("")
    return "\n".join(out)


def checks_text() -> str:
    out = []
    cat = None
    for c in REGISTRY.values():
        if c.category != cat:
            cat = c.category
            out.append(f"\n[{cat}]")
        opts = ", ".join(f"{k}={v!r}" for k, v in c.options.items()) or "-"
        out.append(f"  {c.id:<34} {c.severity:<5} {c.title}")
        out.append(f"      options: {opts}")
    return "\n".join(out).lstrip()


def checks_markdown() -> str:
    out = []
    cat = None
    for c in REGISTRY.values():
        if c.category != cat:
            cat = c.category
            out += [f"### {cat}", ""]
        opts = ", ".join(f"`{k} = {v!r}`" for k, v in c.options.items())
        out += [f"#### `{c.id}` ({c.severity})", "", f"**{c.title}.** {c.description}", ""]
        if opts:
            out += [f"Options: {opts}", ""]
    return "\n".join(out)


def checks_json() -> list[dict]:
    return [{"id": c.id, "category": c.category, "severity": c.severity, "title": c.title,
             "description": c.description, "options": c.options} for c in REGISTRY.values()]


TEMPLATE = '''# data-verify gate: put it right after the step that writes the splits, and make every
# training/evaluation step depend on it (depends_on = ["verify"]).
[[steps]]
name = "verify"
description = "Gate: no leakage between splits, no poisoned or invalid data"
function = "data_verify:verify"
inputs = { train = "build/train.parquet", val = "build/val.parquet", test = "build/test.parquet" }
outputs = { report = "build/verify/report.json", summary = "build/verify/report.md" }
requires = ["data-verify"]

[steps.params]
label = "target"               # the column to predict
# id = "row_id"                # unique row key
# group = "customer_id"        # entity that must not span splits
# time = "date"                # row timestamp; then declare the time rules:
# time_order = "train < val < test"
# label_time = "outcome_date"  # when the label became known
# features = ["f1", "f2"]      # default: every column without a role
# forbidden = ["*_after_*"]    # columns that must never be features
# [steps.params.checks]
# "drift.*" = "off"            # turn checks off or tune them; see `data-verify checks`

# One table holding every split instead of one file per split:
#   inputs = { data = "build/dataset.parquet" }
#   params: split_column = "split", splits = ["train", "val", "test"]
'''
