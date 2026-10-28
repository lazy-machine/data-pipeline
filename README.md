# data-verify

A gate step for [data-pipeline](../data-pipeline) projects. It reads the data your pipeline
produced (the train/val/test splits, plus any source or auxiliary tables) and fails the
step, stopping everything downstream, when it finds **data leakage** (anything that lets a
model see the answers or the future of what it is scored on) or **poisoning** (tampered,
injected or corrupted training data). Every check, passed or not, goes into a JSON report.

```
[2/4] verify   ✗ FAILED  2.1s  VerificationFailed
data-verify blocked the pipeline: 1 blocking check(s) failed
  x split.group_overlap: 12 group value(s) appear in more than one split: train/test (12, e.g. ['C0042', 'C0107'])
    fix: Split by group (e.g. GroupShuffleSplit, or a stable hash of the group key) instead of by row.
report: build/verify/report.json  (details: data-verify explain build/verify/report.json)
```

## Add it to a pipeline

```bash
uv add --editable ../data-verify          # or: uv add "data-verify @ file:///path/to/data-verify"
uv add duckdb                             # only if you use SQL assertions
```

Put the gate right after the step that writes the splits, and make every training and
evaluation step depend on it:

```toml
[[steps]]
name = "verify"
description = "Gate: no leakage between splits, no poisoned or invalid data"
function = "data_verify:verify"
inputs = { train = "build/train.parquet", val = "build/val.parquet", test = "build/test.parquet" }
outputs = { report = "build/verify/report.json", summary = "build/verify/report.md" }
requires = ["data-verify"]
params = { label = "churned", id = "row_id", group = "customer_id" }

[[steps]]
name = "train"
depends_on = ["verify"]        # nothing trains on data the gate has not passed
# ...
```

`data-verify template` prints a commented version of this block.

## The contract

### Inputs: what the step reads

The input **names** say what each file is:

| input name | meaning |
|---|---|
| `train`, `val` / `valid` / `validation` / `dev` / `calib`, `test` / `holdout` / `eval` / `oot` | the splits, recognised automatically, ordered train → val → test |
| any other name | a split if listed in `params.splits`; the one table holding all splits if `params.split_column` is set (`params.table` names it when there are several data inputs); the pre-split data if named by `params.source`; otherwise an auxiliary dataset for assertions, `fitted_levels` and custom checks |
| a `.json` / `.toml` / `.yaml` input used by a `{ from = ... }` reference | configuration, not data (see References below) |
| `spec` | a TOML/JSON file holding the params; step params override it |

Data formats: CSV, TSV, Parquet (a file, or a directory of parquet files), Feather, JSON,
JSONL. Pickle is refused unless `allow_pickle = true`, because loading a pickle runs code.

### Outputs: what the step writes

| output name | content |
|---|---|
| `report` | JSON report (recommended). Deterministic: no timestamps, so downstream steps stay cached. |
| `summary` | Markdown: one table row per check, blocking ones first, then "How to fix". |

Both are optional; any other output name is an error. The report is written **before** the
step fails, so you can always read why it blocked.

### Result

- **All blocking checks pass**: the step succeeds; the status line shows
  `checks=… passed=… warnings=… skipped=…`.
- **A blocking check fails**: the step raises `VerificationFailed` with each failure and its
  fix. data-pipeline marks the step failed, prints that message, and halts (the step is
  critical by default). A check that crashes counts as failed: an unverified check never
  passes silently.
- **The params are invalid** (unknown key, wrong type, missing column for a rule): the step
  raises `SpecError`, which names the key and suggests the closest valid one.

`fail_on` sets what blocks: `"error"` (default; failed checks of severity error),
`"warn"` (warnings too), `"never"` (report only).

### Report format

```jsonc
{
  "tool": "data-verify", "version": "0.1.0",
  "passed": false,                       // false = the step failed
  "fail_on": "error",
  "blocking": ["split.group_overlap"],   // ids of the checks that blocked
  "warnings": ["drift.features"],        // failed, but not blocking
  "counts": {"pass": 30, "fail": 2, "error": 0, "skip": 17, "off": 0},
  "splits": {"train": {"rows": 7000, "columns": 14}, "test": {"rows": 2000, "columns": 14}},
  "train_split": "train",
  "roles": {"label": ["churned"], "task": {"churned": "binary"}, "group": ["customer_id"],
            "features": ["..."], "features_mode": "declared", "...": "..."},
  "datasets": {"train": {"path": "build/train.parquet", "bytes": 81234,
                         "sha256": "…", "role": "split"}},
  "checks": [{
    "id": "split.group_overlap", "category": "split", "severity": "error",
    "status": "fail",                    // pass | fail | error (crashed) | skip | off
    "blocking": true,
    "title": "No entity (group) is in two splits",
    "message": "12 group value(s) appear in more than one split: ...",
    "hint": "Split by group ...",        // how to fix, when failed
    "details": {"pairs": {"train/test": {"shared": 12, "examples": ["C0042", "..."]}}}
  }]
}
```

`data-verify explain build/verify/report.json` prints the failed checks with their details.

## Params

Every key is optional. The more roles you declare, the more checks can run; checks whose
inputs are missing report `skip` with the reason. `data-verify params` prints this table.

### Data layout

| key | value | meaning |
|---|---|---|
| `splits` | list \| table | Split inputs in time/workflow order. Default: inputs named train/val/valid/validation/dev/calib/test/holdout/eval/oot, in that order. A table maps split name -> input name. With split_column: the split values to expect. |
| `split_column` | column | One table holds every split; this column says which split each row is in. |
| `table` | input | With split_column: the input holding the table (needed if there are several data inputs). |
| `train` | split | The split the model is fitted on. Default: the first train-like split. |
| `source` | input | Pre-split data. Enables split.conservation: no split row may be missing from it. |

### Column roles

| key | value | meaning |
|---|---|---|
| `label` | column \| list | Target column(s). |
| `task` | string \| table | auto (default), binary, multiclass or regression; a table sets it per label. |
| `id` | column \| list | Row key: unique within a split, never in two splits. |
| `group` | column \| list | Entity key (patient, user, device, document): all its rows in one split. |
| `time` | column | Each row's timestamp (as-of / prediction time). Dates, datetimes or numbers. |
| `label_time` | column | When the row's label became known (event date, end of label window, censoring date). |
| `point_in_time` | column \| list | Columns holding when a feature was observed; must be <= time. |
| `weight` | column | Sample weight column. |
| `features` | list \| "auto" | Feature columns; glob patterns allowed. "auto" (default) = every column that has no role, i.e. what a naive training step would use. |
| `exclude` | list | Glob patterns removed from the features. |
| `forbidden` | list | Glob patterns that must never be features (post-outcome columns, ...). |

### Rules

| key | value | meaning |
|---|---|---|
| `time_bounds` | table | split = { min = ..., max = ... } (inclusive). Example: train = { max = "2023-12-31" }, test = { min = "2024-01-01" }. |
| `time_order` | string \| list | "train < val < test": every row of the left split is before every row of the right one. |
| `embargo` | duration | Minimum gap for time_order: "7D", "12h", or a number of days (units when time is numeric). |
| `time_order_scope` | global \| group | group: check time_order per entity (per-series forecasting) instead of globally. |
| `split_rule` | table | How splits were assigned. { key, method = sha256\|sha1\|md5\|modulo\|values, template = "{key}", modulo = 100, buckets = { train = [0, 70], ... } } or { key, method = "values", values = { train = [...], ... } }. |
| `split_ratios` | table | Planned shares, e.g. { train = 0.7, val = 0.1, test = 0.2, tolerance = 0.05, unit = "rows" \| "groups" }. |
| `schema` | table | { columns = [...], ordered = bool, dtypes = { col = kind }, ranges = { col = [min, max] }, allowed = { col = [...] }, not_null = [...], unique = [...], datasets = [extra inputs to check] }. |
| `label_values` | list | Allowed label values. |
| `label_range` | [min, max] | Allowed numeric label range. |
| `fitted_levels` | list of tables | Vocabularies a step learned: { levels = [...], column, dataset = train split, where = "pandas query", allow = ["other"], top_k = bool }. Each level must occur in training data. |
| `pins` | table | input = "<sha256>": inputs that must be byte-identical. |

### Extensions

| key | value | meaning |
|---|---|---|
| `assert` | list of tables | `[[steps.params.assert]]` name, and one of: expr (pandas expression true on every row of `dataset`, default every split), sql (DuckDB query returning violating rows), sql_count (DuckDB query returning the number of violations). Optional severity, category, description, hint. `${var}` is replaced from vars. In SQL, every data input and every split is a view under its name. |
| `vars` | table | Values for `${name}` in assertions (refs allowed). |
| `custom` | list | "module:function" checks, called with the Verifier; call v.add(name, ok, message, severity=...) or return a list of dicts. |

### Behaviour

| key | value | meaning |
|---|---|---|
| `checks` | table | Per-check overrides: "<id>" = "off" \| "warn" \| "error", or "<id>" = { severity = ..., <option> = ... }. Globs allowed for severity ("drift.*" = "off"). Options per check: `data-verify checks`. |
| `fail_on` | error \| warn \| never | error (default): failed error-severity checks block. warn: warnings block too. never: report only. |
| `sample_rows` | int | Row cap for statistical checks (default 250000). Exact checks (overlaps, duplicates, time) always use every row. |
| `seed` | int | Seed for sampling and cross-fitting (default 0). |
| `allow_pickle` | bool | Read .pkl inputs (off: loading a pickle runs code). |
| `max_hash_mb` | number | Inputs above this size are not hashed in the report unless pinned (default 4096). |
| `max_examples` | int | Example rows/values per finding (default 5). |

### References: one source of truth

Any value can be read from a JSON/TOML/YAML input instead of being copied into the pipeline
file, so the gate checks the same contract the other steps use:

```toml
inputs = { train = "...", test = "...", study = "config/study.toml", schema = "build/schema.json" }
[steps.params]
features = { from = "schema", key = "features" }
time_bounds.train = { max = { from = "study", key = "split.train_end" } }
vars.stride = { from = "sample", key = "scales.{scale}.stride_days" }   # {scale} looked up first
schema.columns = ["id", "date", { from = "schema", key = "features" }]  # a list ref is spliced
```

Add `default = ...` to a reference for optional keys.

## Recipes

**Random row split (i.i.d. rows).** `label` is enough. Add `id` if rows have a key.

**Grouped data** (several rows per patient, user, device, session, document): set `group`.
This is the most common leak; without `group` the gate cannot see it.

**Time-dependent data** (forecasting, churn, failure, fraud): set `time` and declare the
split in time, plus when labels become known:

```toml
[steps.params]
label = "fails_30d"
group = "device_id"
time = "snapshot_date"
label_time = "label_known_date"      # e.g. snapshot_date + 30 days, or the event date
time_order = "train < val < test"
embargo = "30D"                      # at least the label horizon
```

With `time` but no `time_order`/`time_bounds`, `temporal.overlap` warns that evaluation rows
fall inside the training period.

**One table with a split column:**
`params = { split_column = "split", table = "snapshots", label = "y" }`.

**Unlabeled test split:** `checks = { "schema.roles" = { unlabeled = ["test"] } }`.

**Project rules** (label definitions, censoring, sampling weights): SQL or pandas
assertions, or a Python check. See
[examples/hard-drive-analysis/verify.toml](examples/hard-drive-analysis/verify.toml), which
reproduces a real project's hand-written gate this way.

```toml
[[steps.params.assert]]
name = "no_row_after_event"
sql = "SELECT * FROM snapshots s JOIN events e USING (device_id) WHERE s.date > e.event_date"

[[steps.params.assert]]
name = "positive_amounts"
expr = "amount > 0"
dataset = "train"
severity = "warn"
```

A custom check is a function taking the verifier (`v.splits`, `v.train`, `v.dataset(name)`,
`v.sql(query)`, `v.var(name)`, `v.features`, `v.labels`) and calling
`v.add(name, ok, message, severity="error", details={}, hint="")`. Put its file in the
step's `watch` list so editing it reruns the gate.

## What it checks

49 built-in checks; `data-verify checks` lists them with their options, and
[docs/CHECKS.md](docs/CHECKS.md) describes each one.

| area | blocks (error) | warns |
|---|---|---|
| split | same `id` or `group` in two splits; repeated ids; split not matching `split_rule`; split rows missing from `source` (injected/altered rows) | split shares off `split_ratios` |
| duplicates | identical feature rows in two splits (beyond what different entities share by chance in train) | near-copies across splits (rounded numbers, normalised text); mass-repeated rows in train; identical rows with conflicting labels |
| temporal | rows outside `time_bounds`; splits overlapping in time (or per group); labels known only after the evaluation period starts; rows dated after their own outcome; features observed after the row's time; unparseable dates | time-stamped splits overlapping when no time rule is declared |
| target | label/key/time columns or `forbidden` patterns used as features; a feature equal to the label; one feature (or its missingness) that alone determines the label | strongly predictive single features; features far more predictive in one split (target encoding fitted with evaluation labels); outcome-like names; id-like features |
| preprocessing | scaler fitted on train+evaluation data (standardised or min-max over the union only); fitted vocabularies containing levels absent from training data | |
| labels | missing / out-of-domain labels, single-class train; evaluation classes absent from train | classes missing from a split; label distribution shift |
| weights | negative / infinite / all-zero weights | a few rows holding most of the weight |
| poisoning | pinned input changed (`pins`) | excess of extreme values in train vs held-out data; backdoor-like value or text token tied to one class in train only |
| schema / integrity | missing split, unknown split values, declared columns missing, columns or types differing between splits, `schema` rule violations, infinite feature values | empty or constant features |
| drift | | feature distribution shift (PSI); unseen categories |

### When the gate blocks

1. Read the message and `hint`, or run `data-verify explain <report>`.
2. Fix the data step (split by group, cut by time, fit transforms on train only, drop the
   column) and rerun. That is the expected outcome almost every time.
3. Only if the finding is a property of the domain, tune the narrowest knob and say why in a
   comment, e.g. `"duplicates.cross_split" = { max_fraction = 0.001 }  # identical sensor
   readings across machines are normal`. Turning a check `"off"` is a last resort.

### Limits

The gate sees data, not intentions. It cannot know that a column was computed from the
future unless `time`, `label_time` or `point_in_time` say when things happened; it finds
statistical traces of poisoning (copies, outliers, triggers, shifted labels) but not an
attack crafted to match the clean distribution; and it trusts the roles you declare. Declare
`group`, `time` and `label_time` whenever they exist.

## Caching

data-pipeline fingerprints `data_verify/__init__.py` (an installed module), not the rest of
the package. The version string lives there, so upgrading data-verify reruns the gate.
After editing data-verify itself without a version bump, run `data-pipeline -r verify`.
The gate's own params, inputs and `watch` files invalidate it as usual.

## Command line

```
data-verify run pipeline.toml [--step NAME]   run a data-verify step without the orchestrator
data-verify run verify.toml                   standalone config: [inputs] [outputs] [params]
data-verify explain REPORT.json [--all]       failed checks, details and fixes
data-verify checks [--json | --markdown]      every check, severity and options
data-verify params [--markdown]               every params key
data-verify template                          step block to paste into pipeline.toml
```

Exit codes: `0` passed, `1` blocked, `2` invalid config.

## Development

```bash
uv sync
uv run pytest          # injects every leak/poisoning pattern into clean data
uv run data-verify run examples/hard-drive-analysis/verify.toml   # needs ../hard-drive-analysis-v3/build
```

Layout: `spec.py` (params parsing), `core.py` (loading, hashing, registry, Verifier),
`checks/*.py` (one module per area; a check is a function decorated with `@check`),
`stats.py` (single-feature scores, PSI, robust statistics), `step.py` (runner, report),
`cli.py`, `reference.py` (params and checks reference).
