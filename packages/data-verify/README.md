# data-verify

A gate step for [data-pipeline](../../README.md) projects, shipped in the same repository
(a uv workspace member). It reads the data your pipeline
produced (the train/val/test splits, plus any source or auxiliary tables) and stops the
pipeline when it finds:

- **Data leakage**: anything that lets a model see the answers, or the future, of what it
  is scored on.
- **Poisoning**: tampered, injected or corrupted training data.

Every check, passed or not, goes into a JSON report. Steps that `depends_on = ["verify"]`
never run on data that failed the gate:

```
[3/5] verify    ✗ FAILED     0.51s  exit code 1
┌─ verify: exit code 1 ─ data_verify:verify
Traceback (most recent call last):
  ...
data_verify.step.VerificationFailed: data-verify blocked the pipeline: 2 blocking check(s) failed
  x split.group_overlap: 971 group value(s) appear in more than one split: train/test (398, e.g. ['P0000', 'P0003']), ...
    fix: Split by group (e.g. GroupShuffleSplit, or a stable hash of the group key) instead of by row.
  x target.perfect_predictor: 2 feature(s) alone predict the label (AUC >= 0.99): isnull(followup_lab)->readmitted=1.0, discharge_code->readmitted=0.992616
    fix: A single column that (almost) determines the label is usually computed from the outcome or recorded after it. ...
report: build/verify/report.json  (details: data-verify explain build/verify/report.json)
└────────────────────────────────────────────────────────────
■ halted at 'verify' · 2 cached · 1 failed · 0.52s
```

**Contents**

- [Add it to a pipeline](#add-it-to-a-pipeline)
- [The contract](#the-contract): inputs, outputs, result, report format
- [Params](#params)
- [Recipes](#recipes)
- [What it checks](#what-it-checks)
- [Caching](#caching) · [Command line](#command-line) · [Development](#development)

## Add it to a pipeline

**1. Install it** in the pipeline project:

```bash
uv add --editable "../data-pipeline[verify]"   # runner + gate, from a sibling checkout of data-pipeline
uv add duckdb                                  # only if you use SQL assertions
```

The [usage guide](../../docs/GUIDE.md) (§2 and §7) covers the other ways to depend on it and
where the gate goes in a pipeline; `examples/iris` at the repo root is a working example.

**2. Add the step** right after the step that writes the splits, and make every training
and evaluation step depend on it:

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

**3. Check the params** with `data-verify lint`. It validates every data-verify step in
`./pipeline.toml` without running anything. (The step itself only sees its params when its
turn comes, after the upstream steps ran.)

### Declare every role that exists in your data

The gate can only check what it knows:

- **Without `group`**, rows of one patient in train and test look like different rows. The
  `split.undeclared_group` warning tries to spot such columns.
- **Without `time`**, a random split of time-dependent data looks fine.
- **With the default `features = "auto"`**, every column without a role is treated as a
  feature, including stray keys. List `features` explicitly when you know them.

## The contract

### Inputs: what the step reads

The input **names** say what each file is:

| input name | meaning |
|---|---|
| `train` | training split |
| `val`, `valid`, `validation`, `dev`, `calib` | validation split |
| `test`, `holdout`, `eval`, `oot` | test split |
| `spec` | a TOML/JSON file holding the params; step params override it |
| a `.json` / `.toml` / `.yaml` input used by a `{ from = ... }` reference | configuration, not data (see [References](#references-one-source-of-truth)) |
| any other name | see below |

Splits with the names above are recognised automatically and ordered train → val → test.
An input with any other name is:

1. a split, if listed in `params.splits`;
2. the one table holding all splits, if `params.split_column` is set (`params.table` names
   it when there are several data inputs);
3. the pre-split data, if named by `params.source`;
4. otherwise, an auxiliary dataset for assertions, `fitted_levels` and custom checks.

**Data formats:** CSV, TSV, Parquet (a file, or a directory of parquet files), Feather,
JSON, JSONL. Pickle is refused unless `allow_pickle = true`, because loading a pickle runs
code.

### Outputs: what the step writes

| output name | content |
|---|---|
| `report` | JSON report (recommended). Deterministic: no timestamps, so downstream steps stay cached. |
| `summary` | Markdown: one table row per check, blocking ones first, then "How to fix". |

Both are optional; any other output name is an error. The report is written **before** the
step fails, so you can always read why it blocked.

### Result

**All blocking checks pass.** The step succeeds; the status line shows
`checks=… passed=… warnings=… skipped=…`.

**A blocking check fails.** The step raises `VerificationFailed` with each failure and its
fix. data-pipeline marks the step failed, prints that message, and halts (the step is
critical by default).

- A check that crashes counts as failed: an unverified check never passes silently.
- The step's log lists only failures and warnings; the report has every check.

**The params are invalid** (unknown key, wrong type, missing column for a rule). The step
raises `SpecError`, which names the key and suggests the closest valid one. Outputs of a
previous run are deleted first, so a stale report never survives a failed run.

**What blocks** is set by `fail_on`:

| `fail_on` | blocks on |
|---|---|
| `"error"` (default) | failed checks of severity error |
| `"warn"` | warnings too |
| `"never"` | nothing (report only) |

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
inputs are missing report `skip` with the reason.

The tables below are the output of `data-verify params --markdown`.

### Data layout

| key | value | meaning |
|---|---|---|
| `splits` | list \| table | Split inputs in time/workflow order. Default: inputs named train/val/valid/validation/dev/calib/test/holdout/eval/oot, in that order. A table maps split name -> input name. With split_column: the split values to expect (default: the standard names above; any other value in the column fails schema.splits_present). |
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
| `time_bounds` | table | split = { min = ..., max = ... } (inclusive). Example: train = { max = "2023-12-31" }, test = { min = "2024-01-01" }. Time columns with a timezone are compared in UTC: give such bounds in UTC or with an offset ("2023-12-31T23:59:59-05:00"). |
| `time_order` | string \| list | "train < val < test": every row of a left split is before every row of a split to its right (every pair of the chain is checked). |
| `embargo` | duration | Minimum gap between row timestamps for time_order: "7D", "12h", or a number of days (units when time is numeric). Use the label horizon when there is no label_time. |
| `time_order_scope` | global \| group | group: check time_order per entity (per-series forecasting) instead of globally; split.group_overlap is then skipped, since entities span splits by design. |
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
| `assert` | list of tables | [[steps.params.assert]] name, and one of: expr (pandas expression true on every row of `dataset`, default every split), sql (DuckDB query returning violating rows), sql_count (DuckDB query returning the number of violations). Optional severity, category, description, hint. ${var} is replaced from vars; write $$ for a literal $. In expr, quote column names that are not identifiers in backticks (`my column` > 0). In SQL, every data input and every split is a view under its name. |
| `vars` | table | Values for ${name} in assertions (refs allowed). |
| `custom` | list | "module:function" checks, called with the Verifier; call v.add(name, ok, message, severity=...) or return a list of dicts. |

### Behaviour

| key | value | meaning |
|---|---|---|
| `checks` | table | Per-check overrides: "<id>" = "off" \| "warn" \| "error", or "<id>" = { severity = ..., <option> = ... }. Globs allowed for severity ("drift.*" = "off"); an exact id beats any glob, and among globs the most specific (most non-wildcard characters) wins. Options per check: `data-verify checks`. |
| `fail_on` | error \| warn \| never | error (default): failed error-severity checks block. warn: warnings block too. never: report only. |
| `sample_rows` | int | Row cap for statistical checks (default 250000). Exact checks (overlaps, duplicates, time) always use every row. |
| `seed` | int | Seed for sampling and cross-fitting (default 0). |
| `allow_pickle` | bool | Read .pkl inputs (off: loading a pickle runs code). |
| `max_hash_mb` | number | Inputs above this size are not hashed in the report unless pinned (default 4096). |
| `max_examples` | int | Example rows/values per finding (default 5). |

### References: one source of truth

Any value can be read from a JSON/TOML/YAML input instead of being copied into the pipeline
file. That way the gate checks the same contract the other steps use:

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

### Random row split (i.i.d. rows)

`label` is enough. Add `id` if rows have a key.

### Grouped data

Several rows per patient, user, device, session or document: set `group`. This is the most
common leak, and without `group` the gate cannot see it.

### Time-dependent data

Forecasting, churn, failure, fraud: set `time`, declare how the splits are ordered in time,
and say when labels become known:

```toml
[steps.params]
label = "fails_30d"
group = "device_id"
time = "snapshot_date"
label_time = "label_known_date"      # e.g. snapshot_date + 30 days, or the event date
time_order = "train < val < test"
embargo = "30D"                      # at least the label horizon
```

- A chain `"a < b < c"` checks every pair (a < b, a < c, b < c).
- `embargo` is the minimum gap between the last row of the earlier split and the first row
  of the later one.
  - Without `label_time`, set it to at least the label horizon.
  - With `label_time`, `temporal.label_after_cutoff` checks directly that every
    earlier-split label was known before the later split starts. The embargo is then
    optional, and it is not added on top of that check.
- With `time` but no `time_order`/`time_bounds`, `temporal.overlap` warns that evaluation
  rows fall inside the training period.

### Per-series forecasting

Each store or sensor has its past in train and its future in test. Declare the entity as
`group` and set `time_order_scope = "group"`. Then:

- `split.group_overlap` steps aside, since entities span splits by design;
- `temporal.order` checks, entity by entity, that train rows end before test rows start.

```toml
[steps.params]
label = "demand_next_7d"
group = "store_id"
time = "date"
time_order = "train < test"
time_order_scope = "group"
embargo = "7D"
```

### One table with a split column

```toml
params = { split_column = "split", table = "snapshots", label = "y" }
```

### Text data

Text columns (average length >= 30 characters) get two extra checks, whatever the other
columns say:

- `duplicates.text_cross_split` (blocking) compares them across splits after normalising
  case, punctuation and spacing;
- `poisoning.spurious_association` searches their words for backdoor triggers.

### A stricter gate

Some checks only warn because their signal has legitimate look-alikes: drift, a
backdoor-like token that is a genuine new pattern, id-like columns. When your data should
never show them, promote them:

```toml
[steps.params.checks]
"poisoning.spurious_association" = "error"
"target.id_like" = "error"
"target.strong_predictor" = "error"
"duplicates.near_cross_split" = "error"
```

Or set `fail_on = "warn"` to block on every warning.

### Unlabeled test split

```toml
[steps.params]
checks = { "schema.roles" = { unlabeled = ["test"] } }
```

### Project rules

For label definitions, censoring, sampling weights and the like, write SQL or pandas
assertions, or a Python check.
[examples/hard-drive-analysis/verify.toml](examples/hard-drive-analysis/verify.toml)
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

`${name}` in an assertion is replaced from `params.vars`; write `$$` for a literal `$`. In
`expr`, put column names that are not identifiers in backticks: `` `unit price` > 0 ``.

A **custom check** is a function that takes the verifier and calls
`v.add(name, ok, message, severity="error", details={}, hint="")`. The verifier exposes
`v.splits`, `v.train`, `v.dataset(name)`, `v.sql(query)`, `v.var(name)`, `v.features` and
`v.labels`. Put the check's file in the step's `watch` list so editing it reruns the gate.

## What it checks

51 built-in checks. `data-verify checks` lists them with their options, and
[docs/CHECKS.md](docs/CHECKS.md) describes each one. By area:

**split**
- Blocks: same `id` or `group` in two splits; repeated ids; split not matching
  `split_rule`; split rows missing from `source` (injected/altered rows).
- Warns: split shares off `split_ratios`; an undeclared entity-like column (`*_id`,
  `patient`, `device`, ...) shared across splits.

**duplicates**
- Blocks: identical feature rows in two splits (beyond what different entities share by
  chance in train); the same text (normalised) in two splits.
- Warns: near-copies across splits (rounded numbers, normalised text); mass-repeated rows
  in train; identical rows with conflicting labels.

**temporal**
- Blocks: rows outside `time_bounds`; splits overlapping in time (or per group); labels
  known only after the evaluation period starts; rows dated after their own outcome;
  features observed after the row's time; unparseable dates.
- Warns: time-stamped splits overlapping when no time rule is declared.

**target**
- Blocks: label/key/time columns or `forbidden` patterns used as features; a feature equal
  to the label on ≥ 99% of rows; one feature (or its missingness) that alone predicts the
  label with AUC / normalised accuracy / |Spearman| ≥ 0.99.
- Warns: single features ≥ 0.95; features far more predictive in one split (target
  encoding fitted with evaluation labels); outcome-like names; id-like features.

**preprocessing**
- Blocks: scaler fitted on train+evaluation data (standardised or min-max over the union
  only, also tested without repeated rows and on rows found in `source`, so later
  injections do not hide it); fitted vocabularies containing levels absent from training
  data.

**labels**
- Blocks: missing / out-of-domain labels, single-class train; evaluation classes absent
  from train.
- Warns: classes missing from a split; label distribution shift.

**weights**
- Blocks: negative / infinite / all-zero weights.
- Warns: a few rows holding most of the weight.

**poisoning**
- Blocks: pinned input changed (`pins`).
- Warns: excess of extreme values in train vs held-out data; backdoor-like value or text
  token tied to one class in train only.

**schema / integrity**
- Blocks: missing split, unknown split values, declared columns missing, columns or types
  differing between splits, `schema` rule violations, infinite feature values.
- Warns: empty or constant features.

**drift**
- Warns: feature distribution shift (PSI); unseen categories.

### When the gate blocks

1. **Read** the message and `hint`, or run `data-verify explain <report>`. Several problems
   can hide each other (rows injected after scaling blur the scaling signature, for
   instance), so rerun after each fix.
2. **Fix the data step** (split by group, cut by time, fit transforms on train only, drop
   the column) and rerun. That is the expected outcome almost every time.
3. **Only if the finding is a property of the domain**, tune the narrowest knob and say why
   in a comment:

   ```toml
   "duplicates.cross_split" = { max_fraction = 0.001 }  # identical sensor readings across machines are normal
   ```

   Turning a check `"off"` is a last resort.

### Limits

The gate sees data, not intentions:

- It cannot know that a column was computed from the future unless `time`, `label_time` or
  `point_in_time` say when things happened.
- It finds statistical traces of poisoning (copies, outliers, triggers, shifted labels),
  but not an attack crafted to match the clean distribution.
- It trusts the roles you declare.

Declare `group`, `time` and `label_time` whenever they exist.

## Caching

- data-pipeline fingerprints `data_verify/__init__.py` (an installed module), not the rest
  of the package. The version string lives there, so upgrading data-verify reruns the gate.
- After editing data-verify itself without a version bump (with an editable install, any
  edit to the source), run `data-pipeline -r verify`.
- The gate's own params, inputs and `watch` files invalidate it as usual.

## Command line

```
data-verify run pipeline.toml [--step NAME]   run a data-verify step without the orchestrator
data-verify run verify.toml                   standalone config: [inputs] [outputs] [params]
data-verify lint [pipeline.toml] [--step N]   validate params without running anything
data-verify explain REPORT.json [--all]       failed checks, details and fixes
data-verify checks [--json | --markdown]      every check, severity and options
data-verify params [--markdown]               every params key
data-verify template                          step block to paste into pipeline.toml
```

Exit codes: `0` passed, `1` blocked, `2` invalid config.

## Development

From the repo root (one workspace, one virtual environment):

```bash
uv sync
uv run pytest packages/data-verify     # injects every leak/poisoning pattern into clean data
uv run data-verify run packages/data-verify/examples/hard-drive-analysis/verify.toml
                                       # needs ../hard-drive-analysis-v3/build next to the repo
```

Layout of `src/data_verify/`:

- `spec.py`: params parsing
- `core.py`: loading, hashing, registry, Verifier
- `checks/*.py`: one module per area; a check is a function decorated with `@check`
- `stats.py`: single-feature scores, PSI, robust statistics
- `step.py`: runner, report
- `cli.py`
- `reference.py`: params and checks reference
