# Params

Every key is optional. The more roles you declare, the more checks can run; checks whose
inputs are missing report `skip` with the reason.

The tables below are the output of `data-verify params --markdown`.

## Data layout

| key | value | meaning |
|---|---|---|
| `splits` | list \| table | Split inputs in time/workflow order. Default: inputs named train/val/valid/validation/dev/calib/test/holdout/eval/oot, in that order. A table maps split name -> input name. With split_column: the split values to expect (default: the standard names above; any other value in the column fails schema.splits_present). |
| `split_column` | column | One table holds every split; this column says which split each row is in. |
| `table` | input | With split_column: the input holding the table (needed if there are several data inputs). |
| `train` | split | The split the model is fitted on. Default: the first train-like split. |
| `source` | input | Pre-split data. Enables split.conservation: no split row may be missing from it. |

## Column roles

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

## Rules

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

## Extensions

| key | value | meaning |
|---|---|---|
| `assert` | list of tables | [[steps.params.assert]] name, and one of: expr (pandas expression true on every row of `dataset`, default every split), sql (DuckDB query returning violating rows), sql_count (DuckDB query returning the number of violations). Optional severity, category, description, hint. ${var} is replaced from vars; write $$ for a literal $. In expr, quote column names that are not identifiers in backticks (`my column` > 0). In SQL, every data input and every split is a view under its name. |
| `vars` | table | Values for ${name} in assertions (refs allowed). |
| `custom` | list | "module:function" checks, called with the Verifier; call v.add(name, ok, message, severity=...) or return a list of dicts. |

## Behaviour

| key | value | meaning |
|---|---|---|
| `checks` | table | Per-check overrides: "<id>" = "off" \| "warn" \| "error", or "<id>" = { severity = ..., <option> = ... }. Globs allowed for severity ("drift.*" = "off"); an exact id beats any glob, and among globs the most specific (most non-wildcard characters) wins. Options per check: `data-verify checks`. |
| `fail_on` | error \| warn \| never | error (default): failed error-severity checks block. warn: warnings block too. never: report only. |
| `sample_rows` | int | Row cap for statistical checks (default 250000). Exact checks (overlaps, duplicates, time) always use every row. |
| `seed` | int | Seed for sampling and cross-fitting (default 0). |
| `allow_pickle` | bool | Read .pkl inputs (off: loading a pickle runs code). |
| `max_hash_mb` | number | Inputs above this size are not hashed in the report unless pinned (default 4096). |
| `max_examples` | int | Example rows/values per finding (default 5). |

## References: one source of truth

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
