# Recipes

## Random row split (i.i.d. rows)

`label` is enough. Add `id` if rows have a key.

## Grouped data

Several rows per patient, user, device, session or document: set `group`. This is the most
common leak, and without `group` the gate cannot see it.

## Time-dependent data

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

## Per-series forecasting

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

## One table with a split column

```toml
params = { split_column = "split", table = "snapshots", label = "y" }
```

## Text data

Text columns (average length >= 30 characters) get two extra checks, whatever the other
columns say:

- `duplicates.text_cross_split` (blocking) compares them across splits after normalising
  case, punctuation and spacing;
- `poisoning.spurious_association` searches their words for backdoor triggers.

## A stricter gate

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

`duplicates.near_cross_split` is a candidate for `"error"` when you control how rows are
copied: a copy added to the training split cannot hide itself in its keyless second look, and
that look fires on none of the 61 clean scenarios measured at 60 seeds each. It is not free of
false alarms though. On clean data whose features repeat heavily (about 8% of rows) it fires
on about 0.3% of runs (11 of 3,600 over nine row-count cells at 400 seeds each; worst cell
3/400), and the comparison it runs on all columns first fires on dense or partly overlapping
evaluation segments in 1–10 of 60 seeds — on `main` and here alike — because those shapes
really do share rows. Run it as a warning first and look at what it flags.

Don't promote `poisoning.label_consistency`, alone or through `"poisoning.*" = "error"`, unless
your splits are drawn from the same distribution. It also fires on clean data when the
features vary more in an evaluation split than in train (a covariate shift, such as a wider
spread of values), and it would then block a run that has nothing wrong with its labels.

## Unlabeled test split

```toml
[steps.params]
checks = { "schema.roles" = { unlabeled = ["test"] } }
```

## Project rules

For label definitions, censoring, sampling weights and the like, write SQL or pandas
assertions, or a Python check.
[examples/hard-drive-analysis/verify.toml](../examples/hard-drive-analysis/verify.toml)
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
