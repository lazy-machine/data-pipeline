# What it checks

55 built-in checks. `data-verify checks` lists them with their options, and
[CHECKS.md](CHECKS.md) describes each one. Checks that match rows exactly (ids,
`source` conservation, identical rows across splits) compare numbers as numbers: `3`, `3.0`
and `True == 1` match across CSV and parquet, while integers and timestamps keep every digit,
so ids above 2**53 or times a nanosecond apart stay distinct. Rows are matched by a SipHash
of each value's type and exact value, so an altered row can't be computed to pass for a
source row. A date is not the integer of its nanoseconds, missing text is not any string,
and in a column of mixed values (JSON, categories) `1` is a number, not the text `"1"`.
Missing values match each other (`None`, `NaN`, `NA`, `NaT`). Other objects, such as
lists, arrays and pickled values, match by type and full `repr`: exact for plain data,
but an object whose `repr` hides its state or shows its address is not compared exactly. By area:

**split**
- Blocks: same `id` or `group` in two splits; repeated ids; split not matching
  `split_rule`; split rows missing from `source` (injected/altered rows).
- Warns: split shares off `split_ratios`; an undeclared entity-like column (`*_id`,
  `patient`, `device`, ...) shared across splits.

**duplicates**
- Blocks: identical feature rows in two splits (beyond what different entities share by
  chance in train); the same text (normalised) in two splits.
- Warns: near-copies across splits (rounded numbers, normalised text, ignoring undeclared
  key columns); the split pairs that check does not report — rows that repeat across and
  within splits (copies made before the split, or coarse columns, reported as repeats) and,
  when it is off or tolerates them, rows that match more than the repeats explain;
  mass-repeated rows in train; identical rows with conflicting labels.

**temporal**
- Blocks: rows outside `time_bounds`; splits overlapping in time (or per group); labels
  known only after the evaluation period starts; rows dated after their own outcome;
  features observed after the row's time; unparseable dates.
- Warns: time-stamped splits overlapping when no time rule is declared; date-like columns
  when `time` is not declared.

**target**
- Blocks: label/key/time columns or `forbidden` patterns used as features; a feature equal
  to the label on ≥ 99% of rows; one feature (or its missingness) that alone predicts the
  label with AUC / normalised accuracy / |Spearman| ≥ 0.99.
- Warns: single features ≥ 0.95; features far more predictive in one split (target
  encoding fitted with evaluation labels); outcome-like names; id-like features.

**preprocessing**
- Blocks: scaler fitted on train+evaluation data (standardised or min-max over the union
  only, closer than chance allows, also tested without repeated rows and on rows found in
  `source`, so later injections do not hide it); fitted vocabularies containing levels
  absent from training data.
- Warns: features nearly standardised over the union, where chance can't be ruled out.

**labels**
- Blocks: missing / out-of-domain labels, single-class train; evaluation classes absent
  from train.
- Warns: classes missing from a split; label distribution shift.

**weights**
- Blocks: negative / infinite / all-zero weights.
- Warns: a few rows holding most of the weight.

**poisoning**
- Blocks: pinned input changed (`pins`).
- Warns: training labels fitting the features worse than evaluation labels; excess of
  extreme values in train vs held-out data; backdoor-like value or text
  token tied to one class in train only.

**schema / integrity**
- Blocks: missing split, unknown split values, declared columns missing, columns or types
  differing between splits, `schema` rule violations, infinite feature values.
- Warns: empty or constant features.

**drift**
- Warns: feature distribution shift (PSI); unseen categories.

## What it cannot see

A pass means that no blocking check fired, not that the data is clean.

- **Undeclared roles.** Without `group` or `time`, leaks across entities or across time at
  most warn. With `label` alone, a grouped dataset split by row passes with a warning.
- **Poisoning.** `poisoning.label_consistency` warns about label noise of roughly 15% and up
  on a few thousand rows, and about targeted flips of a subgroup, when numeric features
  predict a binary or regression label. Lower flip rates and a backdoor with a rare trigger
  look like ordinary noise. A shift in class balance doesn't fire it, but features that vary
  more in an evaluation split can, with either kind of label, so read it as a prompt to
  compare labels with their source, not as proof. Seen on real data: a clean phoneme segment
  split (`V1_top25`, test the top quartile) fired 5/5 and adult `race_nonwhite` 1/5, both by
  that covariate shift and not by label noise. Other injected values and rows mostly warn. `schema.ranges`,
  `schema.allowed` and `pins` block, so declare them when you know the valid values.
- **Near-copies under a new key.** With a unique key column that isn't declared as `id`, the
  check also compares the other columns, at two precisions. The fine comparison follows the
  size of the training split: numbers equal to `ceil(log2(train rows)) + 7` bits of mantissa,
  between 20 and 48 (about 6 significant digits at 15,000 training rows, 8 at 150,000), and it
  needs at least 4 standard errors more matches than the two splits' own repeats predict (wider
  than nominal because the counts are overdispersed). That last bar is why a rounded or derived
  column alone is not treated as a copy: the
  shuffled chance estimate collapses, but the splits repeat those vectors as often as they
  share them. The allowance is the larger of what each split's own repeats predict for the
  pair, and a copy added to the training split occurs in the evaluation split, so it is left
  out of both: it is not what the repeats are measured against and cannot hide itself, up to 5
  placements (a sixth copy stops the value being rare). An evaluation split that sits where the
  features are dense is
  explained. A copy jittered past the fine precision still agrees to about 4 significant
  digits, so a comparison at 14 bits needs the same. When no single column is distinct enough
  to go on, the whole row without the keys is the evidence. Between them they find exact
  copies and jitter up to about 1e-5 relative when at least 10 rows match; a handful of copies
  is below the floor, whatever the jitter. `duplicates.repeated_across_splits` warns about
  every split pair that `near_cross_split` does not report: rows that match no more often than
  the splits' own repeats (a warning, not a copy claim), and — when `near_cross_split` is off
  or its own settings let the pair through — rows that match more than those repeats explain.
  The first covers copies made before the split (which also repeat inside the evaluation
  split) and coarse columns that repeat by nature, including a column rounded to about 4
  decimals with another derived from it. Post-split copies in a frame whose rows still repeat — a 5-item
  Likert scale, correlated Likert items, one-hot categories, a sparse binary matrix, a small
  grid, zero-inflated columns, or clustered rows — are 0/20 on the second look and 20/20 on
  the warning with 100 copies placed three times, so only the warning sees them; a denser
  frame (10 Likert items, a 30-column 20%-dense matrix, a 6-column 0-9 grid) does fire. Copies
  of a common vector fire neither check. The comparison on all columns runs first and fires on
  its own: it hits dense and partly overlapping evaluation segments in 1–10 of 60 seeds, at the
  same rate on `main` and here. The keyless look is not silent either: on clean data whose
  features repeat heavily it fires on about 0.3% of runs. So promoting this check to `error` is
  safe with respect to what it claims, but not free of false alarms. Copies are
  still missed when fewer than 10 rows match, when a copy is jittered by 1e-4 or more or
  re-measured, when a row is copied more than 5 times, when a copy carries a common value or a
  fresh timestamp or other per-row value, and in data whose columns are all coarse; declaring
  the keys (`id`, `group`) leaves only the comparison at `bits`. These all warn at best.
- **Leaks that look like signal.** A single feature only blocks at AUC 0.99 and warns from
  0.95, so a leak weaker than that passes. Two columns that leak only together pass too.
- **Preprocessing that leaves little or no signature.** Standardisation and min-max scaling
  fitted on evaluation rows block when the signature is exact: closer to mean 0 and std 1
  than chance allows, with an allowance for rounding to 2 or more decimals. A standardising
  scaler that misses that bar can't be told from chance on large data, so it warns at best
  (`preprocessing.global_scaling_near`). That covers coarser rounding, values kept to
  significant figures, and float32 data with a large offset. Features with only two values
  are not tested, so a scaled flag goes unseen. In the other direction, a coarse feature with
  a few values that happens to be exactly standard still blocks a clean pipeline, because
  nothing tells it from a scaler. Rows added or dropped after
  scaling often give no signal at all, and the more rows changed the less likely any signal
  is. The warning also fires on features that are roughly standard by nature, so a project
  with many of those will often see it. Target encoding and imputation leave no signature.

To be stricter, set `checks = { "poisoning.*" = "error" }`, or `fail_on = "warn"` to block on
every warning. Expect false alarms on heavy-tailed data. Fit preprocessing inside a
pipeline after the split, since the gate can't confirm that from the data.

## Limits

The gate sees data, not intentions:

- It cannot know that a column was computed from the future unless `time`, `label_time` or
  `point_in_time` say when things happened.
- It finds statistical traces of poisoning (copies, outliers, triggers, shifted labels),
  but not an attack crafted to match the clean distribution.
- It trusts the roles you declare.

Declare `group`, `time` and `label_time` whenever they exist.
