### integrity

#### `integrity.pins` (error)

**Pinned inputs are byte-identical.** Every input listed in params.pins must have exactly that sha256 (directories are hashed recursively). Catches a raw file that was swapped, re-downloaded or edited, which is the simplest way to poison a dataset.

### schema

#### `schema.splits_present` (error)

**Every split has rows.** Each split must hold at least `min_rows` rows. In split_column mode, every row must belong to a declared split (no null or unknown split values).

Options: `min_rows = 1`

#### `schema.roles` (error)

**Declared columns exist in every split.** The label, id, group, time, weight and feature columns named in params must exist in every split. Splits listed in the `unlabeled` option may lack the label columns.

Options: `unlabeled = []`

#### `schema.consistent` (error)

**Splits share the same columns and types.** All splits must have the same set of columns, and each column the same kind of type (numbers, text, dates, booleans). A column present in one split only, or parsed as text in one and numbers in another, means the splits were produced differently.

#### `schema.expected` (error)

**Columns match params.schema.columns.** When params.schema.columns is set, each split must have exactly those columns (and in that order when schema.ordered = true). Catches stray columns that could carry the label.

#### `schema.dtypes` (error)

**Column types match params.schema.dtypes.** params.schema.dtypes maps column -> kind (bool, int, float, numeric, datetime, string) or an exact pandas dtype name ("float32"). Every split must match.

#### `schema.ranges` (error)

**Values stay within params.schema.ranges.** params.schema.ranges maps column -> [min, max] or { min, max }. Values outside it are corrupt or injected (a classic poisoning signature: impossible values in train only).

#### `schema.allowed` (error)

**Values are in params.schema.allowed.** params.schema.allowed maps column -> list of permitted values.

#### `schema.not_null` (error)

**Columns in params.schema.not_null have no nulls.** Columns listed in params.schema.not_null must be complete in every split.

#### `schema.unique` (error)

**Keys in params.schema.unique are unique.** params.schema.unique lists columns (or lists of columns) that must not repeat within each split and auxiliary dataset.

### integrity

#### `features.non_finite` (error)

**No infinite feature values.** Numeric features must not contain +inf or -inf (NaN is allowed and reported elsewhere). Infinities usually come from a division by zero in feature engineering.

#### `features.empty` (warn)

**No feature is empty in train.** A feature that is entirely null in the training split carries no information and often means a join or rename failed upstream.

#### `features.constant` (warn)

**No feature is constant in train.** A feature with a single non-null value in the training split cannot help the model; a constant that differs between splits is a split-specific artifact.

### split

#### `split.id_unique` (error)

**Row ids are unique within each split.** The params.id key must identify one row per split. Repeated ids inside a split mean rows were duplicated (an oversampled or injected copy).

#### `split.id_overlap` (error)

**No row id is in two splits.** The same params.id value in train and in val/test means the same record is both learned and scored: the evaluation is contaminated.

#### `split.group_overlap` (error)

**No entity (group) is in two splits.** Rows sharing params.group (a patient, user, device, document, ...) must all be in the same split. Otherwise the model can memorise the entity in train and recognise it in test: the most common leak in grouped and longitudinal data. Skipped when time_order_scope = "group" (per-series forecasting: every entity has an early part in train and a later part in test by design, and temporal.order checks that per entity).

#### `split.rule` (error)

**Split assignment follows params.split_rule.** Recomputes each row's split from params.split_rule and compares. Methods: a stable hash (sha256/sha1/md5 of template.format(key), first 8 bytes as a fraction, times modulo, then [lo, hi) buckets per split), `modulo` (integer key % modulo), or explicit `values` lists. Proves the split is deterministic and was not hand-edited.

#### `split.ratios` (warn)

**Split sizes match params.split_ratios.** Each split's share of rows (or of groups, with unit = "groups") must be within `tolerance` of params.split_ratios. A train share far above plan can mean rows from the evaluation splits were moved into train.

#### `split.conservation` (error)

**Splits come from params.source, nothing added.** Compares the splits with the pre-split data in params.source. Every split row must exist in the source (rows that do not were injected or altered: poisoning). With exact = true, every source row must also land in exactly one split (nothing dropped or duplicated). Rows are matched on params.id when set, otherwise on the `columns` option, otherwise on every column the source and the splits share.

Options: `exact = False`, `columns = []`

#### `split.undeclared_group` (warn)

**Entity-like columns crossing splits are declared as params.group.** Runs when params.group is not set. Looks for columns that look like entity keys (named like *_id, user, patient, device, serial, ..., or text with many distinct values) whose values repeat over several rows (at most `max_rows_per_value` on average) and are shared by more than `min_shared` of an evaluation split's rows with train. If rows with the same value belong to one entity, they leak between splits; declaring the column as params.group turns this warning into the blocking split.group_overlap check.

Options: `max_rows_per_value = 200`, `min_shared = 0.05`

### temporal

#### `temporal.parse` (error)

**Time columns parse.** Every non-null value of params.time, params.label_time and params.point_in_time must parse as a date/time (or be numeric). Unparseable values silently escape every temporal check.

#### `temporal.bounds` (error)

**Each split stays inside params.time_bounds.** params.time_bounds gives each split a min and/or max timestamp (inclusive), e.g. train = { max = "2023-12-31" }, test = { min = "2024-01-01" }. Rows outside their window are rows from the wrong period.

#### `temporal.order` (error)

**Earlier splits end before later ones start.** For every pair in params.time_order (a chain 'train < val < test' means every pair: train < val, train < test, val < test), the latest timestamp of the earlier split must be before the earliest of the later split, by at least params.embargo when set. With time_order_scope = "group", this is checked per entity instead (each group's train rows end before its test rows start), for per-series forecasting.

#### `temporal.overlap` (warn)

**Splits of time-stamped data do not overlap.** When params.time is set but neither time_order nor time_bounds is, warns if the training period overlaps an evaluation split's period. A random split of time-dependent data lets the model learn from the future it is scored on. Declare time_order (or turn this check off if rows are truly independent of time).

#### `temporal.rows_after_outcome` (error)

**No row is dated after its outcome.** params.label_time is when each row's label became known (event date, end of the label window, censoring date). A row whose own timestamp is after its label time was observed after the outcome, so its features can describe the outcome itself (e.g. readings after a failure, records after a diagnosis).

Options: `allow_equal = True`

#### `temporal.label_after_cutoff` (error)

**Training labels were known before the evaluation period.** A label of an earlier split must be known before the later split starts: for 'a < b' in time_order (or bounds where a's max is before b's min), every label_time of a must be before the first timestamp of b (and before b's time_bounds min). A training label that resolves inside the evaluation period was read from the future (e.g. a 'fails within 30 days' label for a row 10 days before the cutoff). The embargo is not added here: it is the gap temporal.order requires between row timestamps, and is usually set to the label horizon for data without label_time. With label_time, an embargo is optional.

#### `temporal.point_in_time` (error)

**Features were observed by the row's time.** Each column in params.point_in_time holds when a feature (or a joined record) was observed. It must not be after the row's params.time: a feature from the future is unavailable at prediction time.

### duplicates

#### `duplicates.cross_split` (error)

**No example appears in two splits.** Rows with identical feature values in two splits (compared on all features, numbers normalised across file formats). Fails when the share of a later split's rows that also occur in an earlier split exceeds `max_fraction` (default 0). When params.group is set (and `baseline` is true), different entities can legitimately look identical (e.g. new devices with no readings yet), so the allowance becomes `ratio` times the share of training rows that match a row of another training entity: copies beyond what chance produces inside train still fail. `include_label = true` only counts copies that also share the label. `columns` compares on those columns instead of all features.

Options: `max_fraction = 0.0`, `include_label = False`, `baseline = True`, `ratio = 2.0`, `columns = []`

#### `duplicates.near_cross_split` (warn)

**No near-copy of an example appears in two splits.** Like duplicates.cross_split, after rounding numbers to `bits` bits of mantissa (10 is about 3 significant digits) and normalising text (case, punctuation, spacing). Catches copies that went through a lossy round trip, re-encoding or light editing. Exact copies are not counted again. Same group baseline as duplicates.cross_split.

Options: `max_fraction = 0.0`, `bits = 10`, `baseline = True`, `ratio = 2.0`, `columns = []`

#### `duplicates.text_cross_split` (error)

**No text appears in two splits.** For each text feature (average length at least `min_chars`), texts that are identical after normalising case, punctuation and spacing, and at least `min_chars` long, must not occur in two splits. Compared column by column, so a copied document with a new id, date or other metadata is still found. Same group baseline as duplicates.cross_split.

Options: `min_chars = 30`, `max_fraction = 0.0`, `baseline = True`, `ratio = 2.0`

#### `duplicates.within_split` (warn)

**Training data is not dominated by repeated rows.** Inside the training split: the share of rows that repeat an earlier row must stay below `max_fraction`, and no single feature vector may make up more than `max_share` of the rows (minimum 10 copies). Mass-injected copies are a cheap poisoning attack; they also skew the class balance, so the per-label repeat rate is reported too.

Options: `max_fraction = 0.05`, `max_share = 0.01`

#### `duplicates.conflicting_labels` (warn)

**Identical training examples agree on their label.** Among training rows with identical features, the share of rows whose label disagrees with the majority label of their copies. Above `max_fraction` this points at label noise or flipped labels (label-flipping poisoning), or at missing features.

Options: `max_fraction = 0.01`

### target

#### `target.forbidden_features` (error)

**No label, key, time or forbidden column is used as a feature.** The features must not include any role column (label, id, group, time, label_time, weight, split column, point_in_time columns) nor anything matching params.forbidden (glob patterns, e.g. "*_after_*", "outcome_*"). With features = "auto" every non-role column is a feature, so a leaky column left in the table fails here.

#### `target.suspicious_names` (warn)

**No feature name hints at the answer.** Feature names that read like an outcome or a look-ahead (target, label, outcome, future, next_, post_, after, result, score, prediction, ...). A name is only a hint, so this warns; confirm the column is known at prediction time, then list it in checks."target.suspicious_names".pattern or exclude it.

Options: `pattern = '(^|_)(target|label|labels|y|outcome|result|answer|truth|ground_?truth|leak|future|next|post|after|actual|response|class|prediction|pred|score|is_?fraud|churn(ed)?|default(ed)?|died|death|survived|failure|failed)(_|$)'`

#### `target.label_copy` (error)

**No feature is a copy of the label.** A feature equal to the label on at least `min_agreement` (default 99%) of the training rows, or (for numbers) an exact linear transform of it (|r| > 0.99999). Leaky columns are often noisy copies (a code entered after the outcome that matches it almost always).

Options: `min_agreement = 0.99`

#### `target.perfect_predictor` (error)

**No single feature (or its missingness) determines the label.** Scores each feature, and the missingness of each partly-null feature, as a standalone predictor of each label on the training split (AUC for binary, normalised accuracy for multiclass, |Spearman| for regression; cross-fitted, so ids score near chance). A score at or above `threshold` (default 0.99) means the label can be read off one column: real signals almost never do that alone, leaks (codes entered after the outcome, fields filled only for one class) routinely do.

Options: `threshold = 0.99`

#### `target.strong_predictor` (warn)

**No single feature is suspiciously predictive.** Same scores as target.perfect_predictor with a lower bar (default 0.95). Real signals rarely reach this alone; look at how the flagged columns are computed. Set its severity to "error" for a stricter gate.

Options: `threshold = 0.95`

#### `target.score_gap` (warn)

**Features predict the label equally well in every split.** Compares each feature's single-feature score in the training split with the same score in each evaluation split. A feature far more predictive in test than in train (gap above `max_gap` with the higher score at least `min_score`) was likely built with the test labels (target encoding or imputation fitted on all data); far more predictive in train only points at in-sample target encoding.

Options: `max_gap = 0.2`, `min_score = 0.8`

#### `target.id_like` (warn)

**No feature looks like a row identifier.** Features that behave like identifiers: row numbers ("Unnamed: 0", "index"), keys named *_id / uuid / key with a distinct value on (almost) every row, distinct text on every row, integers that count up through the rows (a dense sequence), or values that increase with the row order. They let a model memorise rows and often encode collection order. Cumulative counters and measurements with many distinct values are not flagged.

Options: `min_unique = 0.98`

### preprocessing

#### `preprocessing.global_scaling` (error)

**Scalers were fitted on the training split only.** Detects features standardised (mean 0, std 1) or min-max scaled (exactly 0..1) over train plus evaluation rows together while the training rows alone are not: the scaler was fitted on data the model is later scored on. Every union of splits that contains train is tried, so 'fitted on train+val' is caught too. Rows added after scaling would hide the signature, so it is also tested without repeated rows and, when params.source is set, on the rows found in the source only. `tol_mean` and `tol_std` are the widest gap allowed. On large data a match must also be closer than `chance_margin` of the sampling noise, plus what rounding to 2 or more decimals explains, so data that is already roughly standard doesn't block. Raise `chance_margin` to loosen this. A real scaler can miss that bar: on values rounded more coarsely or to significant figures, in float32 with a large offset, or with rows added or dropped afterwards. preprocessing.global_scaling_near warns about what is left within `tol_mean` and `tol_std`.

Options: `tol_mean = 0.0001`, `tol_std = 0.001`, `chance_margin = 0.003`

#### `preprocessing.global_scaling_near` (warn)

**No feature is almost scaled over evaluation rows.** Features within `tol_mean` and `tol_std` of mean 0 and std 1 over train plus evaluation rows that preprocessing.global_scaling doesn't block, because chance can't be ruled out. Either a scaler was fitted on evaluation rows and the result lost precision or changed rows afterwards, or the feature is roughly standard by nature, which on large data with many such features is common.

Options: `tol_mean = 0.0001`, `tol_std = 0.001`

#### `preprocessing.fitted_levels` (error)

**Fitted vocabularies come from training data only.** For each entry in params.fitted_levels (a category list, vocabulary or encoder classes that a step learned), every level except those in `allow` must occur in the training data (`column` of `dataset`, default the train split, optionally filtered by the pandas query `where`). A level seen only in evaluation data proves the encoder saw it. With top_k = true the levels must also be the most frequent training values (ties allowed).

### labels

#### `labels.valid` (error)

**Labels are present and valid.** Every row has a label; labels are within params.label_values / params.label_range when given; a classification training split has at least two classes. Out-of-domain label values are a sign of corrupted or injected rows.

#### `labels.unseen_classes` (error)

**Evaluation classes exist in training.** Every class in an evaluation split must occur in the training split. A class the model never saw cannot be predicted, so the split (or the labels) are broken.

#### `labels.class_presence` (warn)

**Each class is represented in each split.** Each training class should appear at least `min_count` times in every labelled split, so per-class metrics exist (and a class was not moved wholesale into one split).

Options: `min_count = 1`

#### `labels.shift` (warn)

**Label distribution is similar across splits.** Total variation distance between class frequencies (classification), or the shift in median relative to the training spread (regression), of each split against train. A large shift in a random split points at a broken split or at labels altered in one split (label-flipping).

Options: `max_distance = 0.15`

### weights

#### `weights.valid` (error)

**Sample weights are finite and non-negative.** params.weight must be finite, non-negative, and not all zero in any split. A negative or infinite weight lets single rows dominate (or invert) the training loss.

#### `weights.concentration` (warn)

**No few rows dominate the weights.** In the training split, the top 1% of rows by weight must hold at most `max_top_share` of the total weight, and the largest weight must be at most `max_ratio` times the median. Inflated weights on a handful of rows are a low-volume poisoning attack.

Options: `max_top_share = 0.5`, `max_ratio = 1000.0`

### poisoning

#### `poisoning.outliers` (warn)

**Training data has no excess of extreme values.** Share of training rows with at least one feature further than `z` robust deviations from the evaluation data's median, against the same share in the evaluation splits. Training having more than `ratio` times (+`slack`) as many extreme rows as held-out data looks like injected points. Uses the evaluation splits as the reference, so it needs two splits.

Options: `z = 20.0`, `ratio = 3.0`, `slack = 0.005`

#### `poisoning.spurious_association` (warn)

**No rare value or token ties to one class in training only.** Backdoor search. For each classification label, looks for feature values (categories, small-integer values, and words in text columns) that occur in at most `max_coverage` of training rows, at least `min_support` times, and come with one class in at least `purity` of them, while the same value in evaluation splits (at least `min_eval_support` rows) is tied to that class at least `min_drop` less often, or is (nearly) absent from them although their size predicts it there (so few rows has a Poisson chance below 0.001). Such a trigger in train that does not hold elsewhere is what a backdoor (trigger + target label) poisoning attack leaves behind.

Options: `min_support = 20`, `purity = 0.98`, `max_coverage = 0.2`, `min_drop = 0.3`, `min_eval_support = 5`

### drift

#### `drift.features` (warn)

**Feature distributions are stable across splits.** Population stability index of every feature in each evaluation split against train (10 quantile bins fitted on train; top 50 values for categories; missing as its own bin). Warns when more than `max_share` of the features exceed `max_psi`. Expected for time-based splits; in a random split, broad drift means the splits were produced differently.

Options: `max_psi = 0.25`, `max_share = 0.25`

#### `drift.unseen_categories` (warn)

**Evaluation categories were seen in training.** Share of evaluation rows whose categorical feature value never occurs in training (text columns excluded). High shares mean the encoder will fall back to 'unknown' for much of the evaluation data, or that the splits come from different sources.

Options: `max_fraction = 0.05`

