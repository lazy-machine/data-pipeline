# data-verify

A gate step for [data-pipeline](../../README.md) projects, shipped in the same repository
(a uv workspace member). It reads the data your pipeline
produced (the train/val/test splits, plus any source or auxiliary tables) and looks for:

- **Data leakage**: anything that lets a model see the answers, or the future, of what it
  is scored on.
- **Poisoning**: tampered, injected or corrupted training data.

It stops the pipeline on what it can prove, which is most structural leakage once your
columns are declared. Most poisoning signs only warn, because a statistic rarely proves
tampering. See [What it cannot see](docs/COVERAGE.md#what-it-cannot-see).

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

## Quick start

Install it in the pipeline project:

```bash
uv add --editable "../data-pipeline[verify]"   # runner + gate, from a sibling checkout of data-pipeline
```

Add the step right after the step that writes the splits, and make every training and
evaluation step depend on it:

```toml
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { train = "build/train.parquet", val = "build/val.parquet", test = "build/test.parquet" }
outputs = { report = "build/verify/report.json", summary = "build/verify/report.md" }
requires = ["data-verify"]
params = { label = "churned", id = "row_id", group = "customer_id" }

[[steps]]
name = "train"
depends_on = ["verify"]        # nothing trains on data the gate has not passed
```

Run `data-verify lint` to validate the params without running anything. Declare every role
your data has (`group`, `time`, `label_time`, ...): the gate can only check what it knows.

## Documentation

| topic | |
|---|---|
| [Setup](docs/SETUP.md) | Installing, adding the step, declaring roles, caching |
| [The contract](docs/CONTRACT.md) | Inputs, outputs, result, report format |
| [Params](docs/PARAMS.md) | Every params key, and references to config files |
| [Recipes](docs/RECIPES.md) | Grouped, time-dependent, per-series and text data; stricter gates; project rules |
| [What it checks](docs/COVERAGE.md) | The checks by area, and what the gate cannot see |
| [Check reference](docs/CHECKS.md) | Every check, its severity and options |
| [When the gate blocks](docs/WHEN-IT-BLOCKS.md) | Reading a failure and fixing it |
| [Command line](docs/CLI.md) | `data-verify` commands and exit codes |
| [Development](docs/DEVELOPMENT.md) | Running the tests, code layout |

The [usage guide](../../docs/GUIDE.md) (§2 and §7) covers the other ways to depend on it and
where the gate goes in a pipeline; `examples/iris` at the repo root is a working example.
