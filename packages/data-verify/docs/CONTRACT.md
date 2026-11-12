# The contract

What the step reads, what it writes, and how it ends.

## Inputs: what the step reads

The input **names** say what each file is:

| input name | meaning |
|---|---|
| `train` | training split |
| `val`, `valid`, `validation`, `dev`, `calib` | validation split |
| `test`, `holdout`, `eval`, `oot` | test split |
| `spec` | a TOML/JSON file holding the params; step params override it |
| a `.json` / `.toml` / `.yaml` input used by a `{ from = ... }` reference | configuration, not data (see [References](PARAMS.md#references-one-source-of-truth)) |
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

## Outputs: what the step writes

| output name | content |
|---|---|
| `report` | JSON report (recommended). Deterministic: no timestamps, so downstream steps stay cached. |
| `summary` | Markdown: one table row per check, blocking ones first, then "How to fix". |

Both are optional; any other output name is an error. The report is written **before** the
step fails, so you can always read why it blocked.

## Result

**All blocking checks pass.** The step succeeds; the status line shows
`checks=… passed=… warnings=… skipped=…`, plus `warned=…` naming up to three warnings.

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

## Report format

```jsonc
{
  "tool": "data-verify", "version": "0.4.1",
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
