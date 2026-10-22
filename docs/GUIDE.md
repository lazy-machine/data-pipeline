# Using data-pipeline: a practical guide

The [README](../README.md) is the reference for every option and key. This guide covers how
to use them: how to build a pipeline from scratch, how to write steps that cache well, how
to choose which steps are critical, and what to do when something goes wrong.

---

## 1. How it works

Keep these four facts in mind and the rest of the guide follows from them:

1. **Steps talk only through files.** Every step runs in its own fresh Python process.
   A DataFrame or model you build in one step is gone when that step ends. The only thing
   the next step sees is what you wrote to disk.
2. **A step is declared, not discovered.** The TOML file says what a step reads (`inputs`),
   what it writes (`outputs`), and how it's configured (`params`). The orchestrator relies
   on these declarations to order the steps, check them and cache them, so keep them
   accurate.
3. **Caching is based on content.** A step is skipped when its code, config and the bytes of
   its inputs are unchanged, and its outputs are still exactly as it left them.
4. **Failures are either fatal or tolerated.** A critical step that fails stops the run.
   A non-critical step that fails prints a warning, and only the steps that depend on it
   are skipped.

```
pipeline.toml ─► orchestrator ─► for each step, in order:
                                   verify ─► cached? ──yes──► ↷ skip
                                                │no
                                                ▼
                                   spawn worker process ─► run your function
                                                ▼
                                   outputs present? ─► record checkpoint ─► ✓
```

---

## 2. Build your first pipeline

Here's a small pipeline built from scratch: load sales data, aggregate it, and plot it.

### 2.1 Lay out the project

```
sales/
├── pipeline.toml
├── data/
│   └── sales.csv          # your raw data (an input that no step produces)
└── steps/
    ├── prep.py
    └── viz.py
```

Raw data goes in `data/`, code goes in `steps/`, and generated files go in a folder like
`build/` that you can delete at any time. Add `build/` and `.pipeline/` to `.gitignore`.

### 2.2 Write the step functions

Every step function takes one argument, `ctx`, and optionally returns a short summary:

```python
# steps/prep.py
import pandas as pd


def clean(ctx):
    df = pd.read_csv(ctx.inputs["raw"])
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > ctx.params["min_amount"]]
    df.to_csv(ctx.outputs["table"], index=False)
    return {"rows": len(df)}


def monthly(ctx):
    df = pd.read_csv(ctx.inputs["table"], parse_dates=["date"])
    out = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    out.to_csv(ctx.outputs["monthly"])
    return {"months": len(out), "total": float(out.sum())}
```

```python
# steps/viz.py
import matplotlib
matplotlib.use("Agg")          # no display in a worker process
import matplotlib.pyplot as plt
import pandas as pd


def plot(ctx):
    s = pd.read_csv(ctx.inputs["monthly"], index_col=0).squeeze()
    ax = s.plot(kind="bar", title="Monthly sales")
    ax.figure.tight_layout()
    ax.figure.savefig(ctx.outputs["chart"], dpi=120)
```

### 2.3 Declare the pipeline

```toml
# pipeline.toml
[pipeline]
name = "sales"
python_path = ["steps"]          # so "prep:clean" means steps/prep.py, function clean

[[steps]]
name = "clean"
function = "prep:clean"
inputs  = { raw = "data/sales.csv" }
outputs = { table = "build/clean.csv" }
params  = { min_amount = 0 }
requires = ["pandas>=2"]

[[steps]]
name = "monthly"
function = "prep:monthly"
inputs  = { table = "build/clean.csv" }     # same path as clean's output, so it runs after clean
outputs = { monthly = "build/monthly.csv" }
requires = ["pandas>=2"]

[[steps]]
name = "plot"
function = "viz:plot"
inputs  = { monthly = "build/monthly.csv" }
outputs = { chart = "build/monthly.png" }
requires = ["matplotlib"]
critical = false                 # a broken chart shouldn't block anything
```

The `ctx.inputs[...]`/`ctx.outputs[...]` keys in the code are the left-hand names in the
TOML (`raw`, `table`, `monthly`, `chart`). The paths can change without touching the code.

### 2.4 Run it

```bash
uv run data-pipeline sales/pipeline.toml -n   # dry run: checks everything and shows the plan
uv run data-pipeline sales/pipeline.toml      # real run
uv run data-pipeline sales/pipeline.toml      # again: everything shows ↷ cached
```

If you `cd sales` first, plain `uv run data-pipeline` works, because `pipeline.toml` is the
default.

---

## 3. Writing good steps

### The contract

| You get | You must |
|---|---|
| `ctx.inputs[name]`: absolute `Path` | read only the inputs you declared |
| `ctx.outputs[name]`: absolute `Path`, parent directory already created | write **every** declared output |
| `ctx.params`: dict from the TOML `params` table | raise an exception to signal failure |
| `ctx.workdir`: the pipeline's working directory (also the process's cwd) | |
| `ctx.log(...)`: `print` with a `[step-name]` prefix | |

The return value is shown on the status line:

- `None` → nothing extra
- a string → shown as is (`"19 lines"`)
- a flat dict → `key=value` pairs, with floats shortened to 4 significant digits
  (`{"accuracy": 0.947368}` → `accuracy=0.9474`)

Return a few numbers that tell you at a glance whether the step did something reasonable,
such as row counts, dropped rows or a headline metric. The full results belong in output
files.

### Do

- **Load everything from `ctx.inputs` and save everything to `ctx.outputs`.** If a step
  reads a file it didn't declare, the orchestrator can't see changes to that file and the
  step will be wrongly skipped.
- **Seed everything random** (`random_state`, `np.random.default_rng(seed)`, and so on), and
  put the seed in `params`. Deterministic outputs are what let later steps stay cached
  (see §6).
- **Fail loudly and early.** `raise ValueError("expected column 'species'")` gives you a
  useful traceback. Logging the problem and returning gives you a step that "succeeded"
  with bad output.
- **Keep one job per step.** Separate steps cache separately: changing the model shouldn't
  redo the cleaning.
- **Use `matplotlib.use("Agg")`** (or another non-interactive backend) in plotting steps.

### Don't

- **Don't rely on module-level state between steps.** Each step runs in a new interpreter,
  so a global cache filled in step A is empty in step B.
- **Don't call `sys.exit()` in a step.** Even `sys.exit(0)` counts as a failure. Return
  normally instead.
- **Don't write files you didn't declare** if anything downstream depends on them. Declare
  them as outputs, or write a directory and declare the directory (see §8).
- **Don't do interactive I/O.** The worker's stdin is closed and its stdout is captured.

### About `print`

Output is captured, not streamed. You see it **only if the step fails**, or with `-v`, and
only once the step has finished. That makes `print` fine for debugging, but useless as a
live progress bar for a long step. If you need a progress log while the step runs, write it
to a file yourself.

---

## 4. Connecting steps

### Ordering is inferred from paths

If step B lists an input whose path matches step A's output, B always runs after A, no
matter where they appear in the file. Otherwise steps run in file order. Paths are
compared after resolving them, so `build/x.csv` and `./build/x.csv` match.

### `depends_on` for ordering without a file

Use `depends_on` when B must run after A but doesn't read any of A's files, for example a
step that checks a database A just loaded:

```toml
[[steps]]
name = "verify_load"
function = "checks:row_counts"
depends_on = ["load_db"]
```

Note that `depends_on` only affects **order** and **failure propagation**. It doesn't
affect caching: since B reads nothing from A, a rerun of A doesn't invalidate B.

### Rules the loader enforces

When the file is loaded, the orchestrator rejects: dependency cycles, two steps writing the
same path, a step reading its own output, duplicate step names, unknown `depends_on`
targets, and misspelled keys (`colour = 1` is an error rather than being silently ignored).

---

## 5. Critical vs. non-critical

Use `critical = false` for steps whose failure leaves the important results intact.

| Usually **critical** (default) | Usually **non-critical** |
|---|---|
| loading, cleaning, validating data | exploratory statistics, plots |
| feature engineering, train/test split | reports, dashboards, exports |
| model training and evaluation | notifications, uploads of extra copies |
| anything a critical step reads | experimental or optional models |

How a failure spreads:

| What fails | Step that depends on it | Result |
|---|---|---|
| critical step | anything | **pipeline halts** (exit code 1) |
| non-critical step | non-critical step | dependent is `– skipped`, pipeline continues |
| non-critical step | **critical** step | the critical step can't run, so **pipeline halts** |
| non-critical step | unrelated step | runs normally |

The third row is the one that catches people out: **a step that a critical step depends on
is effectively critical.** If `report` (critical) reads `eda`'s output, marking `eda`
non-critical doesn't help. Either make `report` non-critical too, or stop `report` from
depending on `eda`.

Checks before the run follow the same rule. A missing package or input file in a critical
step stops the run **before anything executes** ("preflight failed — nothing was run"),
while the same problem in a non-critical step only produces a warning when that step's turn
comes.

---

## 6. Getting the most out of checkpoints

### What triggers a rerun

| You change… | Effect |
|---|---|
| the content of an input file | that step reruns (`inputs changed: <name>`) |
| `params`, `function`, input/output paths in the TOML | that step reruns (`config changed`) |
| the step's module file (even a comment) | every step in that module reruns (`code changed`) |
| a file listed in `watch` | that step reruns (`code changed`) |
| delete or edit an output | that step reruns (`output missing` / `output modified`) |
| the `description` or `critical` flag | nothing; these don't affect results |
| file timestamps only (`touch`) | nothing; caching compares content |

When a step reruns and writes **byte-identical** outputs, the steps after it stay cached.
For example, adding a comment to `prep.py` reruns `clean`, but `monthly` is skipped if
`build/clean.csv` comes out the same.

### Make outputs deterministic

Later steps stay cached only if a rerun produces identical bytes. Things that break this:

- unseeded randomness (always pass a seed through `params`)
- timestamps written into files (`"generated_at": now()`); leave them out, or put them in a
  separate output that nothing reads
- unordered collections serialized as-is: sort dict keys (`json.dumps(..., sort_keys=True)`)
  and sort rows before writing
- multithreaded floating-point reductions; some libraries vary slightly across runs, so
  fix the thread count if that matters

These don't make results wrong. They just mean later steps rerun more often than needed.

### Track helper modules with `watch`

Only the module file that contains the step function is fingerprinted. If
`steps/model.py` imports `steps/features.py`, editing `features.py` does **not**
invalidate the model step unless you add:

```toml
watch = ["steps/features.py"]
```

`watch` paths are relative to the working directory (the TOML file's folder by default), and
the files must exist. You can watch any file, including a lookup table, a SQL query or a
prompt template.

### Steps that read the outside world

A step that downloads data, queries a database, or reads the clock has no declared
input that changes. After one successful run it is **always cached**. To refresh it:

```bash
uv run data-pipeline -r fetch_prices          # rerun only that step (later steps follow if the data changed)
```

Or give it a parameter that you change on purpose, such as `params = { snapshot = "2026-10" }`.

### Resetting

- One step: `-r STEP` (repeatable).
- Everything: `-f`.
- Clear all saved state: delete `.pipeline/` (state lives in `.pipeline/state.json`).
  Deleting `build/` also forces reruns, because the outputs will be missing.

Checkpoints are content hashes, not absolute paths, so copying or moving a project
(state folder included) keeps everything cached.

---

## 7. Everyday commands

```bash
data-pipeline -n                    # what would run, and why? (runs nothing)
data-pipeline                       # run whatever is out of date
data-pipeline -v                    # same, and show each step's printed output
data-pipeline -r train              # rerun one step regardless of its checkpoint
data-pipeline -r train -r evaluate  # several
data-pipeline -o eda report         # run only these steps (their inputs must exist)
data-pipeline -f                    # rerun everything
```

(Prefix with `uv run` if the virtual environment isn't activated, and pass the TOML path if
it isn't `./pipeline.toml`.)

A typical loop while developing one step:

1. Edit `steps/model.py`.
2. `data-pipeline -n` confirms that only `train` and `evaluate` would rerun.
3. `data-pipeline` reruns just those; everything before them is skipped.

`--only` doesn't consider any step outside the list. It won't run their upstream steps, but
it won't block on them either. Use it once the upstream outputs exist.

### Exit codes (for scripts and CI)

| Code | Meaning |
|---|---|
| `0` | success, **including** runs with non-critical warnings |
| `1` | a critical step failed (or preflight failed) |
| `2` | invalid config file or unknown step name in `-r`/`-o` |
| `130` | interrupted with Ctrl-C |

In CI, `data-pipeline || exit 1` fails the build only on critical problems. If warnings
should fail the build too, check the output for `completed with warnings`.

---

## 8. Recipes

### Many files from one step: use a directory output

```toml
[[steps]]
name = "per_region"
function = "prep:split_by_region"
inputs  = { table = "build/clean.csv" }
outputs = { dir = "build/regions" }       # a directory
```

```python
def split_by_region(ctx):
    out = ctx.outputs["dir"]
    out.mkdir(exist_ok=True)               # only the *parent* of an output is created for you
    for region, df in pd.read_csv(ctx.inputs["table"]).groupby("region"):
        df.to_csv(out / f"{region}.csv", index=False)
```

Directories are hashed recursively (file names and contents), so they work as inputs to
later steps too. Remove stale files at the start of the step if region names can
disappear, otherwise old files remain in the directory and become part of its hash.

### Comparing models: one step per variant

There's no built-in parameter sweep. Declare a step per variant. Each one caches on its own
and can fail without affecting the others:

```toml
[[steps]]
name = "train_rf"
function = "model:train"
inputs  = { train = "build/train.csv" }
outputs = { model = "build/models/rf.pkl" }
params  = { estimator = "random_forest", seed = 42 }

[[steps]]
name = "train_logreg"
function = "model:train"
inputs  = { train = "build/train.csv" }
outputs = { model = "build/models/logreg.pkl" }
params  = { estimator = "logistic", seed = 42 }
critical = false                          # experimental

[[steps]]
name = "compare"
function = "model:compare"
inputs  = { rf = "build/models/rf.pkl", logreg = "build/models/logreg.pkl", test = "build/test.csv" }
outputs = { table = "build/model_comparison.csv" }
critical = false
```

### Shared settings

There's no templating or variable substitution in the TOML. To share a value such as a
label column or a seed across steps, either repeat it in each step's `params` (explicit,
and each step caches correctly), or put it in a small config file that each step lists as
an input:

```toml
inputs = { table = "build/clean.csv", settings = "config/settings.json" }
```

Since the file is an input, editing it reruns exactly the steps that use it.

### Validation steps

A step with no outputs is fine for checks: it runs, and while its inputs stay the same it's
cached. Make it critical to stop bad data from flowing on, and have later steps
`depends_on` it:

```toml
[[steps]]
name = "check_schema"
function = "checks:schema"
inputs = { raw = "data/sales.csv" }

[[steps]]
name = "clean"
depends_on = ["check_schema"]
# ...
```

### Long-running steps

Set a `timeout` (in seconds) on anything that could hang, such as network calls or big
training jobs. When it's exceeded, the worker process is killed and the step is reported as
`timed out after Ns`. This counts as a failure, so it halts the run or warns depending on
`critical`.

### Running from Python

```python
from data_pipeline import config
from data_pipeline.orchestrator import Orchestrator

result = Orchestrator(config.load("pipeline.toml")).run(rerun=["train"])
print(result.ok, result.halted_by)
for name, outcome in result.outcomes.items():
    print(name, outcome.status, outcome.detail, outcome.reason)
```

`run()` accepts `only=`, `force=`, `rerun=` and `dry_run=`, the same as the command-line
options.

---

## 9. Reading the output

```
▶ iris · 6 steps · examples/iris/pipeline.toml
[1/6] clean     ↷ cached            rows=149 dropped=1 classes=3
[3/6] eda       ⚠ warning    0.90s  exit code 1  [non-critical, continuing]
┌─ eda: exit code 1 ─ analysis:explore
  …full captured output and traceback…
└────────────────────────────────────────────────────────────
[4/6] train     ✓ ran        1.10s  estimator=random_forest train_acc=1  (config changed)
[6/6] report    – skipped           upstream step(s) did not complete: eda  [non-critical, continuing]
✓ completed with warnings · 2 ran · 2 cached · 1 warning(s) · 1 skipped · 3.08s
```

| Symbol | Meaning |
|---|---|
| `✓ ran` | executed successfully; the part in parentheses says why it had to run |
| `↷ cached` | skipped; the summary shown is from the last successful run |
| `• will run` | dry run only; `after X` means it waits on a step that will also run |
| `⚠ warning` | a non-critical step failed; the error is printed right below it |
| `– skipped` | a non-critical step couldn't run because something it depends on failed |
| `✗ FAILED` | a critical step failed; the error is printed and the run stops |

In a terminal, a dim `… running` placeholder appears while a step executes and is replaced
by its final line. Colors are disabled with `--no-color` or the `NO_COLOR` environment
variable, and automatically when output is redirected to a file.

---

## 10. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `config error: … unknown key(s)` | typo in the TOML | fix the key; allowed keys are listed in the README |
| `config error: … function must look like 'package.module:function'` | missing `:` | `function = "prep:clean"`, not `"prep.clean"` |
| `module not found: prep (searched …)` | module isn't on `python_path` | set `python_path = ["steps"]` (relative to the TOML file) or fix the module name |
| `package not installed: X` | missing from the virtual env | `uv add X`; the worker uses the same Python as the orchestrator |
| `package version mismatch` | installed version doesn't satisfy the specifier | upgrade, or relax `requires` |
| `input 'raw' not found: data/x.csv` | path is wrong, or relative to the wrong folder | paths are relative to `workdir` (default: the TOML file's folder), not to where you run the command |
| `step did not write output(s): …` | the function returned without saving a declared output | check every branch of the function writes it; check the output key names |
| `worker exited without reporting a result` | the step process ended abnormally before finishing | look at the dumped output; often an extension module crashed |
| `killed by signal 9` | usually the OS ran out of memory | process the data in chunks, use smaller dtypes, or split the step |
| `exit code 1` with `SystemExit` in the traceback | the step called `sys.exit()` | return normally instead |
| `TypeError: Object of type date is not JSON serializable` (orchestrator crash) | a TOML date/time value in `params` | write dates as strings (`day = "2026-01-01"`) and parse them in the step; this is a known limitation |
| a step keeps rerunning even though nothing changed | an earlier step writes non-deterministic output | check `-n` to see which input changed; see §6 *Make outputs deterministic* |
| a step **doesn't** rerun after you edited a helper | helpers aren't fingerprinted | add the helper to `watch` |
| a step doesn't rerun after the source data changed | it reads external data with no declared input | use `-r STEP`, or declare the source file as an input |

When you're not sure why a step behaves the way it does, start with `-n`. The reason column
tells you what the orchestrator thinks changed.

---

## 11. Checklist for a new pipeline

- [ ] Raw data in `data/`, step code in `steps/`, generated files in `build/`.
- [ ] `build/` and `.pipeline/` are in `.gitignore`.
- [ ] Every file a step reads is listed in its `inputs`; every file it writes is in `outputs`.
- [ ] Every step lists its third-party packages in `requires`.
- [ ] All randomness is seeded through `params`.
- [ ] Local helper modules are listed in `watch`.
- [ ] Optional steps (plots, reports, experiments) are `critical = false`, and no critical
      step depends on them.
- [ ] Long or network-bound steps have a `timeout`.
- [ ] `params` values are strings, numbers, booleans, arrays or tables (no TOML dates).
- [ ] `data-pipeline -n` shows the order and reasons you expect.
- [ ] A second `data-pipeline` run shows every step `↷ cached`.
