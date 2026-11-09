# Using data-pipeline: a practical guide

The [README](../README.md) is the reference for every option and key. This guide covers how
to use them: how to set up a project that depends on the runner, how to build a pipeline,
how to write steps that cache well, where the data gate goes, how to build reusable steps
on top, and what to do when something goes wrong. The gate's own contract is in
[packages/data-verify/README.md](../packages/data-verify/README.md).

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
5. **Models only see data that passed the gate.** The `verify` step (data-verify) checks
   the splits for leakage and poisoning, and every training and evaluation step depends on
   it (§7). Passing means no check blocked, not that the data is clean: poisoning mostly
   warns, and some faults are invisible to it.

```
pipeline.toml ─► orchestrator ─► for each step, in order:
                                   verify ─► cached? ──yes──► ↷ skip
                                                │no
                                                ▼
                                   spawn worker process ─► run your function
                                                ▼
                                   outputs present? ─► record checkpoint ─► ✓
```

Three principles follow from these facts. Projects built on the runner, and changes to the
runner itself, should keep to them:

- **Minimal.** The runner depends on nothing but the standard library and `packaging`.
  Capabilities such as the gate are steps in their own packages, not runner features. A
  project adds only the dependencies its steps import.
- **Reusable.** A step is a function whose behaviour is fully set by its `inputs` and
  `params`. A step that more than one project needs goes into a package and is referenced
  as `function = "package:step"`, rather than copied (§11).
- **Verifiable.** Every file is declared, outputs are deterministic, failures are loud, and
  data reaches a model only through the gate. Anyone can rerun the pipeline and get the
  same bytes, or see exactly why not (`-n`).

---

## 2. Set up a project

A project is its own folder with its own `pyproject.toml` and virtual environment. It
depends on data-pipeline and never copies the runner's source: fixes and new checks then
reach every project, and each project's dependencies stay its own.

### 2.1 Depend on the runner and the gate

| You want | Command (run inside the project) |
|---|---|
| runner + gate, following a local checkout (the usual case) | `uv add --editable "../data-pipeline[verify]"` |
| a fixed, reproducible version | `uv add "data-pipeline[verify] @ git+file:///abs/path/to/data-pipeline@<commit>"` |
| the runner only (no model, no gate) | `uv add --editable ../data-pipeline` |

The `verify` extra installs data-verify (pandas, numpy, pyarrow). The paths assume the
project and `data-pipeline` are sibling folders. uv records them in the project's
`pyproject.toml` under `[tool.uv.sources]`, so `uv sync` restores them on another machine
with the same layout. Then add what your steps import (`uv add pandas scikit-learn`), plus
`duckdb` if the gate uses SQL assertions. The worker runs with the project's Python, so a
package your steps need must be installed in the project, not in data-pipeline.

With an editable install, changes to data-pipeline apply to every project the next time it
runs. A pinned commit changes only when you `uv add` a new one; use it for results you
have to reproduce later.

### 2.2 Lay out the project

```
sales/
├── pyproject.toml         # uv init --bare; depends on data-pipeline
├── pipeline.toml
├── data/
│   └── sales.csv          # your raw data (an input that no step produces)
└── steps/
    ├── prep.py
    └── viz.py
```

Raw data goes in `data/`, code goes in `steps/`, and generated files go in a folder like
`build/` that you can delete at any time. Add `build/` and `.pipeline/` to `.gitignore`.
In the commands below, `uv run` runs `data-pipeline` from the project's environment.

### 2.3 Or start from the Iris example

`examples/iris` is a complete project (clean → split → verify → EDA/train → evaluate →
report). It is the quickest correct starting point for a supervised model. Copy its files,
not the runner:

```bash
uv init --bare my-model && cd my-model
cp -r ../data-pipeline/examples/iris/{pipeline.toml,steps,data,.gitignore} .
uv add --editable "../data-pipeline[verify]"
uv add pandas scikit-learn matplotlib joblib
uv run data-pipeline
```

Then replace the data and the steps one at a time, rerunning as you go. Keep the `verify`
step and the `depends_on = ["verify"]` lines, and re-declare the gate's `params` for your
data (§7).

### 2.4 What not to do

- Don't copy `src/data_pipeline` or `packages/data-verify` into a project. Depend on them.
- Don't put project steps or data inside the data-pipeline repository. Only reusable
  packages belong there (§11).
- Don't install a project's step dependencies into data-pipeline's environment, and don't
  run a project's pipeline with `uv run --project ../data-pipeline`. Each project has its
  own environment.

---

## 3. Build your first pipeline

Here's a small pipeline built from scratch: load sales data, aggregate it, and plot it.
Set the project up as in §2.2 first.

### 3.1 Write the step functions

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

### 3.2 Declare the pipeline

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

### 3.3 Run it

From inside `sales/`, so that `uv run` uses the project's environment:

```bash
uv run data-pipeline -n    # dry run: checks everything and shows the plan
uv run data-pipeline       # real run
uv run data-pipeline       # again: everything shows ↷ cached
```

`pipeline.toml` is the default config. Pass a path to run another one.

This pipeline has no model, so it has no gate. As soon as a step fits or scores a model,
add one (§7).

---

## 4. Writing good steps

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
  (see §8).
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
  them as outputs, or write a directory and declare the directory (see §10).
- **Don't do interactive I/O.** The worker's stdin is closed and its stdout is captured.

### About `print`

Everything a step prints (stdout and stderr) is written, line by line as it arrives, to
`.pipeline/logs/<step>.log`. The file is replaced each time the step runs and left alone
when the step is cached. On the console you see the output **only if the step fails**, or
with `-v` once it has finished, or live with `-F/--follow`. So `print` works as a progress
log for a long step: follow it with `-F`, or with `tail -f` on the log file.

Output is read one line at a time, so progress bars that redraw a single line with `\r`
(tqdm and similar) only show up when they print a newline.

---

## 5. Connecting steps

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
same path or overlapping paths (`build/x` and `build/x/y.csv`), a step reading its own output
or a directory that holds it, duplicate step names, unknown `depends_on` targets, and
misspelled keys (`colour = 1` is an error rather than being silently ignored).

---

## 6. Critical vs. non-critical

Use `critical = false` for steps whose failure leaves the important results intact.

| Usually **critical** (default) | Usually **non-critical** |
|---|---|
| loading, cleaning, validating data | exploratory statistics, plots |
| feature engineering, train/test split, **the gate** | reports, dashboards, exports |
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

## 7. The data gate

A model scored on data it has already seen, or trained on tampered data, gives good
numbers that mean nothing, and no error ever shows up. The gate is there to catch this.
`data_verify:verify` reads the splits and fails, which halts the pipeline, when it finds
**leakage** (an id or entity in two splits, copied rows, a feature that encodes the label,
training rows from after the evaluation period, a scaler fitted on test data, …) or
**poisoning** (rows injected after the split, outliers, backdoor-like triggers, pinned
inputs that changed). Every check, passed or not, goes into a JSON report.

### Where it goes

```
… ─► split ─► verify ─┬─► train ─► evaluate ─► …
                      └─ (eda, reports: no need to wait for the gate)
```

Put it right after the step that writes the final splits, which is the last step that
changes the data before training. Every step that fits or scores a model gets
`depends_on = ["verify"]`. The gate writes nothing the models read, so `depends_on` is
what puts them behind it.

```toml
[[steps]]
name = "verify"
description = "Gate: no leakage between splits, no poisoned or invalid data"
function = "data_verify:verify"
inputs = { train = "build/train.csv", test = "build/test.csv", source = "build/clean.csv" }
outputs = { report = "build/verify/report.json", summary = "build/verify/report.md" }
params = { label = "species", source = "source" }
requires = ["data-verify"]

[[steps]]
name = "train"
depends_on = ["verify"]        # nothing trains on data the gate has not passed
# ...
```

`uv run data-verify template` prints a commented version of this block. `uv run data-verify
lint` checks its params without running anything, and fails if a step reads a split without
depending on the gate. The runner can't tell which step is a model, so nothing else
enforces this.

### Rules

- **Keep it critical** (the default). A non-critical gate only warns, and then the models
  run anyway.
- **Gate the data the model actually reads.** If a step after the split transforms the
  data (encoding, scaling, feature selection), point the gate at that step's output, or
  add a second gate there.
- **Declare every role your data has**: `label`, `id`, `group` (patient, user, device),
  `time`, `label_time`. The gate can only check what it knows. A grouped dataset without
  `group` passes, even when the same patient is in train and test. The input names say
  which file is which (`train`, `val`, `test`, and `source` for the data before the split).
- **Read project rules from the files the other steps use** (`{ from = "config", key =
  "..." }`), so that the gate checks the same contract and there is only one place to change
  it.
- **Project-specific rules** (label definitions, censoring, sample weights) go in the gate
  as `[[steps.params.assert]]` (pandas or SQL) or as a `custom` check module listed in
  `watch`, rather than in a separate validation step.

The full list of params and checks is in the
[gate's README](../packages/data-verify/README.md).

### When it blocks

The dump shows each blocking check and how to fix it. The report has the details:

```bash
uv run data-verify explain build/verify/report.json
```

1. Fix the step that made the data (split by group, cut by time, fit transforms on the
   training split only, drop the leaking column) and rerun. This is the right fix almost
   every time.
2. Rerun after each fix, because one problem can hide another.
3. Only when a finding is a real property of the domain, loosen the narrowest option with a
   comment that says why, for example `"duplicates.cross_split" = { max_fraction = 0.001 }
   # identical sensor readings across machines are normal`. Turning a check `"off"` is a
   last resort, and setting `critical = false` on the gate is never the fix.

Warnings (drift, near-duplicates, suspicious names, most poisoning signs) don't block, but
the status line names them (`warnings=1 warned=drift.features`), and the report explains
them. Read them before you trust a result. A project that would rather stop can raise them:
`checks = { "poisoning.*" = "error" }`, or `fail_on = "warn"` for every warning. Expect false
alarms on heavy-tailed data, and on large data with features that are roughly standard by
nature, where `preprocessing.global_scaling_near` often fires: turn that one off there.

---

## 8. Getting the most out of checkpoints

### What triggers a rerun

| You change… | Effect |
|---|---|
| the content of an input file | that step reruns (`inputs changed: <name>`) |
| `params`, `function`, input/output paths in the TOML | that step reruns (`config changed`) |
| the step's module file (even a comment) | every step in that module reruns (`code changed`) |
| a local module the step imports, directly or indirectly | that step reruns (`code changed: <file>`) |
| a file listed in `watch` (data, SQL, templates, dynamic imports) | that step reruns (`code changed`) |
| delete or edit an output | that step reruns (`output missing` / `output modified`) |
| upgrade a step package whose `__version__` changed (e.g. data-verify) | its steps rerun (`code changed`) |
| upgrade a package listed in the step's `requires` by its distribution name | that step reruns (`packages changed: duckdb 1.5.0 → 2.0.0`) |
| upgrade a package listed by its import name (`sklearn`, not `scikit-learn`) | nothing; no version is recorded, so list the distribution name |
| add or remove an entry in `requires` | that step reruns (`packages changed: joblib none → 1.5.0`) |
| change the version specifier of an entry | nothing, as long as the installed version satisfies it |
| the value of a variable listed in the step's `env` | that step reruns (`env changed: <name>`) |
| add or remove a name in `env` | that step reruns (`env changed: <name>`) |
| edit an installed step package without bumping its version | nothing; use `-r STEP` (§11) |
| the `description`, `critical` or `clean_outputs` flag | nothing; these don't affect results |
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

### Helper modules are tracked; use `watch` for everything else

When `steps/model.py` imports `steps/features.py`, editing `features.py` reruns the model
step, and the reason names the file: `code changed: steps/features.py`. The runner reads
the step module's `import` statements (without running them) and follows them through
other local modules, including imports inside functions, relative imports and package
`__init__.py` files. A module is local when it is found under `python_path`. The source of
the standard library and of installed packages is not followed: list a package in `requires`
to track its version.

Two things can't be seen this way, and need `watch`:

- modules loaded dynamically, such as `importlib.import_module(name)`;
- non-Python files the code reads, such as a lookup table, a SQL query or a prompt
  template.

```toml
watch = ["sql/features.sql"]
```

`watch` paths are relative to the working directory (the TOML file's folder by default), and
the files must exist.

Checkpoints written before imports, packages or `env` were tracked are filled in on the next
run instead of being invalidated, by a dry run too, so the first run after upgrading records
that shell's values. If you edited a helper that wasn't in `watch`, or upgraded a package,
since your last run, run once with `-f` (or `-r STEP`) after upgrading.

### Steps that read the outside world

A step that downloads data, queries a database, or reads the clock has no declared
input that changes. After one successful run it is **always cached**. To refresh it:

```bash
uv run data-pipeline -r fetch_prices          # rerun only that step (later steps follow if the data changed)
```

Or give it a parameter that you change on purpose, such as `params = { snapshot = "2026-10" }`.

Environment variables are ignored unless declared. List the ones that change a step's
results in `env`, such as `env = ["API_REGION"]`, and leave out those that only affect
resources, such as a thread count. Only an unsalted hash of the value is stored in the
checkpoint. That keeps it out of plain sight, but a short or guessable secret could still
be recovered from `state.json`, so don't commit that file.

### Resetting

- One step: `-r STEP` (repeatable).
- Everything: `-f`.
- Clear all saved state: delete `.pipeline/` (state lives in `.pipeline/state.json`).
  Deleting `build/` also forces reruns, because the outputs will be missing.
  Keep `state_dir` out of `build/`: the state also caches file hashes, and without it the
  next run reads every input again, which takes minutes for a dataset of tens of GB.

Checkpoints are content hashes, not absolute paths, so copying or moving a project
(state folder included) keeps everything cached.

---

## 9. Everyday commands

```bash
data-pipeline -n                    # what would run, and why? (runs nothing)
data-pipeline                       # run whatever is out of date
data-pipeline -v                    # same, and show each step's printed output
data-pipeline -F                    # same, and print each step's output live, as it runs
data-pipeline -r train              # rerun one step regardless of its checkpoint
data-pipeline -r train -r evaluate  # several
data-pipeline -o eda report         # run only these steps (their inputs must exist)
data-pipeline -f                    # rerun everything
data-verify explain build/verify/report.json   # why the gate blocked, with details
data-verify lint                    # check the gate's params, and that splits pass through it
```

(Run these from the project folder, prefixed with `uv run` if the virtual environment isn't
activated, and pass the TOML path if it isn't `./pipeline.toml`.)

A typical loop while developing one step:

1. Edit `steps/model.py`.
2. `data-pipeline -n` confirms that only `train` and `evaluate` would rerun.
3. `data-pipeline` reruns just those; everything before them is skipped.

`--only` doesn't run any step outside the list, and it won't block on them either. Use it
once the upstream outputs exist. When an upstream step left out isn't up to date, it prints
a warning, and the run ends with `completed with warnings`:

```
⚠ --only left out 'verify', which is not up to date (no checkpoint): 'train' depends on it
```

Take this one seriously when the step named is the gate: the data has not passed it. The
exit code stays `0`, so check for the warning if you script `--only`.

### Exit codes (for scripts and CI)

| Code | Meaning |
|---|---|
| `0` | success, **including** runs with non-critical warnings |
| `1` | a critical step failed (or preflight failed) |
| `2` | invalid config file or unknown step name in `-r`/`-o` |
| `130` | interrupted (Ctrl-C, SIGTERM or SIGHUP) |

In CI, `data-pipeline || exit 1` fails the build only on critical problems. If warnings
should fail the build too, check the output for `with warnings`. A real run ends with
`completed with warnings`, and a dry run with `dry run complete with warnings`.

---

## 10. Recipes

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
later steps too. If region names can disappear, set `clean_outputs = true` on the step:
otherwise old files remain in the directory, become part of its hash, and reach every
step that reads it, the gate and the model included.

```toml
clean_outputs = true                      # the runner empties build/regions before a rerun
```

The runner deletes the declared outputs just before the step runs, never when it is cached
or in a dry run. The step then starts from nothing, so keep the `mkdir`. The trade-off: when
the rerun fails, the previous result is gone too, which is the point (it no longer matches
the inputs), but copy it aside first if you want to compare. The loader refuses an output
that would delete more than outputs: one outside the workdir, or one holding the state
folder, the pipeline file, an import root or a `watch` file.

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

For data that feeds a model, use the gate (§7) and put project rules in its `assert`
or `custom` params. For other checks (raw files, database loads, exports) a small step of
your own works: a step with no outputs is fine for checks: it runs, and while its inputs stay the same it's
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
training jobs. When it's exceeded, the worker and any processes it started are killed and the
step is reported as `timed out after Ns`. This counts as a failure, so it halts the run or warns depending on
`critical`. Whatever the step printed before it was killed is kept in the dump and the log.

To watch a long step, run with `-F`, or follow its log from another terminal:

```bash
tail -f .pipeline/logs/train.log
```

If you press Ctrl-C, the step gets two seconds to stop, then it and any processes it started
are killed. Its log is kept, and the run ends with `■ interrupted · log: <path>`. The log
shows the step's last output, and usually the traceback of where it was when you stopped it.
SIGTERM and SIGHUP (a cancelled CI job, a closed terminal) do the same, unless they were
already being ignored, so `nohup data-pipeline` keeps running after you log out.

Each worker runs in its own process group. A step that means to leave a process running
after it succeeds still can, but a process started by a step that times out or is interrupted
does not survive, unless it moved itself to a new session. If the runner itself dies, for
example from `kill -9`, the worker normally notices within a second and stops the same way, so
nothing goes on writing outputs that no checkpoint will record. A step inside one long call
that holds Python's lock, such as a big `sum()`, can delay that by as long as the call takes.

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
options. `result.warnings` lists what the run warned about, such as a step `only=` left out
that isn't up to date, and `result.ok` stays true, as the exit code does on the command line.

Only the command line turns SIGTERM and SIGHUP into an interrupt. In your own program they do
whatever you've set up. If they end the process, the worker stops itself as described above,
without the two seconds for a traceback.

---

## 11. Building on top: reusable step packages

data-verify is the model to follow. When a step is useful in more than one project, such as
a loader for an in-house format, a standard report or a domain check, turn it into a
package instead of copying `steps/*.py` between projects.

### Where it lives

- **General-purpose** (useful to any data project, like the gate): add it to this
  repository as a workspace member under `packages/<name>/`, with its dependencies in its
  own `pyproject.toml`. Add an extra in the root `pyproject.toml`
  (`<name> = ["<name>"]`, plus `<name> = { workspace = true }` in `[tool.uv.sources]`) so
  projects can install it with `"../data-pipeline[<name>]"`.
- **Specific to a team or domain**: its own uv project and repository, which depends on
  `data-pipeline` the same way a project does (§2.1). Projects then `uv add` it next to the
  runner.
- **Specific to one project**: leave it in that project's `steps/`. A step only becomes a
  package once a second project needs it.

### The contract a step package keeps

1. **Export each step from the package's top-level `__init__.py`** and reference it as
   `function = "package:step"`. The runner finds installed modules without importing them,
   and only at the top level: `package.sub:step` fails with `module not found`.
2. **Keep `__version__` in that `__init__.py` and bump it on every behaviour change.**
   For an installed package the runner fingerprints only the file that holds the step
   function, which here is `__init__.py`. Bumping the version is what makes every project
   rerun the step after an upgrade. With an editable install, use `-r STEP` after editing
   the package without a bump.
3. **Everything goes through `ctx`.** Data comes from `ctx.inputs` and settings from
   `ctx.params`, output goes to `ctx.outputs`, and the package reads no global config or
   environment. Validate `params` up front, reject unknown keys, and name the bad key.
4. **Write deterministic outputs** (no timestamps, sorted keys, rounded floats), so that
   rerunning the step keeps later steps cached.
5. **Fail by raising an exception** that names the problem and its fix. Delete stale
   outputs before running, so that a failed run never leaves an old result looking current.
   A package does this itself, as the gate does: a project can't be relied on to set
   `clean_outputs = true` on every step that uses it.
6. **Document the contract in the package README**: the input names it understands, the
   outputs it writes, every param, and what makes it fail. A project must be able to use
   the step from the README alone.
7. **Test by injecting the fault the step guards against** into clean data, and check that
   it fires while the clean data still passes. Include one end-to-end test that runs the
   step through `data_pipeline.cli` in a temporary project (see
   `packages/data-verify/tests/test_step.py`).
8. **Keep dependencies in the package.** Never add them to the runner.

### Changing the runner instead

Change the runner only for something a step cannot do: ordering, caching, process
isolation or reporting. Don't break the public interface (`StepContext`, `Orchestrator.run`,
the CLI flags, the TOML keys), add no dependencies, and update the README and this guide in
the same commit. If no step needs a feature yet, leave it out.

---

## 12. Reading the output

```
▶ iris · 7 steps · examples/iris/pipeline.toml
[1/7] clean     ↷ cached            rows=149 dropped=1 classes=3
[3/7] verify    ✓ ran        0.51s  checks=29 passed=28 warnings=1 skipped=26 warned=drift.features  (inputs changed: train)
[4/7] eda       ⚠ warning    0.90s  exit code 1  [non-critical, continuing]
┌─ eda: exit code 1 ─ analysis:explore
  …full captured output and traceback…
log: .pipeline/logs/eda.log
└────────────────────────────────────────────────────────────
[5/7] train     ✓ ran        1.10s  estimator=random_forest train_acc=1  (config changed)
[7/7] report    – skipped           upstream step(s) did not complete: eda  [non-critical, continuing]
✓ completed with warnings · 3 ran · 2 cached · 1 warning(s) · 1 skipped · 3.59s
```

| Symbol | Meaning |
|---|---|
| `✓ ran` | executed successfully; the part in parentheses says why it had to run |
| `↷ cached` | skipped; the summary shown is from the last successful run |
| `• will run` | dry run only; `after X` means it waits on a step that will also run |
| `⚠ warning` | a non-critical step failed; the error is printed right below it |
| `– skipped` | a non-critical step couldn't run because something it depends on failed |
| `✗ FAILED` | a critical step failed; the error is printed and the run stops |

The last line of a failure dump is the path of the step's log file. With `-F`, the output
is printed above as it arrives, so the dump shows `(output above)` instead of repeating it.

In a terminal, a dim `… running` placeholder appears while a step executes and is replaced
by its final line. Colors are disabled with `--no-color` or the `NO_COLOR` environment
variable, and automatically when output is redirected to a file.

---

## 13. Troubleshooting

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
| a step keeps rerunning even though nothing changed | an earlier step writes non-deterministic output | check `-n` to see which input changed; see §8 *Make outputs deterministic* |
| a step **doesn't** rerun after you edited a helper | the helper is loaded dynamically, isn't under `python_path`, or isn't Python | add it to `watch` |
| a step doesn't rerun after the source data changed | it reads external data with no declared input | use `-r STEP`, or declare the source file as an input |
| `package not installed: data-verify` | the project was set up without the `verify` extra | `uv add --editable "../data-pipeline[verify]"` |
| `module not found: package.sub` for an installed package | installed step modules are found only at the top level | export the step from the package's `__init__.py` and use `package:step` (§11) |
| `VerificationFailed: data-verify blocked the pipeline` | the gate found leakage or poisoning | read the fixes in the dump or `data-verify explain`; fix the data step, not the gate (§7) |
| `SpecError` in the `verify` step | a gate param is unknown, mistyped, or names a missing column | the message names the key and suggests the closest one; `data-verify lint` checks without running |
| the gate passes on data you know is grouped or time-ordered | `group` / `time` / `label_time` not declared | declare every role the data has (§7) |
| the gate doesn't rerun after you edited data-verify | installed packages are fingerprinted by `__init__.py` only | bump `__version__`, or `-r verify` |

When you're not sure why a step behaves the way it does, start with `-n`. The reason column
tells you what the orchestrator thinks changed.

---

## 14. Checklist for a new pipeline

- [ ] The project has its own `pyproject.toml` and depends on `data-pipeline[verify]` (no
      copied runner code); its step packages are added with `uv add`.
- [ ] Raw data in `data/`, step code in `steps/`, generated files in `build/`.
- [ ] `build/` and `.pipeline/` are in `.gitignore`.
- [ ] Every file a step reads is listed in its `inputs`; every file it writes is in `outputs`.
- [ ] Every step lists its third-party packages in `requires`, by distribution name
      (`scikit-learn`, not `sklearn`), so that upgrades rerun it. Changing a name reruns the
      step once.
- [ ] Environment variables that change a step's results are listed in `env`.
- [ ] All randomness is seeded through `params`.
- [ ] Dynamic imports and non-Python helpers (SQL, templates, lookup tables) are listed in `watch`.
- [ ] Optional steps (plots, reports, experiments) are `critical = false`, and no critical
      step depends on them.
- [ ] Long or network-bound steps have a `timeout`.
- [ ] If a model is trained: a critical `verify` gate reads the final splits, declares every
      role (`label`, `id`, `group`, `time`, `label_time`), and every training and evaluation
      step has `depends_on = ["verify"]`.
- [ ] Any check turned down in the gate has a comment saying why.
- [ ] A step used by a second project has moved into a package (§11).
- [ ] `params` values are strings, numbers, booleans, arrays or tables (no TOML dates).
- [ ] `data-pipeline -n` shows the order and reasons you expect.
- [ ] A second `data-pipeline` run shows every step `↷ cached`.
