# data-pipeline

A general, TOML-driven pipeline runner for data analysis and machine learning.

A thin orchestrator walks the steps one at a time. For every step it:

1. **Verifies** the step: its module exists, required packages are installed (with
   version specifiers), its input files exist, and its upstream steps completed.
2. **Checks the checkpoint**: if code, config and input contents are unchanged and every
   output still exists untouched, the step is skipped (`↷ cached`).
3. Otherwise **spawns a worker subprocess** that imports and runs the step function in a
   fresh interpreter, so steps share no memory, globals or loaded modules.
4. **Verifies and records** the outputs, then prints a one-line status.

Before any model runs, a **data gate** checks the splits for leakage and poisoning and
halts the pipeline if it finds any. The gate is an ordinary step,
[data-verify](packages/data-verify/README.md) (`function = "data_verify:verify"`), shipped in
this repository as a second package so that the runner itself stays dependency-free.

Successful steps get one line each. When a step fails, its captured stdout/stderr and the
full traceback are dumped to stdout. Steps marked `critical = false` only warn on failure
and the pipeline keeps going (their dependents get skipped). A critical failure halts it.

```
▶ iris · 7 steps · examples/iris/pipeline.toml
[1/7] clean     ↷ cached            rows=149 dropped=1 classes=3
[2/7] split     ↷ cached            train=111 test=38
[3/7] verify    ↷ cached            checks=27 passed=26 warnings=1 skipped=24
[4/7] eda       ⚠ warning    0.90s  exit code 1  [non-critical, continuing]
┌─ eda: exit code 1 ─ analysis:explore
Traceback (most recent call last):
  ...
AttributeError: 'DataFrame' object has no attribute 'nope'
└────────────────────────────────────────────────────────────
[5/7] train     ✓ ran        1.10s  estimator=random_forest train_acc=1  (config changed)
[6/7] evaluate  ✓ ran        1.03s  accuracy=0.9474 f1_macro=0.9466  (inputs changed: model)
[7/7] report    – skipped           upstream step(s) did not complete: eda  [non-critical, continuing]
✓ completed with warnings · 2 ran · 3 cached · 1 warning(s) · 1 skipped · 3.08s
```

For a walkthrough, best practices and troubleshooting, see the [usage guide](docs/GUIDE.md).

## Setup

```bash
uv sync                                              # runner, gate, dev and example dependencies
uv run data-pipeline examples/iris/pipeline.toml     # run the example
uv run pytest                                        # run the tests of both packages
```

The repository is a uv workspace with two packages:

| package | path | depends on |
|---|---|---|
| `data-pipeline` (the runner) | `src/data_pipeline/` | the standard library and `packaging` |
| `data-verify` (the gate) | `packages/data-verify/` | pandas, numpy, pyarrow; duckdb for SQL assertions |

The `examples` dependency group (pandas, scikit-learn, matplotlib, data-verify) is only
needed for the Iris example.

## Use it in a project

Keep your project in its own folder with its own `pyproject.toml`, and depend on this one:

```bash
uv init --bare my-analysis && cd my-analysis
uv add --editable "../data-pipeline[verify]"   # runner + gate; drop [verify] for the runner alone
uv add pandas scikit-learn                     # whatever your steps import
uv run data-pipeline -n                        # with a pipeline.toml in place
```

To pin a version instead of following a checkout, use
`uv add "data-pipeline[verify] @ git+file:///abs/path/to/data-pipeline@<commit>"`. The
[usage guide](docs/GUIDE.md) §2 covers project layout and starting from the Iris example.
§7 covers where the gate goes.

## Usage

```
data-pipeline [CONFIG] [-n] [-f] [-r STEP]... [-o STEP ...] [-v] [-F] [--no-color]
```

| option | effect |
|---|---|
| `CONFIG` | pipeline file, default `./pipeline.toml` |
| `-n`, `--dry-run` | verify and show what would run and why, without running anything |
| `-f`, `--force` | ignore all checkpoints |
| `-r`, `--rerun STEP` | ignore the checkpoint of one step (repeatable) |
| `-o`, `--only STEP ...` | run only these steps; their inputs must already exist. Warns when an upstream step left out is not up to date |
| `-v`, `--verbose` | also show the captured output of successful steps |
| `-F`, `--follow` | print each step's output live, as it runs |

Exit codes: `0` success (warnings included), `1` a critical step failed, `2` bad config or
arguments, `130` interrupted (Ctrl-C, SIGTERM or SIGHUP).

## Pipeline file

```toml
[pipeline]
name = "iris"                 # default: the file name
python_path = ["steps"]       # import roots for step modules, relative to this file (default ".")
workdir = "."                 # base for every input/output path (default: this file's directory)
state_dir = ".pipeline"       # where checkpoints live, relative to workdir

[[steps]]
name = "train"
function = "model:train"                      # module:function, found on python_path
inputs = { train = "build/train.csv" }        # name = path
outputs = { model = "build/model.pkl" }       # files or directories
params = { estimator = "random_forest", seed = 42 }
requires = ["scikit-learn>=1.3", "joblib"]    # verified before anything runs
critical = true                               # false: failure only warns
timeout = 600                                 # seconds; the worker and its children are killed after this
depends_on = []                               # extra ordering beyond file dependencies
watch = ["sql/features.sql"]                  # non-Python files or dynamic imports that invalidate the step
env = ["API_REGION"]                          # environment variables that change the step's results
description = "Fit a classifier"
```

Step order follows the file, except that a step always runs after the steps that produce
its inputs and after anything in `depends_on`. Dependencies are inferred from paths: an
input depends on a step when the paths match or one lies inside the other, as with a file
in another step's directory output. Cycles, duplicate names, two steps writing the same or
overlapping outputs, a step reading its own output and unknown keys are rejected when the
file is loaded.

## Writing a step

A step is a plain function that takes a `StepContext`:

```python
def train(ctx):
    df = pd.read_csv(ctx.inputs["train"])          # inputs/outputs are absolute Paths
    model = fit(df, **ctx.params)
    joblib.dump(model, ctx.outputs["model"])       # output parent dirs already exist
    return {"train_acc": model.score(...)}         # shown on the status line
```

`ctx` has `name`, `inputs`, `outputs`, `params`, `workdir` and a `log()` helper. The return
value can be `None`, a string, or a flat dict (floats are shown to 4 significant figures).
Anything printed is written to `<state_dir>/logs/<step>.log` as it arrives, and shown on
the console when the step fails, with `-v`, or live with `-F`. Raising any
exception fails the step. The step must write every declared output.

## Checkpoints

A step's fingerprint combines:

- **code**: the SHA-256 of the step's module file, the local modules it imports (directly
  or through other local modules), and any `watch` files,
- **config**: `function`, `params`, `inputs` and `outputs`,
- **inputs**: the SHA-256 of every input's content (directories are hashed recursively),
- **packages**: the installed version of everything in `requires`,
- **env**: a SHA-256 of each variable listed in `env`, never its value.

After a successful run, the fingerprint and the output hashes go into
`<state_dir>/state.json`. The next run skips the step only if the fingerprint matches and
every output still has the recorded hash. Hashes are content-based, so when an upstream
step re-runs and writes byte-identical outputs, the steps after it stay cached. File hashes
are cached by size and mtime, so large unchanged files are not re-read.

The status line says why a step ran: `no checkpoint`, `code changed`, `config changed`,
`code changed: <files>`, `packages changed: <name> <old> → <new>`, `env changed: <names>`,
`inputs changed: <names>`, `output missing: <name>`, `output modified: <name>`, `forced`.

Imports are found by reading the code, never by running it. A module is local when it is
found under `python_path`; the standard library and installed packages are not tracked.
Modules loaded with `importlib.import_module` and non-Python files belong in `watch`.

After upgrading from a version that didn't track imports, run with `-f` once if you edited
a helper without `watch` since your last run: older checkpoints are filled in rather than
invalidated, so that upgrading doesn't rerun everything. Package versions and `env` work the
same way: they are recorded on the first run after upgrading, without rerunning the step.

## Layout

```
src/data_pipeline/
  config.py        TOML loading, validation, dependency graph, ordering
  checkpoint.py    content hashing, fingerprints, persistent state
  orchestrator.py  the sequential verify → checkpoint → spawn → record loop
  worker.py        subprocess entry point that runs a single step function
  context.py       StepContext passed to step functions
  report.py        status lines and failure dumps
  cli.py           command line interface
tests/             runner tests
packages/data-verify/
  src/data_verify/ the gate step and its checks (README.md is its contract)
  tests/           gate tests
examples/iris/     end-to-end example: clean, split, verify, EDA, train, evaluate, report
docs/GUIDE.md      usage guide
AGENTS.md          rules for agents working on this repository or building on it
```
