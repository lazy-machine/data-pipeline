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

Successful steps get one line each. When a step fails, its captured stdout/stderr and the
full traceback are dumped to stdout. Steps marked `critical = false` only warn on failure
and the pipeline keeps going (their dependents get skipped). A critical failure halts it.

```
▶ iris · 6 steps · examples/iris/pipeline.toml
[1/6] clean     ↷ cached            rows=149 dropped=1 classes=3
[2/6] split     ↷ cached            train=111 test=38
[3/6] eda       ⚠ warning    0.90s  exit code 1  [non-critical, continuing]
┌─ eda: exit code 1 ─ analysis:explore
Traceback (most recent call last):
  ...
AttributeError: 'DataFrame' object has no attribute 'nope'
└────────────────────────────────────────────────────────────
[4/6] train     ✓ ran        1.10s  estimator=random_forest train_acc=1  (config changed)
[5/6] evaluate  ✓ ran        1.03s  accuracy=0.9474 f1_macro=0.9466  (inputs changed: model)
[6/6] report    – skipped           upstream step(s) did not complete: eda  [non-critical, continuing]
✓ completed with warnings · 2 ran · 2 cached · 1 warning(s) · 1 skipped · 3.08s
```

For a walkthrough, best practices and troubleshooting, see the [usage guide](docs/GUIDE.md).

## Setup

```bash
uv sync                                              # core + dev + example dependencies
uv run data-pipeline examples/iris/pipeline.toml     # run the example
uv run pytest                                        # run the tests
```

The core needs only the standard library plus `packaging`. The `examples` dependency group
(pandas, scikit-learn, matplotlib) is only needed for the Iris example.

## Usage

```
data-pipeline [CONFIG] [-n] [-f] [-r STEP]... [-o STEP ...] [-v] [--no-color]
```

| option | effect |
|---|---|
| `CONFIG` | pipeline file, default `./pipeline.toml` |
| `-n`, `--dry-run` | verify and show what would run and why, without running anything |
| `-f`, `--force` | ignore all checkpoints |
| `-r`, `--rerun STEP` | ignore the checkpoint of one step (repeatable) |
| `-o`, `--only STEP ...` | run only these steps; their inputs must already exist |
| `-v`, `--verbose` | also show the captured output of successful steps |

Exit codes: `0` success (warnings included), `1` a critical step failed, `2` bad config or
arguments, `130` interrupted.

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
timeout = 600                                 # seconds; the worker is killed after this
depends_on = []                               # extra ordering beyond file dependencies
watch = ["steps/helpers.py"]                  # extra files whose changes invalidate the step
description = "Fit a classifier"
```

Step order follows the file, except that a step always runs after the steps that produce
its inputs (dependencies are inferred from paths) and after anything in `depends_on`.
Cycles, duplicate names, two steps writing the same output and unknown keys are rejected
when the file is loaded.

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
Anything printed is captured and only shown when the step fails, or with `-v`. Raising any
exception fails the step. The step must write every declared output.

## Checkpoints

A step's fingerprint combines:

- **code**: the SHA-256 of the step's module file plus any `watch` files,
- **config**: `function`, `params`, `inputs` and `outputs`,
- **inputs**: the SHA-256 of every input's content (directories are hashed recursively).

After a successful run, the fingerprint and the output hashes go into
`<state_dir>/state.json`. The next run skips the step only if the fingerprint matches and
every output still has the recorded hash. Hashes are content-based, so when an upstream
step re-runs and writes byte-identical outputs, the steps after it stay cached. File hashes
are cached by size and mtime, so large unchanged files are not re-read.

The status line says why a step ran: `no checkpoint`, `code changed`, `config changed`,
`inputs changed: <names>`, `output missing: <name>`, `output modified: <name>`, `forced`.

Code is fingerprinted per module file. If a step imports a local helper module, list that
helper in `watch` so editing it invalidates the step.

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
examples/iris/     end-to-end example: clean, split, EDA, train, evaluate, report
tests/             pytest suite
```
