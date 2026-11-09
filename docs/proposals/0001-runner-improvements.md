# Proposal 0001: runner improvements from the hard-drive-analysis-v3 build

- **Status:** Part A: A1, A2 and A3 done. Part B: decided (2026-11-09): B3 approved and
  built; B1, B2 and B4 not approved
- **Source:** lessons from building `../hard-drive-analysis-v3` on this runner (12 steps,
  a 22 GB input, training steps that run for hours). Every claim below was checked
  against the code at commit `8f8dcaf`.
- **Audience:** the agents that work on this repository. Part A is ready to implement.
  Part B needs the owner's decision first: **do not implement anything in Part B** until
  this file says it was approved.

## How to work on this proposal

1. Implement **one task at a time**, in the order A1, A2, A3. Each task is one commit on
   its own branch, named `feat/<task-id>-<short-name>` (for example `feat/a1-live-logs`).
2. Before you commit, `uv run pytest` must pass: all existing tests plus the new ones
   listed in the task. Do not delete or weaken an existing test. If one has to change,
   explain why in the commit message.
3. Do not add runtime dependencies. The runner depends only on `packaging` and the
   standard library, and must stay that way.
4. Do not break the public interface: `StepContext` fields, `Orchestrator(pipeline,
   reporter).run(only=, force=, rerun=, dry_run=)`, the existing CLI flags and the TOML
   keys. You may **add** keyword arguments, flags and keys.
5. Match the existing style: `from __future__ import annotations`, type hints, short
   docstrings that say *why*, and no comments that restate the code.
6. Update `docs/GUIDE.md` and `README.md` in the same commit whenever behaviour changes.
   Each task lists the sections to touch.
7. When a task is done, change its status line in this file from `todo` to `done
   (<commit>)`.

---

## Part A: ready to implement

### A1. Live step logs and `--follow`

- **Status:** done (6f4a478)
- **Priority:** highest. This is the main day-to-day problem.

**Problem.** `orchestrator._run_worker` calls `subprocess.run(..., stdout=PIPE)`, so all of
a step's output stays in memory until the worker exits.

- A step that runs for 20 minutes, or for hours, shows nothing while it runs.
- `ctx.log` prints to stdout, so it is captured in the same way.
- If the user presses Ctrl-C on a step that seems stuck, `subprocess.run` kills the
  worker and everything the step printed is lost. That output is exactly what is needed
  to see why it was stuck.
- hda-v3 had to write its own progress logger (`steps/common.py: Progress`, which writes
  to `build/logs/<step>.log`) and follow that with `tail -f`. Every project will end up
  writing the same helper.

**Behaviour to implement.**

1. Every time a step runs, its combined stdout/stderr is also written to
   `<state_dir>/logs/<step name>.log`. The file is overwritten when the step starts and
   written to as each line arrives (flush after every line). Steps that are cached do
   not touch their log. The log is **not** part of any fingerprint.
2. `Outcome.output` keeps the full captured text, so failure dumps and `-v` work as
   before.
3. When a step **times out**, the output it printed before it was killed is kept in the
   log and in `Outcome.output`.
4. On **Ctrl-C**, kill the worker, wait for it to exit, and make sure the log file holds
   everything received so far. Then let `KeyboardInterrupt` propagate. The CLI already
   prints `■ interrupted`; make it print `■ interrupted · log: <path>` when a step was
   running.
5. The failure dump (`Reporter._dump`) ends with a line `log: <path relative to cwd>`.
6. New CLI flag `-F/--follow`: print each output line as it arrives, indented with the
   same `    │ ` prefix that `-v` uses.
   - In follow mode, do **not** print the output again when the step ends: no `-v`
     dump, and the failure box shows only its header, `(output above)`, the log path and
   its footer.
   - When stdout is a TTY, `Reporter.begin` writes a `… running` line without a newline.
     Before the first streamed line, clear it (`\r\033[K`), print the step's prefix and
     name once, and then the streamed lines.
   - Add `follow: bool = False` to `Reporter.__init__`. The orchestrator decides whether
     to stream by reading `reporter.follow`.

**How to implement.** Only `orchestrator.py`, `report.py` and `cli.py` change.

```python
proc = subprocess.Popen(cmd, cwd=..., env=env, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
chunks: list[bytes] = []

def pump() -> None:                      # runs in a background thread
    with log_path.open("wb") as fh:
        for raw in proc.stdout:          # one line at a time, as bytes
            fh.write(raw)
            fh.flush()
            chunks.append(raw)
            if on_line:                  # set only in follow mode
                on_line(raw.decode(errors="replace").rstrip("\n"))

reader = threading.Thread(target=pump, daemon=True)
reader.start()
try:
    proc.wait(timeout=step.timeout)
except subprocess.TimeoutExpired:
    proc.kill(); proc.wait(); reader.join(timeout=5)
    return False, f"timed out after {step.timeout:g}s", b"".join(chunks).decode(errors="replace")
except KeyboardInterrupt:
    proc.kill(); proc.wait(); reader.join(timeout=5)
    raise
reader.join(timeout=5)   # a grandchild that inherited stdout could keep the pipe open
```

- Read `chunks` only after `reader.join`.
- The 5-second join timeout is deliberate. A step can start a background process that
  keeps the pipe open after the worker has exited, and the runner must not hang on it.
- Keep `PYTHONUNBUFFERED=1` in the worker's environment: it is what makes lines arrive
  live.
- The log path comes from the orchestrator and is
  `pipeline.state_dir / "logs" / f"{step.name}.log"`. Create the directory first.

**Tests to add** (`tests/test_pipeline.py`, using the existing `make` helper; add step
functions to `STEPS` as needed):

- `test_step_log_is_written`: a step that prints `hello` and succeeds →
  `.pipeline/logs/<name>.log` contains `hello`. Run it again: the step is cached and the
  log is unchanged (compare the modification time).
- `test_failed_step_log_and_dump_path`: `fns:boom` → its log contains `about to fail` and
  `kaboom`, and the printed output contains `log: `.
- `test_timeout_keeps_partial_output`: a step that prints `started` (flush), then sleeps
  10 s with `timeout = 0.5` → the outcome detail contains `timed out`, and both the log
  and `Outcome.output` contain `started`.
- `test_follow_streams_before_step_ends`: the same step with `timeout = 1`, run through
  `Reporter(stream=io.StringIO(), follow=True)` → in the printed text, `started` comes
  before `timed out`, and `started` appears exactly once. (Without follow, the dump comes
  after the status line, so the order is reversed. That is what proves the streaming.)
- Existing `test_missing_output_timeout_and_hard_exit_fail` must still pass unchanged.

**Docs.**

- `GUIDE.md` §3 "About `print`": output is now streamed to the log file, and can be
  followed with `-F`.
- §7 "Everyday commands": add `-F`.
- §8 "Long-running steps": `tail -f .pipeline/logs/<step>.log`.
- §9: the `log:` line in the dump.
- `README.md` "Usage": add `-F` to the options table.

---

### A2. Fingerprint the local modules a step imports

- **Status:** done (6f4a478)
- **Priority:** high. This is the only correctness risk found.

**Problem.** `checkpoint.fingerprint` hashes the module that holds the step function,
plus the files listed in `watch`, and nothing else. A step that imports a helper module
from the same project does not rerun when the helper changes, unless the helper is listed
in `watch` by hand. Every step in hda-v3's `pipeline.toml` carries a `watch` list for
this reason. One forgotten entry gives stale results that look up to date, which is the
worst failure a caching runner can have. `GUIDE.md` §6 documents the limitation, but
documentation does not prevent the mistake.

**Behaviour to implement.**

1. When it fingerprints a step, the runner finds every **local** module the step module
   imports, directly or through other local modules, and hashes them.
   - A module is local when it resolves to a file under one of `pipeline.python_path`.
     Never use `importlib.util.find_spec` here: installed packages and the standard
     library are not local.
2. Imports are found statically with `ast`, never by importing (the orchestrator must
   never import step code).
   - Count `import x.y` and `from x.y import z`, wherever they appear, including inside
     functions (`ast.walk`).
   - For `import a.b.c`, the candidates are `a`, `a.b` and `a.b.c`.
   - For `from a.b import z`, the candidates are `a`, `a.b` and `a.b.z` (`z` may be a
     submodule).
   - Relative imports (`from . import z`, `from .m import z`, `level >= 1`) resolve
     against the importing module's own package. Work out its dotted name from its path
     relative to the python_path root it was found under. For `pkg/mod.py` and
     `pkg/__init__.py` alike the package is `pkg`; each extra level removes one more
     part.
   - Every candidate that resolves to a local file is included, including package
     `__init__.py` files, because Python runs them on import.
   - Dynamic imports (`importlib.import_module(name)`) cannot be found. `watch` is still
     the way to cover them, and also non-Python files such as SQL or lookup tables.
3. Follow imports recursively, keeping a visited set (import cycles are legal Python).
4. If a file cannot be parsed (`SyntaxError`, `UnicodeDecodeError`, `ValueError`), still
   hash the file but skip its imports. The worker will report the real error.
5. Store the result in the checkpoint as a **new key**, `"imports": {relative posix
   path: file hash}`, sorted by path and relative to the workdir. Do **not** add it to
   the `code` hash. That keeps existing checkpoints valid (see the migration rule below).
6. In `stale_reason`, after the `code` check, compare `imports` and report the changed
   files: `code changed: steps/common.py` (added, removed or modified, at most three
   names, then `, …`).
7. **Migration rule.** If a recorded checkpoint has no `imports` key (it was written by an
   older runner), add the current value and treat it as unchanged: no rerun. Save the
   state. Otherwise every user would face a full rerun after upgrading, which can mean
   hours. Document this in the README: "after upgrading, run with `-f` once if you edited
   a helper without `watch` since your last run".

**Where.** `checkpoint.py`: add `local_imports(pipeline, src: Path) -> list[Path]` and a
helper that finds a module file under python_path only. `module_file` already searches
the roots; factor that loop out rather than copying it. Then extend `Fingerprint` and
`State.record`, and update `stale_reason`. `orchestrator.py` should not need changes
beyond passing the new field through.

**Tests to add** (a new file `tests/test_imports.py` is fine, reusing the `make` and `run`
helpers from `test_pipeline.py`, or moving them to `tests/conftest.py`):

- `test_helper_edit_reruns_step_without_watch`: `steps/fns2.py` does `import helper`.
  Edit `steps/helper.py` → the step reruns, with a reason containing `helper.py`.
- `test_transitive_helper`: a imports b imports c. Editing `c.py` reruns the step.
- `test_from_import_of_submodule_and_relative_import`: package `steps/pkg/` with
  `__init__.py`, `util.py` and `mod.py` doing `from . import util`. Editing `util.py`
  reruns a step whose function lives in `pkg.mod`.
- `test_import_inside_function_counts`.
- `test_unrelated_local_file_keeps_cache`: editing a local module that is not imported
  leaves the step cached.
- `test_installed_and_stdlib_imports_ignored`: a step importing `json` and `packaging`
  has an empty `imports` entry.
- `test_old_checkpoint_without_imports_is_backfilled`: run, delete the `imports` key from
  `state.json`, run again → cached, and `state.json` has `imports` again.
- `test_syntax_error_in_helper_does_not_crash_fingerprinting`: the worker fails and
  reports the error, and the runner does not raise.

**Docs.**

- `GUIDE.md` §6: rewrite "Track helper modules with `watch`". Local imports are tracked
  automatically. `watch` is for data files, SQL, templates and dynamic imports.
- Update the table row about `watch` and the §10 troubleshooting row "a step doesn't
  rerun after you edited a helper".
- The §11 checklist item becomes "dynamic imports and non-Python helpers are listed in
  `watch`".
- `README.md` "Checkpoints" and the `watch` comment in "Pipeline file".

---

### A3. Package versions and declared environment variables in the fingerprint

- **Status:** done (#5)
- **Priority:** medium.

**Problem.** `requires` is checked (installed, and the version satisfies the specifier),
but the installed version is not part of the checkpoint. While hda-v3 was being built,
upgrading DuckDB 1.x → 2.0 changed behaviour, and no checkpoint noticed it. Environment
variables are ignored completely. That is the right default (hda-v3 uses `HDA_THREADS`
for resources only), but a step cannot declare one that does change its results.

**Behaviour to implement.**

1. New checkpoint key `"packages": {distribution name: installed version}`, one entry
   for each `requires` entry of the step.
   - Use `importlib.metadata.version(Requirement(spec).name)`.
   - For a name `_check_requirement` accepts as an import name (for example `sklearn`),
     record `"import-only"`.
2. New optional step key `env = ["NAME", ...]`, a list of strings. Add it to
   `_STEP_KEYS` and validate it like `watch`.
   - New checkpoint key `"env": {NAME: sha256 hex of the value, or "unset"}`.
   - **Store hashes, never raw values**: environment variables often hold secrets, and
     `state.json` is a plain file.
3. In `stale_reason`, after `imports`:
   - `packages changed: duckdb 1.5.0 → 2.0.0` (at most three entries, then `, …`);
   - `env changed: NAME`.
4. The same migration rule as A2: a checkpoint without one of these keys is backfilled,
   not rerun.
5. `-n` shows these reasons like any other.

**Where.** `checkpoint.py` (fingerprint, record, stale_reason) and `config.py` (the `env`
key). Put the version lookup in a small function, `checkpoint.installed_version(name)`,
so tests can monkeypatch it. Fingerprinting runs in the orchestrator process, not in the
worker.

**Tests to add.**

- `test_package_version_change_reruns`: monkeypatch `installed_version` → reason
  `packages changed: …`.
- `test_env_change_reruns_and_value_not_stored`: `monkeypatch.setenv("DP_TEST_SECRET",
  "s3cr3t")`, run, change the value, run → reason `env changed: DP_TEST_SECRET`. The text
  `s3cr3t` does not appear in `state.json`.
- `test_undeclared_env_is_ignored`: changing a variable that is not listed keeps the step
  cached.
- `test_old_checkpoint_without_packages_or_env_is_backfilled`.
- Config validation: `env = "X"` (not a list) gives `ConfigError`. Add it to the existing
  parametrised `test_config_validation`.

**Docs.**

- `GUIDE.md` §6 table: two new rows (package upgrade, declared env var).
- §6 "Steps that read the outside world": mention `env`.
- `README.md` "Pipeline file" (the `env` key) and "Checkpoints".

---

## Part B: needs the owner's decision first

These came out of the same build. Each is a design change, so it waits for a "yes"
written into this file. The notes are there to make that decision easier, not to be
implemented.

**Decision (2026-11-09).** The owner asked for three independent evaluations (Claude Opus
5.5, DeepSeek V4.1 Flash 1017, GLM 5.3) and approved only what all three called an
improvement. Do not reopen B1, B2 or B4 without a new case that answers the objections
below.

### B1. Step templates (`extends`)

- **Decision:** not approved (two of three against). `data-verify lint` reads the raw TOML
  (`packages/data-verify/src/data_verify/cli.py`, `_ungated_steps`), so `inputs`,
  `depends_on` or `function` inherited from a template would be invisible to it, and an
  ungated model step would pass lint. It also reverses the documented choice in
  `GUIDE.md` "Shared settings", for a gain that a shared config input already gives.

hda-v3's three `train_*` steps repeat the same parameter block of about 200 characters.
One of them differs in a single value (`feature_fraction = 1.0`), which is easy to miss
in review. `GUIDE.md` §8 "Shared settings" says there is no templating **on purpose**,
so this reverses a documented choice.

- **Possible shape:** a `[templates.<name>]` table with any step keys, and
  `extends = "<name>"` on a step. Step keys override template keys, and `params` is
  merged recursively.
- **Checkpoints:** the fingerprint uses the merged step, so it does not change unless the
  merged result changes.

### B2. Inputs that name another step's output

- **Decision:** not approved (no consensus). DeepSeek and GLM would build the minimal form
  only if the gate's lint resolved `@` references in the same change; without that,
  `"@split.train"` is no path to lint, and a model reading the split that way escapes the
  ungated-step check. Claude judged the gain (one fewer place to edit) too small for a
  reference syntax that both the runner and the gate must understand. The glob form was
  rejected by all three: it breaks the gate's name-based split detection.

hda-v3's `predict` lists six model paths by hand (`model_<name>`, `meta_<name>`), copied
from the `outputs` of the three train steps. Adding a model means editing two places, and
nothing checks that they agree.

- **Minimal version:** `inputs = { m = "@train_hazard_smart.model" }` resolves to that
  step's declared output path, and an unknown step or key is a `ConfigError`.
- **Fuller version:** a pattern such as `"@train_*.model"` expands to one input per
  matching step (`m.train_hazard_smart`, …).

### B3. Clean outputs before a step reruns

- **Decision:** approved by all three, and built as the opt-in step key `clean_outputs`
  (see `README.md` "Pipeline file" and `GUIDE.md` §10). Opt-in because a failed rerun now
  leaves no previous output, and a step that relied on its directory output surviving
  must create it. The loader refuses a cleaned output outside the workdir or holding the
  state folder, the pipeline file, an import root or a `watch` file. Since PR #13, `-o`
  already warns when it leaves out a step that isn't current, so the case below is now
  covered twice.

Outputs are written in place. A failed rerun leaves the previous or half-written files
behind; the checkpoint is forgotten, so the step does rerun next time.

- A later `-o other_step` run checks only that inputs exist, so it can read a
  half-written file.
- A directory output keeps files that the new run no longer writes. hda-v3's
  `build/report/` could keep a figure that is no longer produced.

**Possible shape:** an opt-in step key `clean_outputs = true` that deletes the declared
outputs before the worker starts. Writing to a temporary path and renaming it on success
would be safer, but `ctx.outputs` would then point somewhere else, which breaks steps that
build paths relative to their outputs.

### B4. Fingerprint huge, unchanging inputs by size and time only

- **Decision:** not approved (all three against). `(size, mtime_ns)` alone is the
  silent-stale cache the runner is built to avoid, and the hash is already cached by size
  and time. The cost came from putting `state_dir` inside `build/`; keep the state folder
  out of anything you delete.

The first run hashes hda-v3's 22 GB input, which takes about 80 s. Later runs reuse the
cached hash, which is keyed by size and `mtime_ns`. The hash cache lives in
`state.json`, so deleting the state folder means paying for the full hash again (hda-v3
made this worse by setting `state_dir` inside `build/`).

**Possible shape:** `[pipeline] trust_stat = ["dataset/"]`. Files under these paths would
be fingerprinted by `(size, mtime_ns)` alone. The trade-off: copying the project without
keeping file times would trigger reruns. This is a small gain, and it would be fine to
reject it.

### Not proposed: running steps in parallel

The three training steps in hda-v3 are independent, but each one already uses every core
and memory is the real limit. Running them in parallel would not have saved time. It
would also need resource hints (`memory_gb`, `threads`) to be safe. Revisit it if a
project with many small, I/O-bound steps turns up.
