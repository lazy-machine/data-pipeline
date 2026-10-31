# data-pipeline

A TOML-driven, checkpointed pipeline runner for data analysis and machine learning, plus the
data gate that runs before any model. Two packages in one uv workspace:

| package | where | import | depends on |
|---|---|---|---|
| `data-pipeline` | `src/data_pipeline/` | the runner: `data-pipeline` CLI | `packaging` only |
| `data-verify` | `packages/data-verify/` | the gate step: `function = "data_verify:verify"` | pandas, numpy, pyarrow (duckdb for SQL assertions) |

Docs: `README.md` is the runner reference, `docs/GUIDE.md` is how to use it (start there
when building a project), `packages/data-verify/README.md` is the gate's contract, and
`packages/data-verify/docs/CHECKS.md` lists every check.

## Principles

Every change here, and every project built on this, keeps to these three:

- **Minimal.** The runner stays the standard library plus `packaging`. New capability ships
  as a step package (as data-verify does), not as runner code; change the runner only for
  what a step cannot do (ordering, caching, isolation, reporting). No new dependency, flag
  or TOML key without a case that needs it.
- **Reusable.** A step is a plain function `module:function` taking a `StepContext`.
  Everything it reads comes through `inputs` and `params` and everything it writes goes to
  `outputs`. Steps used by more than one project live in an installable package and are
  referenced by name, never copied between projects.
- **Verifiable.** Every file a step reads or writes is declared. Outputs are deterministic
  (seeded, sorted, no timestamps), so caching stays correct. Steps fail loudly. Data reaches
  a model only through the `verify` gate (`depends_on = ["verify"]`), and every test injects
  the fault it guards against.

## Commands (from the repo root)

```bash
uv sync                                          # runner + gate + dev + example dependencies
uv run pytest                                    # both test suites
uv run pytest tests                              # runner only
uv run pytest packages/data-verify               # gate only
uv run ruff check packages/data-verify           # lint the gate (the runner has no lint config yet)
uv run data-pipeline examples/iris/pipeline.toml # end-to-end example, gate included
uv run data-verify checks                        # every gate check and its options
```

## Changing this repo

- Don't break the public interface: `StepContext` fields, `Orchestrator(pipeline,
  reporter).run(only=, force=, rerun=, dry_run=)`, the CLI flags, the TOML keys, and the
  gate's params, report format and `function = "data_verify:verify"`. Adding is fine.
- Update the docs in the same commit as the behaviour: `README.md` and `docs/GUIDE.md` for
  the runner, and `packages/data-verify/README.md` for the gate.
- Gate conventions (how to add a check, version bumps, regenerating `CHECKS.md`) are in
  `packages/data-verify/AGENTS.md`.
- Match the style: `from __future__ import annotations`, type hints, short docstrings that
  say *why*, and no comments that restate the code.

## Using it from another project

Depend on it, don't copy its source (details in `docs/GUIDE.md` §2):

```bash
uv add --editable "../data-pipeline[verify]"     # runner + gate, from a sibling checkout
uv run data-pipeline -n                          # check the plan before running
```

Then follow `docs/GUIDE.md` §2 to lay out the project, §7 to place the gate, and the §14
checklist before calling it done.

## Layout

```
src/data_pipeline/          the runner (config, checkpoint, orchestrator, worker, context, report, cli)
tests/                      runner tests
packages/data-verify/       the gate: src/data_verify/, tests/, docs/CHECKS.md, examples/
examples/iris/              runnable example: clean → split → verify → eda/train → evaluate → report
docs/GUIDE.md               the usage guide
```
