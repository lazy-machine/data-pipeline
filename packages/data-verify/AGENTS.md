# data-verify

Leakage and poisoning gate, used as a data-pipeline step: `function = "data_verify:verify"`.
It is a member of the data-pipeline uv workspace (the repo root's `AGENTS.md` holds the
shared principles). README.md is the overview and points into docs/: docs/CONTRACT.md is
the contract (inputs, outputs, report format), docs/PARAMS.md the params, docs/CHECKS.md
every check, and the other pages are guides (setup, recipes, coverage, CLI, development).

## Commands (from the repo root)
- Setup: `uv sync`
- Tests: `uv run pytest packages/data-verify`
- Lint: `uv run ruff check packages/data-verify`
- Real-data check: `uv run data-verify run packages/data-verify/examples/hard-drive-analysis/verify.toml`
  (needs `../hard-drive-analysis-v3/build` next to this repo)

## Conventions
- A check is a function in `src/data_verify/checks/<area>.py` decorated with `@check(id,
  category, severity, title, **options)`; its docstring is its public description. Return
  `passed(...)`, `failed(message, hint, **details)` or `skipped(reason)`.
- Every new check gets a test in `tests/test_checks.py` that injects the fault into clean
  data and asserts the check fires, while the clean fixtures keep passing.
- Reports must stay deterministic (no timestamps, rounded floats).
- Bump `__version__` in `src/data_verify/__init__.py` on every behaviour change, then
  regenerate docs/CHECKS.md from `uv run data-verify checks --markdown`.
  data-pipeline fingerprints only that `__init__.py` for an installed step package, so the
  bump is what reruns the gate in every project that uses it.
- When a params key changes, paste `uv run data-verify params --markdown` into docs/PARAMS.md.
- Docs: README.md stays short (what it is, quick start, a table of the docs/ pages); the
  detail goes in docs/. A new page gets a row in that table.
- Dependencies stay here, never in the runner: the runner must remain `packaging`-only.
