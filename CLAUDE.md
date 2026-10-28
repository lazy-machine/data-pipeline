# data-verify

Leakage and poisoning gate, used as a data-pipeline step: `function = "data_verify:verify"`.
README.md is the contract (inputs, outputs, params, report format); docs/CHECKS.md lists
every check.

## Commands
- Setup: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check src tests examples`
- Real-data check: `uv run data-verify run examples/hard-drive-analysis/verify.toml`

## Conventions
- A check is a function in `src/data_verify/checks/<area>.py` decorated with `@check(id,
  category, severity, title, **options)`; its docstring is its public description. Return
  `passed(...)`, `failed(message, hint, **details)` or `skipped(reason)`.
- Every new check gets a test in `tests/test_checks.py` that injects the fault into clean
  data and asserts the check fires, while the clean fixtures keep passing.
- Reports must stay deterministic (no timestamps, rounded floats).
- Bump `__version__` in `src/data_verify/__init__.py` on every behaviour change, then
  regenerate docs/CHECKS.md (`uv run data-verify checks --markdown`).
