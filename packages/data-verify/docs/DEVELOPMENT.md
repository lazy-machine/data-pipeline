# Development

From the repo root (one workspace, one virtual environment):

```bash
uv sync
uv run pytest packages/data-verify     # injects every leak/poisoning pattern into clean data
uv run data-verify run packages/data-verify/examples/hard-drive-analysis/verify.toml
                                       # needs ../hard-drive-analysis-v3/build next to the repo
```

Layout of `src/data_verify/`:

- `spec.py`: params parsing
- `core.py`: loading, hashing, registry, Verifier
- `checks/*.py`: one module per area; a check is a function decorated with `@check`
- `stats.py`: single-feature scores, PSI, robust statistics
- `step.py`: runner, report
- `cli.py`
- `reference.py`: params and checks reference
