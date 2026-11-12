# Command line

```
data-verify run pipeline.toml [--step NAME]   run a data-verify step without the orchestrator
data-verify run verify.toml                   standalone config: [inputs] [outputs] [params]
data-verify lint [pipeline.toml] [--step N]   validate params, and that splits pass the gate
data-verify explain REPORT.json [--all]       failed checks, details and fixes
data-verify checks [--json | --markdown]      every check, severity and options
data-verify params [--markdown]               every params key
data-verify template                          step block to paste into pipeline.toml
```

Exit codes: `0` passed, `1` blocked, `2` invalid config.
