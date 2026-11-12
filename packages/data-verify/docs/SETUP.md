# Add it to a pipeline

**1. Install it** in the pipeline project:

```bash
uv add --editable "../data-pipeline[verify]"   # runner + gate, from a sibling checkout of data-pipeline
uv add duckdb                                  # only if you use SQL assertions
```

The [usage guide](../../../docs/GUIDE.md) (§2 and §7) covers the other ways to depend on it and
where the gate goes in a pipeline; `examples/iris` at the repo root is a working example.

**2. Add the step** right after the step that writes the splits, and make every training
and evaluation step depend on it:

```toml
[[steps]]
name = "verify"
description = "Gate: no leakage between splits, no poisoned or invalid data"
function = "data_verify:verify"
inputs = { train = "build/train.parquet", val = "build/val.parquet", test = "build/test.parquet" }
outputs = { report = "build/verify/report.json", summary = "build/verify/report.md" }
requires = ["data-verify"]
params = { label = "churned", id = "row_id", group = "customer_id" }

[[steps]]
name = "train"
depends_on = ["verify"]        # nothing trains on data the gate has not passed
# ...
```

`data-verify template` prints a commented version of this block.

**3. Check the params** with `data-verify lint`. It validates every data-verify step in
`./pipeline.toml` without running anything. (The step itself only sees its params when its
turn comes, after the upstream steps ran.) It also exits 1 if a step reads a split without
the gate upstream of it, directly or through other steps. Steps that feed the gate, and
steps that read only `source`, are allowed. It prints `not checked` and exits 0 when it can't
tell which inputs hold the splits: a reference or `spec` isn't readable yet, a split name
isn't an input, an unbuilt path has an unknown suffix, or `source` is set with several other
tables and no split names. With a config file listed first and `source` set, lint looks at
the data table while the gate takes the config as the split and fails, so list the data first.
In CI, treat `not checked` as a failure if you rely on this check: the exit code doesn't show it.

## Declare every role that exists in your data

The gate can only check what it knows:

- **Without `group`**, rows of one patient in train and test look like different rows. The
  `split.undeclared_group` warning tries to spot such columns.
- **Without `time`**, a random split of time-dependent data looks fine.
- **With the default `features = "auto"`**, every column without a role is treated as a
  feature, including stray keys. List `features` explicitly when you know them.

## Caching

- data-pipeline fingerprints `data_verify/__init__.py` (an installed module), not the rest
  of the package. The version string lives there, so upgrading data-verify reruns the gate.
- After editing data-verify itself without a version bump (with an editable install, any
  edit to the source), run `data-pipeline -r verify`.
- The gate's own params, inputs and `watch` files invalidate it as usual.
