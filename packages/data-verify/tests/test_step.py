"""data-verify as a data-pipeline worker, and its command line."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

import data_verify
from data_verify.cli import main
from data_verify.core import REGISTRY
from data_verify.spec import parse, split_input_names
from data_verify.step import _named

PREP = '''
import sys
import pandas as pd
sys.path.insert(0, "{tests}")
from conftest import group_split, make_source


def prepare(ctx):
    frames = group_split(make_source())
    if ctx.params["leak"]:
        stolen = frames["test"].head(25).copy()
        stolen["row_id"] += 10**7
        stolen["customer"] = "X" + stolen["customer"]
        frames["train"] = pd.concat([frames["train"], stolen])
    for name, df in frames.items():
        df.to_parquet(ctx.outputs[name], index=False)


def train(ctx):
    ctx.outputs["model"].write_text("fitted")
'''

PIPELINE = '''
[pipeline]
name = "demo"
python_path = ["steps"]
state_dir = ".pipeline"

[[steps]]
name = "prepare"
function = "prep:prepare"
outputs = {{ train = "build/train.parquet", val = "build/val.parquet", test = "build/test.parquet" }}
params = {{ leak = {leak} }}

[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = {{ train = "build/train.parquet", val = "build/val.parquet", test = "build/test.parquet" }}
outputs = {{ report = "build/verify/report.json", summary = "build/verify/report.md" }}
requires = ["data-verify"]
params = {{ label = "y", id = "row_id", group = "customer", weight = "weight", features = ["f1", "f2", "f3", "f4", "plan", "region"], forbidden = ["*_date"] }}

[[steps]]
name = "train"
function = "prep:train"
inputs = {{ train = "build/train.parquet" }}
outputs = {{ model = "build/model.txt" }}
depends_on = ["verify"]
'''


def make_project(root: Path, leak: bool) -> Path:
    (root / "steps").mkdir(parents=True)
    tests = Path(__file__).parent
    (root / "steps" / "prep.py").write_text(PREP.format(tests=tests))
    (root / "pipeline.toml").write_text(PIPELINE.format(leak=str(leak).lower()))
    return root


def run_pipeline(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "data_pipeline.cli", str(root / "pipeline.toml"),
                           "--no-color"], capture_output=True, text=True, timeout=300)


def test_gate_passes_in_pipeline(tmp_path):
    root = make_project(tmp_path / "ok", leak=False)
    res = run_pipeline(root)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (root / "build/model.txt").exists()
    report = json.loads((root / "build/verify/report.json").read_text())
    assert report["passed"] and report["datasets"]["train"]["path"] == "build/train.parquet"
    assert (root / "build/verify/report.md").read_text().startswith("# data-verify: PASSED")
    again = run_pipeline(root)
    assert "verify" in again.stdout and "cached" in again.stdout


def test_gate_blocks_pipeline(tmp_path):
    root = make_project(tmp_path / "leaky", leak=True)
    res = run_pipeline(root)
    assert res.returncode == 1
    assert "duplicates.cross_split" in res.stdout and "fix:" in res.stdout
    assert not (root / "build/model.txt").exists()
    report = json.loads((root / "build/verify/report.json").read_text())
    assert report["blocking"] == ["duplicates.cross_split"]


def test_cli_run_explain_and_reference(tmp_path, capsys):
    root = make_project(tmp_path / "cli", leak=True)
    assert run_pipeline(root).returncode == 1
    assert main(["run", str(root / "pipeline.toml")]) == 1
    out = capsys.readouterr().out
    assert "BLOCK duplicates.cross_split" in out and "BLOCKED" in out
    assert main(["explain", str(root / "build/verify/report.json")]) == 1
    out = capsys.readouterr().out
    assert "[BLOCK] duplicates.cross_split" in out and "fix:" in out
    assert main(["checks", "--json"]) == 0
    ids = [c["id"] for c in json.loads(capsys.readouterr().out)]
    assert "split.group_overlap" in ids and len(ids) == len(set(ids))
    assert main(["params"]) == 0 and "time_order" in capsys.readouterr().out
    assert main(["template"]) == 0 and "data_verify:verify" in capsys.readouterr().out


def test_cli_standalone_config_and_errors(tmp_path, capsys):
    root = make_project(tmp_path / "st", leak=False)
    run_pipeline(root)
    cfg = tmp_path / "verify.toml"
    cfg.write_text(textwrap.dedent(f'''
        [inputs]
        train = "{root}/build/train.parquet"
        test = "{root}/build/test.parquet"
        [outputs]
        report = "out/report.json"
        [params]
        label = "y"
        group = "customer"
    '''))
    assert main(["run", str(cfg)]) == 0
    assert (tmp_path / "out/report.json").exists()
    cfg.write_text(cfg.read_text().replace('group = "customer"', 'gruop = "customer"'))
    assert main(["run", str(cfg)]) == 2
    assert "did you mean 'group'" in capsys.readouterr().err


def test_lint_and_stale_report(tmp_path, capsys):
    root = make_project(tmp_path / "lint", leak=False)
    assert main(["lint", str(root / "pipeline.toml")]) == 0
    assert "structure only" in capsys.readouterr().out
    assert run_pipeline(root).returncode == 0
    assert main(["lint", str(root / "pipeline.toml")]) == 0
    assert "fully resolved" in capsys.readouterr().out
    toml = root / "pipeline.toml"
    toml.write_text(toml.read_text().replace('label = "y"', 'lable = "y"'))
    assert main(["lint", str(toml)]) == 2
    assert "did you mean" in capsys.readouterr().err
    assert run_pipeline(root).returncode == 1
    assert not (root / "build/verify/report.json").exists()   # no stale verdict


def test_unknown_output_key(tmp_path):
    from types import SimpleNamespace

    from data_verify import SpecError, verify
    ctx = SimpleNamespace(name="v", inputs={}, outputs={"rep": tmp_path / "r.json"}, params={},
                          workdir=tmp_path, log=print)
    with pytest.raises(SpecError, match="'report'"):
        verify(ctx)


def test_cli_run_contract(tmp_path, splits, capsys):
    """Config problems exit 2, a failed run leaves no old verdict, --step picks only
    data-verify steps, and unknown output names are errors."""
    for name, d in splits.items():
        d.to_parquet(tmp_path / f"{name}.parquet")
    cfg = tmp_path / "verify.toml"
    good = textwrap.dedent('''
        inputs = { train = "train.parquet", test = "test.parquet" }
        outputs = { report = "out/report.json" }
        params = { label = "y", group = "customer" }
    ''')
    cfg.write_text(good)
    assert main(["run", str(cfg)]) == 0
    report = tmp_path / "out/report.json"
    assert json.loads(report.read_text())["passed"]

    cfg.write_text(good.replace('label = "y"', 'labl = "y"'))
    assert main(["run", str(cfg)]) == 2
    assert not report.exists()                                    # no stale PASSED verdict

    cfg.write_text("this is [not valid toml\n")
    assert main(["run", str(cfg)]) == 2
    assert "not valid TOML" in capsys.readouterr().err

    (tmp_path / "empty.csv").write_text("")
    cfg.write_text(good.replace("train.parquet", "empty.csv"))
    assert main(["run", str(cfg)]) == 2
    assert "cannot read input empty.csv" in capsys.readouterr().err

    cfg.write_text(good.replace('report = "out/report.json"',
                                'report = "out/report.json", sumary = "out/report.md"'))
    assert main(["run", str(cfg)]) == 2
    assert "did you mean 'summary'" in capsys.readouterr().err

    cfg.write_text(good)
    assert main(["run", str(cfg), "--step", "verify"]) == 2
    assert "only applies to a pipeline file" in capsys.readouterr().err

    pipe = tmp_path / "pipeline.toml"
    step = 'inputs = { train = "train.parquet", test = "test.parquet" }\n' \
           'params = { label = "y", group = "customer" }\n'
    pipe.write_text('[[steps]]\nname = "split"\nfunction = "local:split"\n' + step
                    + '[[steps]]\nname = "verify"\nfunction = "data_verify:verify"\n' + step)
    assert main(["run", str(pipe), "--step", "split"]) == 2
    assert "not a data-verify step" in capsys.readouterr().err
    assert main(["run", str(pipe), "--step", "verify"]) == 0


# -- lint: nothing reads a split without the gate ---------------------------------------------

def _lint_project(root: Path, edit=lambda text: text) -> Path:
    project = make_project(root, leak=False)
    toml = project / "pipeline.toml"
    toml.write_text(edit(toml.read_text()))
    return toml


def _ungated(text: str) -> str:
    return text.replace('depends_on = ["verify"]\n', "")


def test_lint_flags_a_step_reading_a_split_without_the_gate(tmp_path, capsys):
    toml = _lint_project(tmp_path / "p", _ungated)
    assert main(["lint", str(toml)]) == 1
    out = capsys.readouterr().out
    assert "step 'train' reads build/train.parquet without the gate" in out
    assert 'add depends_on = ["verify"]' in out


def test_lint_accepts_the_gate_reached_through_another_step(tmp_path):
    toml = _lint_project(tmp_path / "p", lambda t: _ungated(t).replace(
        'inputs = { train = "build/train.parquet" }\noutputs = { model = "build/model.txt" }',
        'inputs = { train = "build/train.parquet", m = "build/marker.txt" }\n'
        'outputs = { model = "build/model.txt" }') + """
[[steps]]
name = "after_gate"
function = "prep:train"
outputs = { model = "build/marker.txt" }
depends_on = ["verify"]
""")
    assert main(["lint", str(toml)]) == 0


def test_lint_allows_a_step_that_feeds_the_gate(tmp_path):
    toml = _lint_project(tmp_path / "p", lambda t: t.replace(
        'test = "build/test.parquet" }\noutputs = { report',
        'test = "build/test.parquet", stats = "build/stats.json" }\noutputs = { report') + """
[[steps]]
name = "stats"
function = "prep:train"
inputs = { train = "build/train.parquet" }
outputs = { model = "build/stats.json" }
""")
    assert main(["lint", str(toml)]) == 0


def test_lint_flags_a_model_the_gate_merely_runs_after(tmp_path, capsys):
    """Ordering the gate after a step doesn't check what that step read."""
    toml = _lint_project(tmp_path / "p", lambda t: _ungated(t).replace(
        'name = "verify"\n', 'name = "verify"\ndepends_on = ["train"]\n'))
    assert main(["lint", str(toml)]) == 1
    assert "'train' reads" in capsys.readouterr().out


# the raw table comes first and the data is named nothing special, so only `source` tells
# the two apart
SOURCE_ONLY = """
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = {{ {inputs} }}
params = {{ label = "y"{params} }}

[[steps]]
name = "overview"
function = "m:f"
inputs = {{ raw = "build/raw.parquet" }}
outputs = {{ o = "build/o.txt" }}
"""


RAW_FIRST = 'raw = "build/raw.parquet", rows = "build/rows.parquet"'


def test_lint_allows_a_step_reading_only_the_source(tmp_path):
    toml = tmp_path / "pipeline.toml"
    toml.write_text(SOURCE_ONLY.format(inputs=RAW_FIRST, params=', source = "raw"'))
    assert main(["lint", str(toml)]) == 0


def test_lint_reads_the_source_from_a_spec_input(tmp_path):
    (tmp_path / "gate.toml").write_text('source = "raw"\n')
    toml = tmp_path / "pipeline.toml"
    toml.write_text(SOURCE_ONLY.format(inputs=RAW_FIRST + ', spec = "gate.toml"', params=""))
    assert main(["lint", str(toml)]) == 0


def test_lint_flags_a_split_named_only_in_a_spec_input(tmp_path, capsys):
    (tmp_path / "gate.toml").write_text('label = "y"\nsplits = ["a", "b"]\n')
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { a = "build/a.parquet", b = "build/b.parquet", spec = "gate.toml" }

[[steps]]
name = "train"
function = "m:f"
inputs = { a = "build/a.parquet" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 1
    assert "'train' reads build/a.parquet" in capsys.readouterr().out


def test_the_iris_example_passes_lint():
    assert main(["lint", str(Path(__file__).parents[3] / "examples/iris/pipeline.toml")]) == 0


@pytest.mark.parametrize("params, names, expected", [
    ({"label": "y"}, ["train", "test", "aux"], ["train", "test"]),
    ({"label": "y", "source": "raw"}, ["train", "test", "raw"], ["train", "test"]),
    ({"label": "y", "splits": ["a", "b"]}, ["a", "b", "aux"], ["a", "b"]),
    ({"label": "y", "splits": {"train": "a", "test": "b"}}, ["a", "b"], ["a", "b"]),
    ({"label": "y", "split_column": "s", "table": "t", "splits": ["train", "test"]},
     ["t", "aux"], ["t"]),
    ({"label": "y", "split_column": "s", "splits": ["train", "test"]}, ["t"], ["t"]),
    ({"label": "y"}, ["only"], ["only"]),
    ({"label": "y", "source": "raw"}, ["rows", "raw"], ["rows"]),
    ({"label": "y", "splits": "a"}, ["a", "b"], ["a"]),
])
def test_lint_layout_agrees_with_the_real_parser(tmp_path, params, names, expected):
    inputs = {n: tmp_path / f"{n}.parquet" for n in names}
    spec = parse(params, inputs, REGISTRY)
    assert split_input_names(params, inputs) == expected
    assert set(expected) == ({spec.table} if spec.table else set(spec.split_inputs.values()))


def test_lint_sees_a_directory_split_that_is_not_built_yet(tmp_path, capsys):
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { train = "build/train_ds", test = "build/test_ds" }
params = { label = "y" }

[[steps]]
name = "train"
function = "m:f"
inputs = { train = "build/train_ds" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 1
    assert "'train' reads build/train_ds" in capsys.readouterr().out


def test_lint_follows_a_gate_reached_through_a_directory(tmp_path):
    """A step reads a directory another gated step writes into: the same rule as the runner."""
    toml = _lint_project(tmp_path / "p", lambda t: _ungated(t).replace(
        'inputs = { train = "build/train.parquet" }\noutputs = { model = "build/model.txt" }',
        'inputs = { train = "build/train.parquet", gated = "build/gated" }\n'
        'outputs = { model = "build/model.txt" }') + """
[[steps]]
name = "after_gate"
function = "prep:train"
outputs = { model = "build/gated/marker.txt" }
depends_on = ["verify"]
""")
    assert main(["lint", str(toml)]) == 0


CONFIG_REFS = """
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = {{ raw = "build/raw.parquet", a = "build/a.parquet", config = "config.toml" }}
params = {{ label = "y", {layout} }}

[[steps]]
name = "overview"
function = "m:f"
inputs = {{ raw = "build/raw.parquet" }}
outputs = {{ o = "build/o.txt" }}
"""


@pytest.mark.parametrize("layout", [
    'source = { from = "config", key = "source" }',
    'splits = { from = "config", key = "splits" }',
])
def test_lint_resolves_references_in_the_layout(tmp_path, layout):
    (tmp_path / "config.toml").write_text('source = "raw"\nsplits = ["a"]\n')
    toml = tmp_path / "pipeline.toml"
    toml.write_text(CONFIG_REFS.format(layout=layout))
    assert main(["lint", str(toml)]) == 0


def test_lint_flags_a_split_named_through_a_reference(tmp_path, capsys):
    (tmp_path / "config.toml").write_text('splits = ["raw"]\n')
    toml = tmp_path / "pipeline.toml"
    toml.write_text(CONFIG_REFS.format(layout='splits = { from = "config", key = "splits" }'))
    assert main(["lint", str(toml)]) == 1
    assert "'overview' reads build/raw.parquet" in capsys.readouterr().out


def test_lint_says_so_when_a_layout_reference_is_not_built_yet(tmp_path, capsys):
    toml = tmp_path / "pipeline.toml"
    toml.write_text(CONFIG_REFS.format(layout='source = { from = "config", key = "source" }'))
    assert main(["lint", str(toml)]) == 0
    assert "not checked" in capsys.readouterr().out


def test_lint_says_so_when_a_reference_is_malformed(tmp_path, capsys):
    (tmp_path / "config.toml").write_text("not = = toml")
    toml = tmp_path / "pipeline.toml"
    toml.write_text(CONFIG_REFS.format(layout='source = { from = "config", key = "source" }'))
    assert main(["lint", str(toml)]) == 0
    assert "not checked" in capsys.readouterr().out


def test_lint_says_so_when_a_spec_input_is_not_built_yet(tmp_path, capsys):
    toml = tmp_path / "pipeline.toml"
    toml.write_text(SOURCE_ONLY.format(inputs=RAW_FIRST + ', spec = "build/gate.toml"', params=""))
    assert main(["lint", str(toml)]) == 0
    assert "not checked" in capsys.readouterr().out


def test_an_unbuilt_reference_elsewhere_does_not_stop_the_check(tmp_path, capsys):
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { train = "build/train.parquet", test = "build/test.parquet", meta = "build/meta.toml" }
params = { label = { from = "meta", key = "label" } }

[[steps]]
name = "model"
function = "m:f"
inputs = { train = "build/train.parquet" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 1
    assert "'model' reads" in capsys.readouterr().out


def test_lint_reads_a_single_split_name(tmp_path):
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { a = "build/a.parquet", b = "build/b.parquet" }
params = { label = "y", splits = "a" }

[[steps]]
name = "lookup"
function = "m:f"
inputs = { b = "build/b.parquet" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 0


def test_lint_survives_malformed_layouts(tmp_path):
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { a = "build/a.parquet" }
params = { label = "y", splits = [{ a = 1 }] }

[[steps]]
name = "odd"
function = "m:f"
inputs = ["build/a.parquet"]
""")
    assert main(["lint", str(toml)]) in (0, 2)


def test_lint_sees_a_dotted_directory_split_that_is_not_built_yet(tmp_path, capsys):
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { train = "build/train.v2", test = "build/test.v2" }
params = { label = "y" }

[[steps]]
name = "model"
function = "m:f"
inputs = { train = "build/train.v2" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 1
    assert "'model' reads" in capsys.readouterr().out


def test_lint_says_so_when_the_table_is_unclear(tmp_path, capsys):
    """Listing a lookup first would make it the split, and a step reading it the offender."""
    toml = tmp_path / "pipeline.toml"
    toml.write_text(SOURCE_ONLY.format(
        inputs='raw = "build/raw.parquet", aux = "build/aux.parquet", rows = "build/rows.parquet"',
        params=', source = "raw"') + """
[[steps]]
name = "lookup"
function = "m:f"
inputs = { aux = "build/aux.parquet" }
outputs = { o = "build/l.txt" }
""")
    assert main(["lint", str(toml)]) == 0
    assert "not checked" in capsys.readouterr().out


def test_lint_finds_the_table_beside_a_config_file(tmp_path, capsys):
    (tmp_path / "config.json").write_text('{"source": "raw"}')
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { config = "config.json", raw = "build/raw.parquet", rows = "build/rows.parquet" }
params = { label = "y", source = { from = "config", key = "source" } }

[[steps]]
name = "model"
function = "m:f"
inputs = { rows = "build/rows.parquet" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 1
    assert "'model' reads build/rows.parquet" in capsys.readouterr().out


def test_lint_treats_extra_tables_as_auxiliary_when_there_are_no_named_splits(tmp_path):
    toml = tmp_path / "pipeline.toml"
    toml.write_text(SOURCE_ONLY.format(inputs=RAW_FIRST + ', aux = "build/aux.parquet"',
                                       params=', source = "raw"') + """
[[steps]]
name = "lookup"
function = "m:f"
inputs = { aux = "build/aux.parquet" }
outputs = { o = "build/l.txt" }
""")
    assert main(["lint", str(toml)]) == 0


def test_lint_covers_a_table_holding_every_split(tmp_path, capsys):
    toml = tmp_path / "pipeline.toml"
    toml.write_text("""
[[steps]]
name = "verify"
function = "data_verify:verify"
inputs = { rows = "build/rows.parquet", extra = "build/source.parquet" }
params = { label = "y", split_column = "split", table = "rows", splits = ["train", "test"] }

[[steps]]
name = "train"
function = "m:f"
inputs = { rows = "build/rows.parquet" }
outputs = { model = "build/model.txt" }

[[steps]]
name = "side"
function = "m:f"
inputs = { extra = "build/source.parquet" }
outputs = { o = "build/o.txt" }
""")
    assert main(["lint", str(toml)]) == 1
    out = capsys.readouterr().out
    assert "'train' reads build/rows.parquet" in out and "'side'" not in out


def test_lint_follows_a_directory_input(tmp_path, capsys):
    toml = _lint_project(tmp_path / "p", lambda t: _ungated(t)
                         .replace('"build/train.parquet"', '"build/splits/train.parquet"')
                         .replace('"build/val.parquet"', '"build/splits/val.parquet"')
                         .replace('"build/test.parquet"', '"build/splits/test.parquet"')
                         .replace('inputs = { train = "build/splits/train.parquet" }\n'
                                  'outputs = { model',
                                  'inputs = { every = "build/splits" }\noutputs = { model'))
    assert main(["lint", str(toml)]) == 1
    assert "'train' reads" in capsys.readouterr().out


def test_lint_skips_the_step_check_for_a_standalone_config(tmp_path, capsys):
    cfg = tmp_path / "verify.toml"
    cfg.write_text('inputs = { train = "t.csv", test = "u.csv" }\nparams = { label = "y" }\n')
    assert main(["lint", str(cfg)]) == 0


# -- status line ------------------------------------------------------------------------------

@pytest.mark.parametrize("ids, shown", [
    (["a"], "a"),
    (["a", "b", "c"], "a,b,c"),
    (["a", "b", "c", "d", "e"], "a,b,c,+2"),
])
def test_warning_names_are_capped(ids, shown):
    assert _named(ids) == shown


def test_status_line_names_the_warnings(tmp_path):
    root = make_project(tmp_path / "w", leak=False)
    prep = root / "steps" / "prep.py"
    prep.write_text(prep.read_text().replace(
        '    for name, df in frames.items():\n',
        '    for name, df in frames.items():\n        df["f4"] = 1.0\n'))
    res = run_pipeline(root)
    assert res.returncode == 0, res.stdout
    line = next(ln for ln in res.stdout.splitlines() if "verify" in ln and "checks=" in ln)
    assert "warned=features.constant" in line


def test_status_line_has_no_warned_key_without_warnings(tmp_path):
    root = make_project(tmp_path / "c", leak=False)
    res = run_pipeline(root)
    assert "warnings=0" in res.stdout and "warned=" not in res.stdout


# -- versions ---------------------------------------------------------------------------------

def test_package_version_matches_the_code():
    """The runner fingerprints the distribution version, and __version__ reaches reports."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert tomllib.loads(pyproject.read_text())["project"]["version"] == data_verify.__version__
