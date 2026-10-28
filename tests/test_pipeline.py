"""data-verify as a data-pipeline worker, and its command line."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from data_verify.cli import main

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
