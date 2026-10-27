import io
import json
import os
import textwrap
from pathlib import Path

import pytest

from data_pipeline import config
from data_pipeline.cli import main
from data_pipeline.orchestrator import Orchestrator
from data_pipeline.report import Reporter

STEPS = '''
import os, json, sys, time

_CALLS = []

def copy(ctx):
    text = ctx.inputs["src"].read_text()
    ctx.outputs["dst"].write_text(text.upper() + ctx.params.get("suffix", ""))
    return {"chars": len(text)}

def count(ctx):
    n = len(ctx.inputs["src"].read_text())
    ctx.outputs["n"].write_text(str(n))
    return f"n={n}"

def boom(ctx):
    print("about to fail")
    raise RuntimeError("kaboom")

def no_output(ctx):
    pass

def sleepy(ctx):
    time.sleep(10)

def pid(ctx):
    _CALLS.append(1)
    ctx.outputs["out"].write_text(json.dumps({"pid": os.getpid(), "calls": len(_CALLS)}))

def hard_exit(ctx):
    os._exit(3)

def hello(ctx):
    print("hello")

def start_then_hang(ctx):
    print("started", flush=True)
    time.sleep(10)
'''


def make(tmp_path: Path, steps_toml: str, header: str = "") -> Path:
    (tmp_path / "steps").mkdir(exist_ok=True)
    (tmp_path / "steps" / "fns.py").write_text(STEPS)
    (tmp_path / "in.txt").write_text("hello")
    cfg = tmp_path / "pipeline.toml"
    cfg.write_text(textwrap.dedent(f"""
        [pipeline]
        name = "t"
        python_path = ["steps"]
        {header}
    """) + textwrap.dedent(steps_toml))
    return cfg


def run(cfg: Path, **kw):
    buf = io.StringIO()
    result = Orchestrator(config.load(cfg), Reporter(buf, color=False)).run(**kw)
    return result, buf.getvalue()


CHAIN = """
[[steps]]
name = "upper"
function = "fns:copy"
inputs = { src = "in.txt" }
outputs = { dst = "out/upper.txt" }

[[steps]]
name = "count"
function = "fns:count"
inputs = { src = "out/upper.txt" }
outputs = { n = "out/n.txt" }
"""


def statuses(result):
    return {k: o.status for k, o in result.outcomes.items()}


def test_runs_then_caches(tmp_path):
    cfg = make(tmp_path, CHAIN)
    r, out = run(cfg)
    assert r.ok and statuses(r) == {"upper": "ran", "count": "ran"}
    assert (tmp_path / "out/upper.txt").read_text() == "HELLO"
    assert (tmp_path / "out/n.txt").read_text() == "5"
    assert "chars=5" in out and "n=5" in out

    r, out = run(cfg)
    assert statuses(r) == {"upper": "cached", "count": "cached"}
    assert "chars=5" in out  # cached steps still show their last summary


def test_input_change_invalidates_downstream(tmp_path):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    (tmp_path / "in.txt").write_text("hello world")
    r, out = run(cfg)
    assert statuses(r) == {"upper": "ran", "count": "ran"}
    assert "inputs changed: src" in out


def test_identical_upstream_output_keeps_downstream_cached(tmp_path):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    (tmp_path / "in.txt").write_text("HELLO")  # different input, same uppercased output
    r, _ = run(cfg)
    assert statuses(r) == {"upper": "ran", "count": "cached"}


def test_param_and_code_changes_invalidate(tmp_path):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    cfg.write_text(cfg.read_text().replace('outputs = { dst = "out/upper.txt" }',
                                           'outputs = { dst = "out/upper.txt" }\nparams = { suffix = "!" }'))
    r, out = run(cfg)
    assert r.outcomes["upper"].reason == "config changed"
    assert r.outcomes["count"].status == "ran"

    fns = tmp_path / "steps" / "fns.py"
    fns.write_text(fns.read_text() + "\n# edit\n")
    r, _ = run(cfg)
    assert r.outcomes["upper"].reason == "code changed"


def test_missing_or_modified_output_reruns(tmp_path):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    (tmp_path / "out/n.txt").unlink()
    r, _ = run(cfg)
    assert statuses(r) == {"upper": "cached", "count": "ran"}
    assert r.outcomes["count"].reason == "output missing: n"

    (tmp_path / "out/n.txt").write_text("tampered")
    r, _ = run(cfg)
    assert r.outcomes["count"].reason == "output modified: n"


def test_force_and_rerun(tmp_path):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    r, _ = run(cfg, rerun=["count"])
    assert statuses(r) == {"upper": "cached", "count": "ran"}
    r, _ = run(cfg, force=True)
    assert statuses(r) == {"upper": "ran", "count": "ran"}


def test_critical_failure_halts_and_dumps_output(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "bad"
        function = "fns:boom"

        [[steps]]
        name = "after"
        function = "fns:copy"
        inputs = { src = "in.txt" }
        outputs = { dst = "x.txt" }
    """)
    r, out = run(cfg)
    assert not r.ok and r.halted_by == "bad"
    assert "after" not in r.outcomes
    assert "about to fail" in out and "RuntimeError: kaboom" in out and "Traceback" in out
    assert "halted at 'bad'" in out


def test_non_critical_failure_warns_and_skips_dependents(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "flaky"
        function = "fns:boom"
        outputs = { o = "flaky.txt" }
        critical = false

        [[steps]]
        name = "uses_flaky"
        function = "fns:count"
        inputs = { src = "flaky.txt" }
        outputs = { n = "n.txt" }
        critical = false

        [[steps]]
        name = "independent"
        function = "fns:copy"
        inputs = { src = "in.txt" }
        outputs = { dst = "ok.txt" }
    """)
    r, out = run(cfg)
    assert r.ok
    assert statuses(r) == {"flaky": "warned", "uses_flaky": "skipped", "independent": "ran"}
    assert "RuntimeError: kaboom" in out and "completed with warnings" in out


def test_critical_step_blocked_by_non_critical_failure_halts(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "flaky"
        function = "fns:boom"
        outputs = { o = "flaky.txt" }
        critical = false

        [[steps]]
        name = "needs_it"
        function = "fns:count"
        inputs = { src = "flaky.txt" }
        outputs = { n = "n.txt" }
    """)
    r, _ = run(cfg)
    assert r.halted_by == "needs_it"
    assert "upstream" in r.outcomes["needs_it"].detail


def test_missing_output_timeout_and_hard_exit_fail(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "lazy"
        function = "fns:no_output"
        outputs = { o = "never.txt" }
        critical = false

        [[steps]]
        name = "slow"
        function = "fns:sleepy"
        timeout = 0.5
        critical = false

        [[steps]]
        name = "crash"
        function = "fns:hard_exit"
        critical = false
    """)
    r, _ = run(cfg)
    assert "did not write output" in r.outcomes["lazy"].detail
    assert "timed out" in r.outcomes["slow"].detail
    assert r.outcomes["crash"].detail == "exit code 3"
    assert not (tmp_path / ".pipeline/state.json").exists() or \
        not json.loads((tmp_path / ".pipeline/state.json").read_text())["steps"]


def test_each_step_runs_in_a_fresh_process(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "a"
        function = "fns:pid"
        outputs = { out = "a.json" }

        [[steps]]
        name = "b"
        function = "fns:pid"
        outputs = { out = "b.json" }
    """)
    run(cfg)
    a = json.loads((tmp_path / "a.json").read_text())
    b = json.loads((tmp_path / "b.json").read_text())
    assert a["pid"] != b["pid"] != os.getpid()
    assert a["calls"] == b["calls"] == 1  # module globals did not carry over


def test_preflight_blocks_critical_problems(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "ok"
        function = "fns:copy"
        inputs = { src = "in.txt" }
        outputs = { dst = "x.txt" }

        [[steps]]
        name = "needs_pkg"
        function = "fns:copy"
        inputs = { src = "missing.txt" }
        outputs = { dst = "y.txt" }
        requires = ["definitely-not-installed-pkg", "pytest>=999"]
    """)
    r, out = run(cfg)
    assert not r.ok and r.outcomes == {}
    assert "nothing was run" in out
    assert "package not installed: definitely-not-installed-pkg" in out
    assert "does not satisfy" in out and "input 'src' not found: missing.txt" in out


def test_preflight_problem_in_non_critical_step_only_warns(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "optional"
        function = "nomodule:fn"
        critical = false

        [[steps]]
        name = "ok"
        function = "fns:copy"
        inputs = { src = "in.txt" }
        outputs = { dst = "x.txt" }
    """)
    r, out = run(cfg)
    assert r.ok and statuses(r) == {"optional": "warned", "ok": "ran"}
    assert "module not found: nomodule" in out


def test_dry_run_runs_nothing(tmp_path):
    cfg = make(tmp_path, CHAIN)
    r, out = run(cfg, dry_run=True)
    assert statuses(r) == {"upper": "would run", "count": "would run"}
    assert not (tmp_path / "out").exists()


def test_dependency_order_inferred_from_files(tmp_path):
    cfg = make(tmp_path, """
        [[steps]]
        name = "count"
        function = "fns:count"
        inputs = { src = "out/upper.txt" }
        outputs = { n = "out/n.txt" }

        [[steps]]
        name = "upper"
        function = "fns:copy"
        inputs = { src = "in.txt" }
        outputs = { dst = "out/upper.txt" }
    """)
    assert [s.name for s in config.load(cfg).steps] == ["upper", "count"]


@pytest.mark.parametrize("steps, msg", [
    ('[[steps]]\nname = "a"\nfunction = "nocolon"\n', "function must look like"),
    ('[[steps]]\nname = "a"\nfunction = "m:f"\n[[steps]]\nname = "a"\nfunction = "m:f"\n',
     "duplicate step name"),
    ('[[steps]]\nname = "a"\nfunction = "m:f"\ndepends_on = ["zzz"]\n', "unknown step 'zzz'"),
    ('[[steps]]\nname = "a"\nfunction = "m:f"\ncolour = 1\n', "unknown key"),
    ('[[steps]]\nname = "a"\nfunction = "m:f"\ninputs = {i = "b.txt"}\noutputs = {o = "a.txt"}\n'
     '[[steps]]\nname = "b"\nfunction = "m:f"\ninputs = {i = "a.txt"}\noutputs = {o = "b.txt"}\n',
     "dependency cycle"),
    ('[[steps]]\nname = "a"\nfunction = "m:f"\noutputs = {o = "x"}\n'
     '[[steps]]\nname = "b"\nfunction = "m:f"\noutputs = {o = "x"}\n', "produced by both"),
])
def test_config_validation(tmp_path, steps, msg):
    cfg = make(tmp_path, steps)
    with pytest.raises(config.ConfigError, match=msg):
        config.load(cfg)


def test_cli_exit_codes(tmp_path, capsys):
    cfg = make(tmp_path, CHAIN)
    assert main([str(cfg), "--no-color"]) == 0
    assert main([str(cfg), "--no-color", "--only", "nope"]) == 2
    assert main([str(tmp_path / "missing.toml")]) == 2
    bad = make(tmp_path, '[[steps]]\nname = "x"\nfunction = "fns:boom"\n')
    assert main([str(bad), "--no-color"]) == 1


def test_step_log_is_written(tmp_path):
    cfg = make(tmp_path, '[[steps]]\nname = "greet"\nfunction = "fns:hello"\n')
    run(cfg)
    log = tmp_path / ".pipeline/logs/greet.log"
    assert "hello" in log.read_text()
    mtime = log.stat().st_mtime_ns
    r, _ = run(cfg)
    assert statuses(r) == {"greet": "cached"}
    assert log.stat().st_mtime_ns == mtime


def test_failed_step_log_and_dump_path(tmp_path):
    cfg = make(tmp_path, '[[steps]]\nname = "bad"\nfunction = "fns:boom"\n')
    _, out = run(cfg)
    text = (tmp_path / ".pipeline/logs/bad.log").read_text()
    assert "about to fail" in text and "kaboom" in text
    assert "log: " in out and "bad.log" in out


HANG = '[[steps]]\nname = "hang"\nfunction = "fns:start_then_hang"\ntimeout = {}\n'


def test_timeout_keeps_partial_output(tmp_path):
    cfg = make(tmp_path, HANG.format(2))
    r, _ = run(cfg)
    o = r.outcomes["hang"]
    assert "timed out" in o.detail
    assert "started" in o.output
    assert "started" in (tmp_path / ".pipeline/logs/hang.log").read_text()


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("stream", [io.StringIO, FakeTTY])
def test_follow_streams_before_step_ends(tmp_path, stream):
    cfg = make(tmp_path, HANG.format(2))
    buf = stream()
    Orchestrator(config.load(cfg), Reporter(buf, color=False, follow=True)).run()
    out = buf.getvalue()
    assert out.count("started") == 1
    assert out.index("started") < out.index("timed out")
    assert "(output above)" in out and "log: " in out
    if stream is FakeTTY:  # the "… running" placeholder is erased before the first line
        assert "running\r\033[K[1/1] hang\n    │ started\n" in out
