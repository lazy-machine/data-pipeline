import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from data_pipeline import checkpoint, config
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

def gate(ctx):
    if "BAD" in ctx.inputs["src"].read_text():
        raise RuntimeError("gate blocked")
    ctx.outputs["report"].write_text("ok")

def make_dir(ctx):
    ctx.outputs["d"].mkdir(exist_ok=True)
    (ctx.outputs["d"] / "a.txt").write_text(ctx.params["v"])

def read_dir(ctx):
    files = sorted(ctx.inputs["d"].glob("*"))
    ctx.outputs["o"].write_text("".join(f.read_text() for f in files))

def write_param(ctx):
    ctx.outputs["o"].write_text(ctx.params["v"])

def edit_input_then_read(ctx):
    ctx.inputs["src"].write_text("edited during the run")
    ctx.outputs["dst"].write_text(ctx.inputs["src"].read_text())

def fail_on_flag(ctx):
    if (ctx.workdir / "flag").exists():
        raise RuntimeError("flag set")
    ctx.outputs["dst"].write_text("done")

def exit_zero_early(ctx):
    ctx.outputs["dst"].write_text("partial")
    sys.stdout.flush()
    os._exit(0)

def start_then_finish(ctx):
    print("started", flush=True)
    time.sleep(1.5)
    ctx.outputs["dst"].write_text("finished")

def spawn_child_then_hang(ctx):
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (ctx.workdir / "child.pid").write_text(str(child.pid))
    print("started", flush=True)
    time.sleep(60)
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
    ('[[steps]]\nname = "a"\nfunction = "m:f"\nenv = "X"\n', "env must be a list of strings"),
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


# -- --only and the steps it leaves out ---------------------------------------------------

GATED = """
[[steps]]
name = "split"
function = "fns:copy"
inputs = { src = "in.txt" }
outputs = { dst = "out/split.txt" }

[[steps]]
name = "verify"
function = "fns:gate"
inputs = { src = "out/split.txt" }
outputs = { report = "out/report.txt" }

[[steps]]
name = "train"
function = "fns:count"
inputs = { src = "out/split.txt" }
outputs = { n = "out/model.txt" }
depends_on = ["verify"]
"""


def test_only_warns_when_a_blocked_gate_is_left_out(tmp_path):
    cfg = make(tmp_path, GATED)
    run(cfg)
    (tmp_path / "in.txt").write_text("bad")
    result, _ = run(cfg)
    assert result.halted_by == "verify"

    result, out = run(cfg, only=["train"])
    assert result.ok
    assert "--only left out 'verify', which is not up to date (no checkpoint)" in out
    assert "'train' depends on it" in out
    assert "completed with warnings" in out


def test_only_warns_when_a_selected_step_makes_the_gate_stale(tmp_path):
    cfg = make(tmp_path, GATED)
    run(cfg)
    (tmp_path / "in.txt").write_text("changed")  # 'verify' is current until 'split' reruns

    _, out = run(cfg, only=["split", "train"])
    assert "--only left out 'verify', which is not up to date (inputs changed: src)" in out

    run(cfg)
    (tmp_path / "in.txt").write_text("changed again")
    result, out = run(cfg, only=["split", "train"], dry_run=True)
    assert "--only left out 'verify', which is not up to date (after split)" in out
    assert result.outcomes["split"].status == "would run"
    assert "dry run complete with warnings" in out


def test_only_is_quiet_when_left_out_steps_are_current(tmp_path):
    cfg = make(tmp_path, GATED)
    run(cfg)
    result, out = run(cfg, only=["train"], rerun=["train"])
    assert statuses(result) == {"train": "ran"}
    assert not result.warnings and "left out" not in out


def test_only_warns_once_per_left_out_step(tmp_path):
    cfg = make(tmp_path, GATED + """
[[steps]]
name = "evaluate"
function = "fns:count"
inputs = { src = "out/model.txt" }
outputs = { n = "out/metrics.txt" }
depends_on = ["verify"]
""")
    run(cfg)
    (tmp_path / "out" / "report.txt").unlink()
    result, _ = run(cfg, only=["train", "evaluate"])
    assert len(result.warnings) == 1


def test_a_short_step_name_inside_an_earlier_warning_is_still_warned_about(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "gate_a"
function = "fns:gate"
inputs = { src = "in.txt" }
outputs = { report = "out/gate.txt" }

[[steps]]
name = "a"
function = "fns:copy"
inputs = { src = "in.txt" }
outputs = { dst = "out/a.txt" }

[[steps]]
name = "train"
function = "fns:count"
inputs = { src = "out/a.txt" }
outputs = { n = "out/n.txt" }
depends_on = ["gate_a"]
""")
    run(cfg)
    (tmp_path / "in.txt").write_text("changed")
    result, _ = run(cfg, only=["train"])
    assert [w.split("'")[1] for w in result.warnings] == ["gate_a", "a"]


# -- dependencies through directories -----------------------------------------------------

def test_input_inside_a_directory_output_runs_after_it(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "use"
function = "fns:copy"
inputs = { src = "out/parts/a.txt" }
outputs = { dst = "out/final.txt" }

[[steps]]
name = "produce"
function = "fns:make_dir"
outputs = { d = "out/parts" }
params = { v = "old" }
""")
    result, _ = run(cfg)  # preflight must accept a file a directory output will hold
    assert list(result.outcomes) == ["produce", "use"]
    assert (tmp_path / "out" / "final.txt").read_text() == "OLD"

    cfg.write_text(cfg.read_text().replace('"old"', '"new"'))
    result, _ = run(cfg)
    assert statuses(result) == {"produce": "ran", "use": "ran"}
    assert (tmp_path / "out" / "final.txt").read_text() == "NEW"


def test_directory_input_runs_after_a_step_writing_into_it(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "use"
function = "fns:read_dir"
inputs = { d = "out/parts" }
outputs = { o = "out/final.txt" }

[[steps]]
name = "produce"
function = "fns:write_param"
outputs = { o = "out/parts/a.txt" }
params = { v = "old" }
""")
    result, _ = run(cfg)
    assert list(result.outcomes) == ["produce", "use"]
    assert (tmp_path / "out" / "final.txt").read_text() == "old"


def test_failed_directory_producer_blocks_the_reader_of_a_file_inside(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "produce"
function = "fns:boom"
outputs = { d = "out/parts" }
critical = false

[[steps]]
name = "use"
function = "fns:copy"
inputs = { src = "out/parts/a.txt" }
outputs = { dst = "out/final.txt" }
critical = false
""")
    (tmp_path / "out" / "parts").mkdir(parents=True)
    (tmp_path / "out" / "parts" / "a.txt").write_text("stale")
    result, _ = run(cfg)
    assert statuses(result) == {"produce": "warned", "use": "skipped"}


@pytest.mark.parametrize("steps, msg", [
    ("""
[[steps]]
name = "a"
function = "fns:no_output"
outputs = { o = "out/x" }
[[steps]]
name = "b"
function = "fns:no_output"
outputs = { o = "out/x/y.csv" }
""", "outputs overlap: 'out/x' of 'a' and 'out/x/y.csv' of 'b'"),
    ("""
[[steps]]
name = "a"
function = "fns:no_output"
inputs = { i = "out/x/y.csv" }
outputs = { o = "out/x" }
""", "consumes its own output"),
    ("""
[[steps]]
name = "a"
function = "fns:no_output"
inputs = { i = "out" }
outputs = { o = "out/x" }
""", "consumes its own output"),
])
def test_overlapping_paths_are_rejected(tmp_path, steps, msg):
    with pytest.raises(config.ConfigError, match=msg):
        config.load(make(tmp_path, steps))


def test_one_step_may_declare_a_directory_and_a_file_inside_it(tmp_path):
    config.load(make(tmp_path, """
[[steps]]
name = "a"
function = "fns:no_output"
outputs = { d = "out/x", f = "out/x/y.csv" }
"""))


# -- stopping a step stops what it started --------------------------------------------------

needs_groups = pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups")

SPAWNER = """
[[steps]]
name = "spawn"
function = "fns:spawn_child_then_hang"
outputs = { o = "out/o.txt" }
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")  # a killed child stays a zombie until it is reaped
    return not (stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")


def wait_until(condition, seconds: float = 10) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@needs_groups
def test_timeout_kills_the_steps_children(tmp_path):
    cfg = make(tmp_path, SPAWNER + "timeout = 3\n")
    result, _ = run(cfg)
    assert "timed out" in result.outcomes["spawn"].detail
    child = int((tmp_path / "child.pid").read_text())
    assert wait_until(lambda: not alive(child))


@needs_groups
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interrupting_the_runner_stops_the_step_and_keeps_its_traceback(tmp_path, sig):
    cfg = make(tmp_path, SPAWNER)
    proc = subprocess.Popen([sys.executable, "-m", "data_pipeline.cli", str(cfg), "--no-color"],
                            stdout=subprocess.PIPE, text=True)
    pid_file = tmp_path / "child.pid"
    log = tmp_path / ".pipeline" / "logs" / "spawn.log"
    try:
        assert wait_until(lambda: pid_file.exists() and pid_file.read_text()
                          and log.exists() and "started" in log.read_text())
        child = int(pid_file.read_text())
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=15)
    finally:
        proc.kill()
    assert proc.returncode == 130
    assert "interrupted" in out
    assert wait_until(lambda: not alive(child))
    assert "KeyboardInterrupt" in log.read_text()


def start_runner(tmp_path: Path, cfg: Path, **popen) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "data_pipeline.cli", str(cfg), "--no-color"],
                            stdout=subprocess.PIPE, text=True, **popen)
    log = tmp_path / ".pipeline" / "logs" / "spawn.log"
    assert wait_until(lambda: log.exists() and "started" in log.read_text())
    return proc


@needs_groups
@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM])
def test_a_signal_the_caller_ignores_stays_ignored(tmp_path, sig):
    """`nohup data-pipeline` must survive logout: an inherited SIG_IGN is the caller's choice."""
    cfg = make(tmp_path, """
[[steps]]
name = "spawn"
function = "fns:start_then_finish"
outputs = { dst = "out/o.txt" }
""")
    proc = start_runner(tmp_path, cfg, preexec_fn=lambda: signal.signal(sig, signal.SIG_IGN))
    try:
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=15)
    finally:
        proc.kill()
    assert proc.returncode == 0, out
    assert (tmp_path / "out" / "o.txt").read_text() == "finished"


@needs_groups
@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGQUIT])
def test_a_step_does_not_outlive_a_killed_runner(tmp_path, sig):
    cfg = make(tmp_path, SPAWNER)
    proc = start_runner(tmp_path, cfg)
    child = int((tmp_path / "child.pid").read_text())
    try:
        proc.send_signal(sig)
        proc.wait(timeout=15)
    finally:
        proc.kill()
    assert wait_until(lambda: not alive(child))


@needs_groups
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_step_does_not_outlive_a_python_caller(tmp_path, sig):
    """Only the CLI turns these signals into an interrupt; a Python caller just dies."""
    cfg = make(tmp_path, SPAWNER)
    code = ("import sys; from data_pipeline import config; "
            "from data_pipeline.orchestrator import Orchestrator; "
            "Orchestrator(config.load(sys.argv[1])).run()")
    proc = subprocess.Popen([sys.executable, "-c", code, str(cfg)], stdout=subprocess.DEVNULL)
    pid_file = tmp_path / "child.pid"
    try:
        assert wait_until(lambda: pid_file.exists() and pid_file.read_text())
        child = int(pid_file.read_text())
        proc.send_signal(sig)
        proc.wait(timeout=15)
    finally:
        proc.kill()
    assert wait_until(lambda: not alive(child))


# -- guarantees a checkpoint depends on -----------------------------------------------------

def test_input_edited_during_the_run_is_not_checkpointed(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "a"
function = "fns:edit_input_then_read"
inputs = { src = "in.txt" }
outputs = { dst = "out/o.txt" }
""")
    run(cfg)  # the output came from the edited text, not from what was fingerprinted
    (tmp_path / "in.txt").write_text("hello")
    result, _ = run(cfg)
    assert statuses(result) == {"a": "ran"}


def test_failed_rerun_does_not_leave_the_old_checkpoint_valid(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "a"
function = "fns:fail_on_flag"
outputs = { dst = "out/o.txt" }
""")
    run(cfg)
    (tmp_path / "flag").touch()
    result, _ = run(cfg, rerun=["a"])
    assert result.halted_by == "a"
    (tmp_path / "flag").unlink()
    result, _ = run(cfg)
    assert statuses(result) == {"a": "ran"}


def test_worker_that_exits_zero_without_a_result_fails(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "a"
function = "fns:exit_zero_early"
outputs = { dst = "out/o.txt" }
""")
    result, _ = run(cfg)
    assert result.halted_by == "a"
    assert result.outcomes["a"].detail == "worker exited without reporting a result"


# -- package versions and declared environment variables ------------------------------------

PINNED = """
[[steps]]
name = "a"
function = "fns:copy"
inputs = { src = "in.txt" }
outputs = { dst = "out/o.txt" }
requires = ["packaging>=20", "pytest"]
env = ["DP_TEST_SECRET"]
"""


def fake_versions(monkeypatch, **versions):
    real = checkpoint.installed_version
    monkeypatch.setattr(checkpoint, "installed_version",
                        lambda name: versions.get(name) or real(name))


def state_of(tmp_path: Path) -> dict:
    return json.loads((tmp_path / ".pipeline" / "state.json").read_text())


def test_package_version_change_reruns(tmp_path, monkeypatch):
    cfg = make(tmp_path, PINNED)
    fake_versions(monkeypatch, packaging="1.0")
    run(cfg)
    result, _ = run(cfg)
    assert statuses(result) == {"a": "cached"}

    fake_versions(monkeypatch, packaging="2.0")
    result, _ = run(cfg, dry_run=True)
    assert result.outcomes["a"].detail == "packages changed: packaging 1.0 → 2.0"
    result, _ = run(cfg)
    assert result.outcomes["a"].reason == "packages changed: packaging 1.0 → 2.0"


def test_many_package_changes_are_shortened(tmp_path, monkeypatch):
    cfg = make(tmp_path, PINNED.replace('["packaging>=20", "pytest"]', '["p1", "p2", "p3", "p4"]'))
    versions = {f"p{i}": "1" for i in range(1, 5)}
    monkeypatch.setattr(checkpoint, "installed_version", lambda name: versions[name])
    monkeypatch.setattr("data_pipeline.orchestrator._check_requirement", lambda spec: None)
    run(cfg)
    versions.update({k: "2" for k in versions})
    result, _ = run(cfg)
    assert result.outcomes["a"].reason == "packages changed: p1 1 → 2, p2 1 → 2, p3 1 → 2, …"


def test_adding_a_requirement_reruns_and_names_it(tmp_path):
    cfg = make(tmp_path, PINNED.replace('"pytest"', '"packaging"'))
    run(cfg)
    cfg.write_text(cfg.read_text().replace('["packaging>=20", "packaging"]',
                                           '["packaging>=20", "pytest"]'))
    result, _ = run(cfg)
    assert result.outcomes["a"].reason.startswith("packages changed: pytest none → ")


def test_import_name_is_recorded_as_import_only(tmp_path):
    cfg = make(tmp_path, PINNED.replace('"pytest"', '"sqlite3"'))
    run(cfg)
    assert state_of(tmp_path)["steps"]["a"]["packages"]["sqlite3"] == "import-only"


def test_env_change_reruns_and_value_not_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("DP_TEST_SECRET", "s3cr3t")
    cfg = make(tmp_path, PINNED)
    run(cfg)
    result, _ = run(cfg)
    assert statuses(result) == {"a": "cached"}

    monkeypatch.setenv("DP_TEST_SECRET", "other")
    result, _ = run(cfg)
    assert result.outcomes["a"].reason == "env changed: DP_TEST_SECRET"
    state = (tmp_path / ".pipeline" / "state.json").read_text()
    assert "s3cr3t" not in state and "other" not in state

    monkeypatch.delenv("DP_TEST_SECRET")
    result, _ = run(cfg)
    assert result.outcomes["a"].reason == "env changed: DP_TEST_SECRET"
    assert state_of(tmp_path)["steps"]["a"]["env"]["DP_TEST_SECRET"] == "unset"


def test_empty_env_var_is_not_the_same_as_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("DP_TEST_SECRET", raising=False)
    cfg = make(tmp_path, PINNED)
    run(cfg)
    monkeypatch.setenv("DP_TEST_SECRET", "")
    result, _ = run(cfg)
    assert result.outcomes["a"].reason == "env changed: DP_TEST_SECRET"


def test_undeclared_env_is_ignored(tmp_path, monkeypatch):
    cfg = make(tmp_path, PINNED)
    monkeypatch.setenv("DP_TEST_OTHER", "1")
    run(cfg)
    monkeypatch.setenv("DP_TEST_OTHER", "2")
    result, _ = run(cfg)
    assert statuses(result) == {"a": "cached"}


@pytest.mark.parametrize("key", ["packages", "env"])
def test_old_checkpoint_without_packages_or_env_is_backfilled(tmp_path, monkeypatch, key):
    monkeypatch.setenv("DP_TEST_SECRET", "s3cr3t")
    cfg = make(tmp_path, PINNED)
    run(cfg)
    path = tmp_path / ".pipeline" / "state.json"
    state = json.loads(path.read_text())
    expected = state["steps"]["a"].pop(key)
    path.write_text(json.dumps(state))

    result, _ = run(cfg)
    assert statuses(result) == {"a": "cached"}
    assert state_of(tmp_path)["steps"]["a"][key] == expected
