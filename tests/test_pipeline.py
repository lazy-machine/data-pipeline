import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
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

def ignore_interrupts_then_hang(ctx):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("started", flush=True)
    time.sleep(60)

def finish_quietly(ctx):
    time.sleep(0.5)
    ctx.outputs["dst"].write_text("finished")

def write_files(ctx):
    ctx.outputs["d"].mkdir(exist_ok=True)
    for name in ctx.params["names"]:
        (ctx.outputs["d"] / name).write_text(name)

def link_data(ctx):
    (ctx.workdir / "data").symlink_to(ctx.params["target"], target_is_directory=True)
    ctx.outputs["o"].write_text("linked")

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


def test_only_warnings_are_counted_in_the_footer(tmp_path):
    cfg = make(tmp_path, GATED)
    run(cfg)
    (tmp_path / "out" / "report.txt").unlink()
    _, out = run(cfg, only=["train"])
    assert "1 warning(s)" in out


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
    try:
        # a killed child stays a zombie until it is reaped
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):   # reaped between the two looks
        return not Path("/proc/self").exists()  # unless there is no /proc: os.kill said alive


def wait_until(condition, seconds: float = 30) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


AFTER_SIGNAL = 10  # the watch polls twice a second; starting up is what can take long


def default_signals(*keep_ignored: int) -> Callable[[], None]:
    """For preexec_fn. A test run started as a shell background job ignores these, and that
    would carry over to the runner, which is right for the runner but not for these tests."""
    def reset() -> None:
        for sig in (signal.SIGINT, signal.SIGQUIT):
            signal.signal(sig, signal.SIG_DFL)
        for sig in keep_ignored:
            signal.signal(sig, signal.SIG_IGN)
    return reset


@pytest.mark.skipif(not Path("/proc/self").exists(), reason="needs /proc")
@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_alive_survives_a_process_reaped_between_its_two_looks(monkeypatch, error):
    def reaped(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", reaped)
    assert alive(os.getpid()) is False


@needs_groups
@pytest.mark.parametrize("sig, code", [(signal.SIGINT, 130), (signal.SIGQUIT, -signal.SIGQUIT)])
def test_a_test_run_that_ignores_signals_can_still_reach_the_runner(tmp_path, sig, code):
    """As in a shell background job: without the reset this hangs until the timeout."""
    cfg = make(tmp_path, SPAWNER)
    previous = signal.signal(sig, signal.SIG_IGN)
    try:
        proc = start_runner(tmp_path, cfg)
        proc.send_signal(sig)
        proc.communicate(timeout=15)
    finally:
        signal.signal(sig, previous)
        proc.kill()
    assert proc.returncode == code


@needs_groups
def test_timeout_kills_the_steps_children(tmp_path):
    cfg = make(tmp_path, SPAWNER + "timeout = 3\n")
    result, _ = run(cfg)
    assert "timed out" in result.outcomes["spawn"].detail
    child = int((tmp_path / "child.pid").read_text())
    assert wait_until(lambda: not alive(child), AFTER_SIGNAL)


@needs_groups
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interrupting_the_runner_stops_the_step_and_keeps_its_traceback(tmp_path, sig):
    cfg = make(tmp_path, SPAWNER)
    proc = subprocess.Popen([sys.executable, "-m", "data_pipeline.cli", str(cfg), "--no-color"],
                            stdout=subprocess.PIPE, text=True, preexec_fn=default_signals())
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
    assert wait_until(lambda: not alive(child), AFTER_SIGNAL)
    assert "KeyboardInterrupt" in log.read_text()


def start_runner(tmp_path: Path, cfg: Path, *keep_ignored: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "data_pipeline.cli", str(cfg), "--no-color"],
                            stdout=subprocess.PIPE, text=True,
                            preexec_fn=default_signals(*keep_ignored))
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
    proc = start_runner(tmp_path, cfg, sig)
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
    assert wait_until(lambda: not alive(child), AFTER_SIGNAL)


@needs_groups
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_step_does_not_outlive_a_python_caller(tmp_path, sig):
    """Only the CLI turns these signals into an interrupt; a Python caller just dies."""
    cfg = make(tmp_path, SPAWNER)
    code = ("import sys; from data_pipeline import config; "
            "from data_pipeline.orchestrator import Orchestrator; "
            "Orchestrator(config.load(sys.argv[1])).run()")
    proc = subprocess.Popen([sys.executable, "-c", code, str(cfg)], stdout=subprocess.DEVNULL,
                            preexec_fn=default_signals())
    pid_file = tmp_path / "child.pid"
    try:
        assert wait_until(lambda: pid_file.exists() and pid_file.read_text())
        child = int(pid_file.read_text())
        proc.send_signal(sig)
        proc.wait(timeout=15)
    finally:
        proc.kill()
    assert wait_until(lambda: not alive(child), AFTER_SIGNAL)


@needs_groups
def test_a_step_that_ignores_interrupts_is_killed_after_the_grace(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "spawn"
function = "fns:ignore_interrupts_then_hang"
outputs = { o = "out/o.txt" }
""")
    proc = start_runner(tmp_path, cfg)
    started = time.monotonic()
    try:
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=20)
    finally:
        proc.kill()
    assert proc.returncode == 130
    assert 1.5 < time.monotonic() - started < 15


@needs_groups
def test_any_exception_in_the_runner_stops_the_step(tmp_path):
    """A caller's SIGTERM handler that raises, then carries on, must not leave the step."""
    cfg = make(tmp_path, SPAWNER)
    code = textwrap.dedent("""
        import signal, sys, time
        from data_pipeline import config
        from data_pipeline.orchestrator import Orchestrator

        def stop(*_):
            raise RuntimeError("caller gave up")

        signal.signal(signal.SIGTERM, stop)
        try:
            Orchestrator(config.load(sys.argv[1])).run()
        except RuntimeError:
            time.sleep(30)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code, str(cfg)], stdout=subprocess.DEVNULL)
    pid_file = tmp_path / "child.pid"
    try:
        assert wait_until(lambda: pid_file.exists() and pid_file.read_text())
        child = int(pid_file.read_text())
        proc.send_signal(signal.SIGTERM)
        assert wait_until(lambda: not alive(child), AFTER_SIGNAL)
        assert proc.poll() is None   # the caller is still going
    finally:
        proc.kill()


@needs_groups
def test_a_runner_that_dies_while_the_worker_starts_stops_the_step(tmp_path):
    """The worker is still starting when the runner goes, so the step must never run."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(textwrap.dedent("""
        import sys, time
        if "data_pipeline.worker" in " ".join(sys.orig_argv):
            time.sleep(1.5)
    """))
    # quiet, because printing to a dead runner would fail the step whatever the watch did
    cfg = make(tmp_path, """
[[steps]]
name = "spawn"
function = "fns:finish_quietly"
outputs = { dst = "out/o.txt" }
""")
    (tmp_path / "out").mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(site), os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen([sys.executable, "-m", "data_pipeline.cli", str(cfg), "--no-color"],
                            stdout=subprocess.DEVNULL, env=env, preexec_fn=default_signals())
    log = tmp_path / ".pipeline" / "logs" / "spawn.log"
    try:
        assert wait_until(log.exists)   # the runner has opened the log: the worker is starting
        proc.kill()
        proc.wait(timeout=15)
    finally:
        proc.kill()
    time.sleep(4)   # long enough for the step to start and finish if it were allowed to
    assert not (tmp_path / "out" / "o.txt").exists()


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


def test_requirement_not_installed_is_not_called_an_import():
    assert checkpoint.installed_version("dp-test-not-a-package") == "not installed"


def test_a_dotted_name_is_not_imported_to_find_it():
    assert checkpoint.installed_version("dp_test_missing.sub") == "not installed"


def test_an_import_name_with_a_version_specifier_has_no_version_to_test(tmp_path):
    """Preflight rejects these, so the record must not say the name is fine."""
    step = config.load(make(tmp_path, PINNED.replace('"pytest"', '"sqlite3>=1"'))).steps[0]
    assert checkpoint._packages(step)["sqlite3"] == "not installed"


def test_respelling_a_requirement_is_not_a_change(tmp_path):
    cfg = make(tmp_path, PINNED)
    run(cfg)
    cfg.write_text(cfg.read_text().replace('"packaging>=20"', '"Packaging>=20"'))
    result, _ = run(cfg)
    assert statuses(result) == {"a": "cached"}


def test_invalid_requirement_does_not_break_fingerprinting(tmp_path):
    """--only fingerprints steps it leaves out, and preflight reports them only if selected."""
    cfg = make(tmp_path, """
[[steps]]
name = "broken"
function = "fns:copy"
inputs = { src = "in.txt" }
outputs = { dst = "out/a.txt" }
requires = ["not a requirement!!"]
critical = false

[[steps]]
name = "next"
function = "fns:count"
inputs = { src = "out/a.txt" }
outputs = { n = "out/n.txt" }
""")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "a.txt").write_text("x")
    result, out = run(cfg, only=["next"])
    assert statuses(result) == {"next": "ran"}
    assert "--only left out 'broken'" in out


def test_env_value_that_is_not_utf8_is_hashed(tmp_path):
    def recorded(value: bytes) -> str:
        env = dict(os.environ, DP_TEST_SECRET=value.decode("utf-8", "surrogateescape"))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [
            str(Path(checkpoint.__file__).parents[1]), env.get("PYTHONPATH")]))
        cmd = [sys.executable, "-m", "data_pipeline.cli", str(cfg), "--no-color"]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stdout + res.stderr
        return state_of(tmp_path)["steps"]["a"]["env"]["DP_TEST_SECRET"]

    cfg = make(tmp_path, PINNED)
    assert recorded(b"caf\xe9") != recorded(b"caf\xe8")


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


# -- clean_outputs ------------------------------------------------------------------------

REGIONS = """
[[steps]]
name = "regions"
function = "fns:write_files"
outputs = { d = "out/regions" }
params = { names = ["a.csv", "b.csv"] }
clean_outputs = CLEAN
"""


@pytest.mark.parametrize("clean, left", [("true", ["a.csv"]), ("false", ["a.csv", "b.csv"])])
def test_clean_outputs_drops_files_a_rerun_no_longer_writes(tmp_path, clean, left):
    cfg = make(tmp_path, REGIONS.replace("CLEAN", clean))
    run(cfg)
    cfg.write_text(cfg.read_text().replace('"a.csv", "b.csv"', '"a.csv"'))
    result, _ = run(cfg)
    assert statuses(result) == {"regions": "ran"}
    assert sorted(f.name for f in (tmp_path / "out/regions").iterdir()) == left


FLAGGED = """
[[steps]]
name = "a"
function = "fns:fail_on_flag"
outputs = { dst = "out/o.txt" }
clean_outputs = CLEAN

[[steps]]
name = "b"
function = "fns:copy"
inputs = { src = "out/o.txt" }
outputs = { dst = "out/b.txt" }
"""


def test_failed_rerun_with_clean_outputs_leaves_no_old_output_to_read(tmp_path):
    cfg = make(tmp_path, FLAGGED.replace("CLEAN", "true"))
    run(cfg)
    (tmp_path / "flag").touch()
    result, _ = run(cfg, rerun=["a"])
    assert result.halted_by == "a"
    assert not (tmp_path / "out/o.txt").exists()
    result, _ = run(cfg, only=["b"])
    assert statuses(result) == {"b": "failed"}
    assert "input 'src' not found" in result.outcomes["b"].output


def test_failed_rerun_without_clean_outputs_keeps_the_old_output(tmp_path):
    cfg = make(tmp_path, FLAGGED.replace("CLEAN", "false"))
    run(cfg)
    (tmp_path / "flag").touch()
    run(cfg, rerun=["a"])
    assert (tmp_path / "out/o.txt").read_text() == "done"


def test_clean_outputs_leaves_cached_and_dry_run_steps_alone(tmp_path):
    cfg = make(tmp_path, REGIONS.replace("CLEAN", "true"))
    run(cfg)
    before = (tmp_path / "out/regions/a.csv").stat()
    result, _ = run(cfg)
    assert statuses(result) == {"regions": "cached"}
    cfg.write_text(cfg.read_text().replace('"a.csv", "b.csv"', '"a.csv"'))
    result, _ = run(cfg, dry_run=True)
    assert statuses(result) == {"regions": "would run"}
    after = (tmp_path / "out/regions/a.csv").stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert (tmp_path / "out/regions/b.csv").exists()


def test_clean_outputs_leaves_outputs_alone_when_upstream_failed(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "a"
function = "fns:fail_on_flag"
outputs = { dst = "out/a.txt" }
critical = false

[[steps]]
name = "b"
function = "fns:copy"
inputs = { src = "out/a.txt" }
outputs = { dst = "out/b.txt" }
clean_outputs = true
critical = false
""")
    run(cfg)
    (tmp_path / "flag").touch()
    result, _ = run(cfg, rerun=["a", "b"])
    assert statuses(result) == {"a": "warned", "b": "skipped"}
    assert (tmp_path / "out/b.txt").read_text() == "DONE"


def test_clean_outputs_unlinks_a_symlink_without_following_it(tmp_path):
    cfg = make(tmp_path, REGIONS.replace("CLEAN", "true"))
    kept = tmp_path / "elsewhere"
    kept.mkdir()
    (kept / "keep.txt").write_text("keep")
    (tmp_path / "out").mkdir()
    (tmp_path / "out/regions").symlink_to(kept, target_is_directory=True)
    result, _ = run(cfg)
    assert statuses(result) == {"regions": "ran"}
    assert not (tmp_path / "out/regions").is_symlink()
    assert (tmp_path / "out/regions/a.csv").read_text() == "a.csv"
    assert sorted(f.name for f in kept.iterdir()) == ["keep.txt"]


def test_turning_clean_outputs_on_does_not_rerun_the_step(tmp_path):
    cfg = make(tmp_path, REGIONS.replace("CLEAN", "false"))
    run(cfg)
    cfg.write_text(cfg.read_text().replace("clean_outputs = false", "clean_outputs = true"))
    result, _ = run(cfg)
    assert statuses(result) == {"regions": "cached"}


def test_an_output_that_cannot_be_cleaned_fails_the_step(tmp_path, monkeypatch):
    cfg = make(tmp_path, REGIONS.replace("CLEAN", "true"))
    run(cfg)

    def refuse(path, *a, **kw):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("data_pipeline.orchestrator.shutil.rmtree", refuse)
    result, _ = run(cfg, rerun=["regions"])
    assert result.halted_by == "regions"
    assert "could not clean output 'd' (out/regions)" in result.outcomes["regions"].detail
    assert (tmp_path / "out/regions/b.csv").exists()


@pytest.mark.parametrize("output, header, msg", [
    (".", "", "must name a file or directory"),
    ("out/..", "", "must name a file or directory"),
    ("../elsewhere", "", "must lie inside the workdir"),
    ("/tmp/x", "", "must lie inside the workdir"),
    ("steps", "", "it contains .*steps"),
    ("state", 'state_dir = "state/runs"', "it contains .*runs"),
    ("sql", "", "it contains .*q.sql"),
    ("pipeline.toml", "", "it contains .*pipeline.toml"),
    (".pipeline/logs", "", "inside the state folder"),
    ("linked/regions", "", "must lie inside the workdir"),
])
def test_clean_outputs_refuses_paths_that_hold_more_than_outputs(tmp_path, output, header, msg):
    cfg = make(tmp_path, f"""
[[steps]]
name = "a"
function = "fns:no_output"
outputs = {{ o = "{output}" }}
watch = WATCH
clean_outputs = true
""".replace("WATCH", '["sql/q.sql"]' if output == "sql" else "[]"), header)
    (tmp_path / "sql").mkdir()
    (tmp_path / "sql/q.sql").write_text("select 1")
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(config.ConfigError, match=msg):
        config.load(cfg)
    cfg.write_text(cfg.read_text().replace("clean_outputs = true", "clean_outputs = false"))
    config.load(cfg)


def test_clean_outputs_must_be_a_boolean(tmp_path):
    cfg = make(tmp_path, '[[steps]]\nname = "a"\nfunction = "m:f"\nclean_outputs = "yes"\n')
    with pytest.raises(config.ConfigError, match="clean_outputs must be true or false"):
        config.load(cfg)


def test_clean_outputs_rechecks_a_path_an_earlier_step_turned_into_a_symlink(tmp_path):
    (tmp_path / "proj").mkdir()
    shared = tmp_path / "shared"
    (shared / "regions").mkdir(parents=True)
    (shared / "regions" / "keep.txt").write_text("not ours")
    cfg = make(tmp_path, f"""
[[steps]]
name = "link"
function = "fns:link_data"
outputs = {{ o = "ready.txt" }}
params = {{ target = "{shared}" }}

[[steps]]
name = "regions"
function = "fns:write_files"
inputs = {{ ready = "ready.txt" }}
outputs = {{ d = "data/regions" }}
params = {{ names = ["a.csv"] }}
clean_outputs = true
""", 'workdir = "proj"')
    result, _ = run(cfg)
    assert result.halted_by == "regions"
    assert "must lie inside the workdir" in result.outcomes["regions"].detail
    assert (shared / "regions" / "keep.txt").read_text() == "not ours"


def test_clean_outputs_refuses_an_output_holding_another_steps_input(tmp_path):
    cfg = make(tmp_path, """
[[steps]]
name = "features"
function = "fns:write_files"
outputs = { d = "data" }
params = { names = ["clean.csv"] }
clean_outputs = true

[[steps]]
name = "train"
function = "fns:copy"
inputs = { src = "data/raw.csv" }
outputs = { dst = "model.txt" }
""")
    with pytest.raises(config.ConfigError, match="step 'train' reads 'data/raw.csv' inside it"):
        config.load(cfg)
    cfg.write_text(cfg.read_text().replace('{ d = "data" }', '{ d = "data", f = "data/raw.csv" }'))
    config.load(cfg)
    cfg.write_text(cfg.read_text().replace('src = "data/raw.csv"', 'src = "data"'))
    config.load(cfg)


def test_clean_outputs_refuses_an_output_holding_a_steps_source(tmp_path):
    make(tmp_path, "")
    cfg = tmp_path / "flat.toml"
    cfg.write_text(textwrap.dedent("""
        [[steps]]
        name = "a"
        function = "steps.fns:no_output"
        outputs = { d = "steps" }
        clean_outputs = true
    """))
    with pytest.raises(config.ConfigError, match="it contains .*steps"):
        config.load(cfg)


# -- trust in the file-hash cache and the state file ---------------------------------------


def _swap_keeping_size_and_mtime(path: Path, text: str) -> None:
    st = path.stat()
    assert len(text.encode()) == st.st_size
    time.sleep(0.05)  # past the coarse tick of the kernel clock, so ctime moves
    path.write_text(text)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_mtime_ns == st.st_mtime_ns


def test_hash_cache_sees_bytes_changed_under_the_same_size_and_mtime(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("safe")
    state = checkpoint.State(tmp_path / ".pipeline")
    before = state.hash_path(f)
    _swap_keeping_size_and_mtime(f, "evil")
    assert state.hash_path(f) != before


def test_hash_cache_sees_a_file_replaced_by_one_with_the_same_size_and_mtime(tmp_path):
    f, other = tmp_path / "f.txt", tmp_path / "other.txt"
    f.write_text("safe")
    other.write_text("evil")
    state = checkpoint.State(tmp_path / ".pipeline")
    before = state.hash_path(f)
    st = f.stat()
    os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns))
    other.replace(f)
    assert state.hash_path(f) != before


def test_tampered_input_and_output_with_restored_mtime_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "_RACY_NS", 0)  # keep every hash, as an old file would be
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    assert str(tmp_path / "out" / "n.txt") in state_of(tmp_path)["files"]
    _swap_keeping_size_and_mtime(tmp_path / "out" / "n.txt", "6")
    r, _ = run(cfg)
    assert r.outcomes["count"].reason == "output modified: n"

    _swap_keeping_size_and_mtime(tmp_path / "in.txt", "jello")
    r, _ = run(cfg)
    assert r.outcomes["upper"].reason == "inputs changed: src"


def test_hash_of_a_file_changed_just_before_hashing_is_not_kept(tmp_path, monkeypatch):
    """Timestamps tick coarsely, so a change in the same tick as the hash would leave them
    as they were: such a hash is used for this run only and the file is read again later."""
    f = tmp_path / "f.txt"
    f.write_text("fresh")
    state = checkpoint.State(tmp_path / ".pipeline")
    state.hash_path(f)
    state.save()
    assert str(f) not in state_of(tmp_path)["files"]

    monkeypatch.setattr(checkpoint, "_RACY_NS", 10_000_000)
    time.sleep(0.05)
    state = checkpoint.State(tmp_path / ".pipeline")
    state.hash_path(f)
    state.save()
    assert str(f) in state_of(tmp_path)["files"]


def test_old_three_field_hash_entries_are_read_again(tmp_path, monkeypatch):
    f = tmp_path / "f.txt"
    f.write_text("evil")
    st = f.stat()
    (tmp_path / ".pipeline").mkdir()
    (tmp_path / ".pipeline" / "state.json").write_text(json.dumps(
        {"version": 1, "steps": {}, "files": {str(f): [st.st_size, st.st_mtime_ns, "0" * 64]}}))
    state = checkpoint.State(tmp_path / ".pipeline")
    assert state.hash_path(f) != "0" * 64
    assert not state.problems


@pytest.mark.parametrize("header, msg", [
    ("workdir = 42", r"\[pipeline\].workdir must be a string"),
    ("state_dir = []", r"\[pipeline\].state_dir must be a string"),
    ('workdir = "a\\u0000b"', r"\[pipeline\].workdir contains a NUL character"),
    ('state_dir = "s\\u0000"', r"\[pipeline\].state_dir contains a NUL character"),
])
def test_pipeline_metadata_types_are_validated(tmp_path, header, msg):
    cfg = make(tmp_path, CHAIN, header)
    with pytest.raises(config.ConfigError, match=msg):
        config.load(cfg)


@pytest.mark.parametrize("body, msg", [
    ('[pipeline]\nname = 1\n', r"\[pipeline\].name must be a string"),
    ('[pipeline]\npython_path = ["a\\u0000"]\n', r"python_path: a path contains a NUL"),
])
def test_pipeline_name_and_python_path_are_validated(tmp_path, body, msg):
    cfg = tmp_path / "p.toml"
    cfg.write_text(body + '[[steps]]\nname = "a"\nfunction = "m:f"\n')
    with pytest.raises(config.ConfigError, match=msg):
        config.load(cfg)


@pytest.mark.parametrize("steps", [
    '[[steps]]\nname = "a"\nfunction = "m:f"\nwatch = ["x\\u0000"]\n',
    '[[steps]]\nname = "a"\nfunction = "m:f"\ninputs = { i = "x\\u0000" }\n',
])
def test_step_paths_with_a_nul_character_are_rejected(tmp_path, steps):
    with pytest.raises(config.ConfigError, match="NUL|non-empty path string"):
        config.load(make(tmp_path, steps))


def test_cli_reports_invalid_pipeline_metadata_as_a_config_error(tmp_path, capsys):
    cfg = make(tmp_path, CHAIN, "workdir = 42")
    assert main([str(cfg), "--no-color"]) == 2
    assert "config error: [pipeline].workdir must be a string" in capsys.readouterr().out


@pytest.mark.parametrize("content, warning", [
    ("[]", "not a version 1 state file"),
    ('"state"', "not a version 1 state file"),
    ('{"version": 99, "steps": {}}', "not a version 1 state file"),
    ("{not json", "state.json is unreadable"),
    ('{"version": 1, "steps": []}', "'steps' is not a table"),
    ('{"version": 1, "steps": {"upper": [], "count": []}}', "checkpoint of 'upper' is malformed"),
])
def test_unusable_state_file_reruns_every_step_with_a_warning(tmp_path, content, warning):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    (tmp_path / ".pipeline" / "state.json").write_text(content)
    r, out = run(cfg)
    assert r.ok and statuses(r) == {"upper": "ran", "count": "ran"}
    assert warning in out and any(warning in w for w in r.warnings)
    r, out = run(cfg)  # the run rewrote a valid state file
    assert statuses(r) == {"upper": "cached", "count": "cached"} and not r.warnings


@pytest.mark.parametrize("field, value", [
    ("code", 1), ("config", None), ("inputs", []), ("outputs", "x"), ("imports", {"a": 1}),
    ("packages", [1]), ("env", 0), ("summary", {}),
])
def test_malformed_checkpoint_reruns_only_its_step(tmp_path, field, value):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    path = tmp_path / ".pipeline" / "state.json"
    state = json.loads(path.read_text())
    state["steps"]["count"][field] = value
    path.write_text(json.dumps(state))
    r, out = run(cfg)
    assert statuses(r) == {"upper": "cached", "count": "ran"}
    assert "checkpoint of 'count' is malformed" in out


@pytest.mark.parametrize("files", ['[]', '{"x": "y"}', '{"x": [1, 2, 3, 4]}', '{"x": [1, 2, 3, true, "h"]}'])
def test_malformed_hash_cache_is_dropped_silently(tmp_path, files):
    cfg = make(tmp_path, CHAIN)
    run(cfg)
    path = tmp_path / ".pipeline" / "state.json"
    state = json.loads(path.read_text())
    state["files"] = json.loads(files)
    path.write_text(json.dumps(state))
    r, _ = run(cfg)
    assert statuses(r) == {"upper": "cached", "count": "cached"} and not r.warnings
