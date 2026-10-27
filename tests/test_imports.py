"""Local modules a step imports are part of its fingerprint, without ``watch``."""

import json
import textwrap
from pathlib import Path

from test_pipeline import make, run, statuses

STEP = """
[[steps]]
name = "s"
function = "{function}"
outputs = {{ o = "o.txt" }}
"""

WRITE = "\ndef main(ctx):\n    ctx.outputs['o'].write_text('x')\n"


def setup(tmp_path: Path, files: dict[str, str], function: str = "fns2:main") -> Path:
    cfg = make(tmp_path, STEP.format(function=function))
    for rel, text in files.items():
        f = tmp_path / "steps" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(text))
    return cfg


def edit(tmp_path: Path, rel: str) -> None:
    f = tmp_path / "steps" / rel
    f.write_text(f.read_text() + "\n# edit\n")


def recorded_imports(tmp_path: Path) -> dict:
    return json.loads((tmp_path / ".pipeline/state.json").read_text())["steps"]["s"]["imports"]


def test_helper_edit_reruns_step_without_watch(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import helper" + WRITE, "helper.py": "X = 1\n"})
    assert run(cfg)[0].ok
    assert statuses(run(cfg)[0]) == {"s": "cached"}
    edit(tmp_path, "helper.py")
    r, _ = run(cfg)
    assert r.outcomes["s"].status == "ran"
    assert r.outcomes["s"].reason == "code changed: steps/helper.py"


def test_transitive_helper(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import b" + WRITE, "b.py": "import c\n", "c.py": "C = 1\n"})
    run(cfg)
    edit(tmp_path, "c.py")
    r, _ = run(cfg)
    assert "c.py" in r.outcomes["s"].reason


def test_from_import_of_submodule_and_relative_import(tmp_path):
    cfg = setup(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/util.py": "U = 1\n",
        "pkg/mod.py": "from . import util" + WRITE,
        "fns2.py": "from pkg import mod\n",
    }, function="pkg.mod:main")
    assert run(cfg)[0].ok
    assert set(recorded_imports(tmp_path)) == {"steps/pkg/__init__.py", "steps/pkg/util.py"}
    edit(tmp_path, "pkg/util.py")
    r, _ = run(cfg)
    assert r.outcomes["s"].reason == "code changed: steps/pkg/util.py"


def test_import_inside_function_counts(tmp_path):
    cfg = setup(tmp_path, {
        "fns2.py": "def main(ctx):\n    import helper\n    ctx.outputs['o'].write_text('x')\n",
        "helper.py": "X = 1\n",
    })
    run(cfg)
    edit(tmp_path, "helper.py")
    assert run(cfg)[0].outcomes["s"].status == "ran"


def test_unrelated_local_file_keeps_cache(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import helper" + WRITE, "helper.py": "", "other.py": ""})
    run(cfg)
    edit(tmp_path, "other.py")
    assert statuses(run(cfg)[0]) == {"s": "cached"}


def test_installed_and_stdlib_imports_ignored(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import json\nfrom packaging import version" + WRITE})
    assert run(cfg)[0].ok
    assert recorded_imports(tmp_path) == {}


def test_package_shadows_same_named_module(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import helper" + WRITE, "helper.py": "", "helper/__init__.py": ""})
    run(cfg)
    assert set(recorded_imports(tmp_path)) == {"steps/helper/__init__.py"}


def test_import_cycle_terminates(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import a" + WRITE, "a.py": "import b\n", "b.py": "import a\n"})
    assert run(cfg)[0].ok
    assert set(recorded_imports(tmp_path)) == {"steps/a.py", "steps/b.py"}


def test_old_checkpoint_without_imports_is_backfilled(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import helper" + WRITE, "helper.py": ""})
    run(cfg)
    state_file = tmp_path / ".pipeline/state.json"
    state = json.loads(state_file.read_text())
    del state["steps"]["s"]["imports"]
    state_file.write_text(json.dumps(state))
    assert statuses(run(cfg)[0]) == {"s": "cached"}
    assert set(recorded_imports(tmp_path)) == {"steps/helper.py"}


def test_syntax_error_in_helper_does_not_crash_fingerprinting(tmp_path):
    cfg = setup(tmp_path, {"fns2.py": "import helper" + WRITE, "helper.py": "def (:\n"})
    r, out = run(cfg)
    assert r.outcomes["s"].status == "failed"
    assert "SyntaxError" in out
