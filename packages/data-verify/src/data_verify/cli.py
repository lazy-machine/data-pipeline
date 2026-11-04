"""Command line: run the gate outside the orchestrator, and look up checks and params."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from . import __version__
from .reference import (
    TEMPLATE, checks_json, checks_markdown, checks_text, params_markdown, params_text,
)
from .spec import SpecError, split_input_names
from .step import console_lines, failure_message, prepare_outputs, run, write_outputs

USAGE = """\
data-verify: leakage and poisoning gate for data-pipeline projects

  data-verify run pipeline.toml [--step NAME]   run a data-verify step of a pipeline file
  data-verify run verify.toml                   run a standalone config (inputs/outputs/params)
  data-verify lint [pipeline.toml] [--step N]   validate params, and that splits pass the gate
  data-verify explain REPORT.json [--all]       show what failed and how to fix it
  data-verify checks [--json | --markdown]      every check, its severity and options
  data-verify params [--markdown]               every params key
  data-verify template                          a step block to paste into pipeline.toml

exit codes: 0 passed, 1 blocked, 2 bad config
"""


def _load_target(path: Path, step_name: str | None, need_inputs: bool = True):
    try:
        data = tomllib.loads(path.read_text())
    except OSError as e:
        raise SpecError(f"cannot read {path}: {e.strerror or e}") from None
    except tomllib.TOMLDecodeError as e:
        raise SpecError(f"{path} is not valid TOML: {e}") from None
    base = path.parent
    if "steps" in data:
        pipe = data.get("pipeline", {})
        workdir = (base / pipe.get("workdir", ".")).resolve()
        py_path = [(base / p).resolve() for p in pipe.get("python_path", ["."])]
        steps = [s for s in data["steps"] if s.get("function") == "data_verify:verify"]
        if step_name:
            named = [s for s in data["steps"] if s.get("name") == step_name]
            if not named:
                raise SpecError(f"no step named '{step_name}' in {path}")
            steps = [s for s in steps if s.get("name") == step_name]
            if not steps:
                raise SpecError(f"step '{step_name}' is not a data-verify step (function = "
                                f"{named[0].get('function')!r}, not 'data_verify:verify')")
        if len(steps) != 1:
            names = [s.get("name") for s in steps]
            raise SpecError(f"{path} has {len(steps)} data-verify steps {names}: pick one "
                            "with --step NAME")
        step = steps[0]
    else:
        if step_name:
            raise SpecError(f"--step only applies to a pipeline file; {path} is a standalone "
                            "config (no [[steps]])")
        workdir = (base / data.get("workdir", ".")).resolve()
        py_path = [(base / p).resolve() for p in data.get("python_path", ["."])]
        step = data
        unknown = set(step) - {"inputs", "outputs", "params", "workdir", "python_path", "name"}
        if unknown:
            raise SpecError(f"{path}: unknown key(s) {sorted(unknown)}; a standalone config has "
                            "inputs, outputs, params (and optional workdir, python_path)")
    inputs = {k: (workdir / p).resolve() for k, p in step.get("inputs", {}).items()}
    outputs = {k: (workdir / p).resolve() for k, p in step.get("outputs", {}).items()}
    missing = [f"{k}: {p}" for k, p in inputs.items() if not p.exists()]
    if missing and need_inputs:
        raise SpecError("input(s) not found: " + "; ".join(missing))
    return step.get("params", {}), inputs, outputs, workdir, py_path


def cmd_run(args) -> int:
    params, inputs, outputs, workdir, py_path = _load_target(Path(args.config), args.step)
    for p in reversed(py_path):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    prepare_outputs(outputs)
    report = run(params, inputs, log=lambda *a: print(*a, file=sys.stderr), workdir=workdir)
    write_outputs(report, outputs)
    for line in console_lines(report):
        if args.all or not line.startswith(("PASS", "skip", "off")):
            print(line)
    c = report["counts"]
    print(f"{'PASSED' if report['passed'] else 'BLOCKED'}: {c['pass']} passed, "
          f"{len(report['blocking'])} blocking, {len(report['warnings'])} warning(s), "
          f"{c['skip']} skipped")
    if not report["passed"]:
        rp = outputs.get("report")
        print(failure_message(report, str(rp) if rp else None), file=sys.stderr)
        return 1
    return 0


def _overlap(a: Path, b: Path) -> bool:
    return a.is_relative_to(b) or b.is_relative_to(a)


def _ungated_steps(path: Path, gate_name: str | None, params, inputs, workdir) -> list[str]:
    """One line for each step that reads a split the gate checks but can run without it.

    Steps upstream of the gate are left out: they can't depend on it, and what they read
    isn't the data the model sees until the gate has run.
    """
    steps = tomllib.loads(path.read_text()).get("steps")
    if not steps:
        return []  # a standalone config has no other steps
    gates = [s for s in steps if s.get("function") == "data_verify:verify"]
    gate = next(s for s in gates if gate_name in (None, s.get("name")))

    def paths(step, key):
        return [(workdir / p).resolve() for p in step.get(key, {}).values()]

    def ancestors(step) -> set[str]:
        seen: set[str] = set()
        queue = [step]
        while queue:
            current = queue.pop()
            reads = paths(current, "inputs")
            for other in steps:
                name = other.get("name")
                wrote = any(_overlap(i, o) for i in reads for o in paths(other, "outputs"))
                if (wrote or name in current.get("depends_on", [])) and name not in seen:
                    seen.add(name)
                    queue.append(other)
        return seen

    split_files = [inputs[n] for n in split_input_names(params, inputs)]
    before_gate = ancestors(gate)
    found = []
    for step in steps:
        if step in gates or step.get("name") in before_gate:
            continue
        read = [f for f in split_files if any(_overlap(f, i) for i in paths(step, "inputs"))]
        if read and gate.get("name") not in ancestors(step):
            where = read[0].relative_to(workdir) if read[0].is_relative_to(workdir) else read[0]
            found.append(f"step '{step.get('name')}' reads {where} without the gate: add "
                         f"depends_on = [\"{gate.get('name')}\"]")
    return found


def cmd_lint(args) -> int:
    from .core import REGISTRY
    from .spec import check_structure, parse
    params, inputs, _, workdir, _ = _load_target(Path(args.config), args.step, need_inputs=False)
    missing = sorted(k for k, p in inputs.items() if not p.exists())
    if missing:
        check_structure(params, REGISTRY)
        print(f"params OK (structure only: input(s) {missing} not built yet, so references "
              "and column names are checked when the step runs)")
    else:
        parse(params, inputs, REGISTRY)
        print("params OK (fully resolved against the inputs)")
    ungated = _ungated_steps(Path(args.config), args.step, params, inputs, workdir)
    for line in ungated:
        print(f"x {line}")
    return 1 if ungated else 0


def cmd_explain(args) -> int:
    report = json.loads(Path(args.report).read_text())
    print(f"{'PASSED' if report['passed'] else 'BLOCKED'} (data-verify {report['version']}, "
          f"fail_on = {report['fail_on']})")
    print("splits: " + ", ".join(f"{n}={s['rows']}" for n, s in report["splits"].items())
          + f"; train split: {report['train_split']}")
    roles = report["roles"]
    print(f"label: {roles['label']}  task: {roles['task']}  group: {roles['group']}  "
          f"time: {roles['time']}  features: {len(roles['features'])} ({roles['features_mode']})")
    shown = [r for r in report["checks"]
             if args.all or r["status"] in ("fail", "error")]
    for r in shown:
        tag = ("BLOCK" if r["blocking"] else "WARN") if r["status"] in ("fail", "error") \
            else r["status"].upper()
        print(f"\n[{tag}] {r['id']} ({r['severity']}): {r['title']}")
        print(f"  {r['message']}")
        if r["hint"]:
            print(f"  fix: {r['hint']}")
        if r["details"] and r["status"] in ("fail", "error"):
            text = json.dumps(r["details"], indent=2, ensure_ascii=False)
            lines = text.splitlines()
            print("  details: " + "\n  ".join(lines[:40]) + ("\n  ..." if len(lines) > 40 else ""))
    if not shown:
        print("\nno failed checks")
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    ap = argparse.ArgumentParser(prog="data-verify", usage=USAGE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--step")
    r.add_argument("--all", action="store_true", help="also list passed and skipped checks")
    li = sub.add_parser("lint")
    li.add_argument("config", nargs="?", default="pipeline.toml")
    li.add_argument("--step")
    e = sub.add_parser("explain")
    e.add_argument("report")
    e.add_argument("--all", action="store_true")
    c = sub.add_parser("checks")
    c.add_argument("--json", action="store_true")
    c.add_argument("--markdown", action="store_true")
    p = sub.add_parser("params")
    p.add_argument("--markdown", action="store_true")
    sub.add_parser("template")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "explain":
            return cmd_explain(args)
        if args.cmd == "lint":
            return cmd_lint(args)
        if args.cmd == "checks":
            print(json.dumps(checks_json(), indent=2) if args.json else
                  checks_markdown() if args.markdown else checks_text())
        elif args.cmd == "params":
            print(params_markdown() if args.markdown else params_text())
        elif args.cmd == "template":
            print(TEMPLATE, end="")
        return 0
    except SpecError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
