"""The orchestrator: a thin, sequential loop over steps.

For each step it (1) verifies code, packages, inputs and upstream steps, (2) skips the
step if its checkpoint is still valid, otherwise (3) runs the step function in a fresh
worker subprocess and (4) checks and records the outputs. It never imports step code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from importlib import metadata, util
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from . import checkpoint
from .config import Pipeline, Step
from .report import Reporter

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass
class Outcome:
    status: str  # ran | cached | warned | failed | skipped | would run
    detail: str = ""
    duration: float = 0.0
    output: str = ""
    reason: str = ""  # why the step ran


@dataclass
class RunResult:
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    halted_by: str | None = None
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.halted_by is None


# -- verification -----------------------------------------------------------------------

def _check_requirement(spec: str) -> str | None:
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return f"invalid requirement '{spec}'"
    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        # allow import names that differ from the distribution name, e.g. "sklearn"
        if not req.specifier and "." not in req.name and util.find_spec(req.name):
            return None
        return f"package not installed: {spec}"
    if req.specifier and not req.specifier.contains(version, prereleases=True):
        return f"package version mismatch: {req.name} {version} does not satisfy '{req.specifier}'"
    return None


def preflight(pipeline: Pipeline) -> dict[str, list[str]]:
    """Static checks that can run before anything executes: code, packages, source files."""
    produced = {pipeline.resolve(r) for s in pipeline.steps for r in s.outputs.values()}
    issues: dict[str, list[str]] = {}
    for step in pipeline.steps:
        found = []
        if checkpoint.module_file(pipeline, step.module) is None:
            found.append(f"module not found: {step.module} (searched {', '.join(map(str, pipeline.python_path))})")
        found += [e for r in step.requires if (e := _check_requirement(r))]
        for key, rel in step.inputs.items():
            p = pipeline.resolve(rel)
            if p not in produced and not p.exists():
                found.append(f"input '{key}' not found: {rel}")
        for rel in step.watch:
            if not pipeline.resolve(rel).exists():
                found.append(f"watched file not found: {rel}")
        if found:
            issues[step.name] = found
    return issues


def _missing_inputs(pipeline: Pipeline, step: Step) -> list[str]:
    return [f"input '{k}' not found: {rel}" for k, rel in step.inputs.items()
            if not pipeline.resolve(rel).exists()]


# -- execution ----------------------------------------------------------------------------

def _run_worker(pipeline: Pipeline, step: Step) -> tuple[bool, str, str]:
    """Run one step in a fresh interpreter. Returns (ok, summary-or-error, captured output)."""
    for rel in step.outputs.values():
        pipeline.resolve(rel).parent.mkdir(parents=True, exist_ok=True)

    tmp_root = pipeline.state_dir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix=f"{step.name}-") as tmp:
        spec_path, result_path = Path(tmp, "spec.json"), Path(tmp, "result.json")
        spec_path.write_text(json.dumps({
            "name": step.name,
            "module": step.module,
            "attr": step.attr,
            "inputs": {k: str(pipeline.resolve(v)) for k, v in step.inputs.items()},
            "outputs": {k: str(pipeline.resolve(v)) for k, v in step.outputs.items()},
            "params": step.params,
            "workdir": str(pipeline.workdir),
            "python_path": [str(p) for p in pipeline.python_path],
        }))
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [_PKG_ROOT, env.get("PYTHONPATH")]))
        cmd = [sys.executable, "-m", "data_pipeline.worker", str(spec_path), str(result_path)]
        try:
            proc = subprocess.run(
                cmd, cwd=pipeline.workdir, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=step.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.output.decode(errors="replace") if exc.output else ""
            return False, f"timed out after {step.timeout:g}s", out

        output = proc.stdout.decode(errors="replace")
        if proc.returncode != 0:
            how = (f"killed by signal {-proc.returncode}" if proc.returncode < 0
                   else f"exit code {proc.returncode}")
            return False, how, output
        try:
            summary = json.loads(result_path.read_text()).get("summary", "")
        except (FileNotFoundError, ValueError):
            return False, "worker exited without reporting a result", output
        return True, summary, output


class Orchestrator:
    def __init__(self, pipeline: Pipeline, reporter: Reporter | None = None):
        self.pipeline = pipeline
        self.report = reporter or Reporter()
        self.state = checkpoint.State(pipeline.state_dir)

    def run(self, only: list[str] | None = None, force: bool = False,
            rerun: list[str] | None = None, dry_run: bool = False) -> RunResult:
        """Run the pipeline. ``force`` ignores all checkpoints, ``rerun`` only the named."""
        p = self.pipeline
        steps = [s for s in p.steps if not only or s.name in only]
        forced = set(rerun or [])
        result = RunResult()
        start = time.perf_counter()

        self.report.header(p, len(steps), dry_run)
        issues = preflight(p)
        blocking = {n: e for n, e in issues.items()
                    if p.step(n).critical and any(s.name == n for s in steps)}
        if blocking and not dry_run:
            self.report.preflight_failed(blocking)
            result.halted_by = next(iter(blocking))
            return result

        for idx, step in enumerate(steps, 1):
            self.report.begin(idx, len(steps), step)
            outcome = self._step(step, issues.get(step.name, []), result,
                                 force or step.name in forced, dry_run)
            result.outcomes[step.name] = outcome
            self.report.end(idx, len(steps), step, outcome)
            if outcome.status == "failed":
                result.halted_by = step.name
                break

        result.elapsed = time.perf_counter() - start
        self.report.footer(result, dry_run)
        return result

    def _step(self, step: Step, issues: list[str], result: RunResult,
              forced: bool, dry_run: bool) -> Outcome:
        p = self.pipeline
        bad = "warned" if not step.critical else "failed"

        # 1. verify: upstream steps, then static issues, then inputs present on disk
        broken = [u for u in p.upstream[step.name]
                  if (o := result.outcomes.get(u)) and o.status in ("warned", "skipped", "failed")]
        if broken:
            status = "skipped" if not step.critical else "failed"
            return Outcome(status, f"upstream step(s) did not complete: {', '.join(broken)}")
        pending = [u for u in p.upstream[step.name]
                   if (o := result.outcomes.get(u)) and o.status == "would run"]
        if dry_run and pending:
            return Outcome("would run", f"after {', '.join(pending)}")
        problems = issues or _missing_inputs(p, step)
        if problems:
            return Outcome(bad, "verification failed", output="\n".join(problems))

        # 2. checkpoint
        fp = checkpoint.fingerprint(p, step, self.state)
        reason = "forced" if forced else checkpoint.stale_reason(p, step, self.state, fp)
        if reason is None:
            self.state.save()  # persist refreshed file-hash cache
            return Outcome("cached", self.state.steps[step.name].get("summary", ""))
        if dry_run:
            return Outcome("would run", reason)

        # 3. execute in an isolated worker
        self.state.forget(step.name)
        t0 = time.perf_counter()
        ok, info, output = _run_worker(p, step)
        took = time.perf_counter() - t0
        if not ok:
            return Outcome(bad, info, took, output)

        # 4. verify and record outputs
        hashes = {k: self.state.hash_path(p.resolve(rel)) for k, rel in step.outputs.items()}
        missing = [f"{k} ({step.outputs[k]})" for k, h in hashes.items() if h is None]
        if missing:
            return Outcome(bad, f"step did not write output(s): {', '.join(missing)}", took, output)
        # inputs may have been read lazily; re-fingerprint so a mid-run edit is not masked
        fp_after = checkpoint.fingerprint(p, step, self.state)
        if fp_after == fp:
            self.state.record(step.name, fp, hashes, summary=info, duration=round(took, 3),
                              finished=time.time())
        return Outcome("ran", info, took, output, reason)
