"""The orchestrator: a thin, sequential loop over steps.

For each step it (1) verifies code, packages, inputs and upstream steps, (2) skips the
step if its checkpoint is still valid, otherwise (3) runs the step function in a fresh
worker subprocess and (4) checks and records the outputs. It never imports step code.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata, util
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from . import checkpoint
from .config import Pipeline, Step, output_entry, overlaps
from .report import Reporter

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
_INTERRUPT_GRACE = 2.0  # seconds a step gets to print where it was before it is killed


@dataclass
class Outcome:
    status: str  # ran | cached | warned | failed | skipped | would run
    detail: str = ""
    duration: float = 0.0
    output: str = ""
    reason: str = ""  # why the step ran
    log: Path | None = None  # where the step's output was written, if it ran


@dataclass
class RunResult:
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    halted_by: str | None = None
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)

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
            if not p.exists() and not any(overlaps(p, o) for o in produced):
                found.append(f"input '{key}' not found: {rel}")
        for rel in step.watch:
            if not pipeline.resolve(rel).exists():
                found.append(f"watched file not found: {rel}")
        if found:
            issues[step.name] = found
    return issues


def _clean_outputs(pipeline: Pipeline, step: Step) -> str | None:
    """Delete a step's declared outputs before it reruns. Returns an error, or None.

    Without this a failed rerun leaves the previous outputs looking current, and a
    directory output keeps files the new run no longer writes.
    """
    for key, rel in step.outputs.items():
        if problem := pipeline.cleaning_problem(rel):
            return f"could not clean output '{key}' ({rel}): {problem}"
        entry = output_entry(pipeline.workdir, rel)
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        except OSError as exc:
            return f"could not clean output '{key}' ({rel}): {exc}"
    return None


def _missing_inputs(pipeline: Pipeline, step: Step) -> list[str]:
    return [f"input '{k}' not found: {rel}" for k, rel in step.inputs.items()
            if not pipeline.resolve(rel).exists()]


# -- execution ----------------------------------------------------------------------------

def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the worker and whatever it started; the worker leads its own process group."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def _stop(proc: subprocess.Popen, reader: threading.Thread, forwarding: threading.Event,
          grace: float = 0.0) -> None:
    """Kill the step. With ``grace``, first interrupt it so its traceback reaches the log."""
    groups = hasattr(os, "killpg")  # POSIX only; elsewhere just the worker is killed
    try:
        if grace and groups:
            _kill_group(proc, signal.SIGINT)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if groups:
            _kill_group(proc, signal.SIGKILL)
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        forwarding.clear()


def _run_worker(pipeline: Pipeline, step: Step, log_path: Path,
                on_line: Callable[[str], None] | None = None) -> tuple[bool, str, str]:
    """Run one step in a fresh interpreter. Returns (ok, summary-or-error, captured output).

    Output is copied line by line to ``log_path`` as it arrives, so a long or stuck step
    can be followed, and nothing is lost on a timeout or Ctrl-C.
    """
    for rel in step.outputs.values():
        pipeline.resolve(rel).parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

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
            "runner": os.getpid(),
        }))
        env = dict(os.environ, PYTHONUNBUFFERED="1")  # what makes lines arrive live
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [_PKG_ROOT, env.get("PYTHONPATH")]))
        cmd = [sys.executable, "-m", "data_pipeline.worker", str(spec_path), str(result_path)]
        chunks: list[bytes] = []
        forwarding = threading.Event()  # cleared once we stop waiting for the pipe
        forwarding.set()
        fh = log_path.open("wb")  # opened here so a failure surfaces before the worker starts
        try:
            proc = subprocess.Popen(
                cmd, cwd=pipeline.workdir, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group, so children can be killed with it
            )
        except BaseException:
            fh.close()
            raise

        def pump() -> None:
            with fh, proc.stdout:
                for raw in proc.stdout:
                    fh.write(raw)
                    fh.flush()
                    if forwarding.is_set():
                        chunks.append(raw)
                        if on_line:
                            on_line(raw.decode(errors="replace").rstrip("\r\n"))

        reader = threading.Thread(target=pump, daemon=True)
        try:
            reader.start()
            proc.wait(timeout=step.timeout)
            # bounded: a grandchild that inherited stdout could keep the pipe open
            reader.join(timeout=5)
        except subprocess.TimeoutExpired:
            _stop(proc, reader, forwarding)
            return False, f"timed out after {step.timeout:g}s", b"".join(chunks).decode(errors="replace")
        except BaseException:  # a caller's signal handler may raise anything, then carry on
            _stop(proc, reader, forwarding, grace=_INTERRUPT_GRACE)
            raise
        forwarding.clear()  # late lines still reach the log, but not this step's outcome

        output = b"".join(chunks).decode(errors="replace")
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
        self.running_log: Path | None = None  # log of the step in progress, for Ctrl-C

    def run(self, only: list[str] | None = None, force: bool = False,
            rerun: list[str] | None = None, dry_run: bool = False) -> RunResult:
        """Run the pipeline. ``force`` ignores all checkpoints, ``rerun`` only the named."""
        p = self.pipeline
        steps = [s for s in p.steps if not only or s.name in only]
        forced = set(rerun or [])
        result = RunResult()
        start = time.perf_counter()

        self.report.header(p, len(steps), dry_run)
        for text in self.state.problems:
            result.warnings.append(text)
            self.report.warning(text)
        self.state.problems = []
        issues = preflight(p)
        blocking = {n: e for n, e in issues.items()
                    if p.step(n).critical and any(s.name == n for s in steps)}
        if blocking and not dry_run:
            self.report.preflight_failed(blocking)
            result.halted_by = next(iter(blocking))
            return result

        selected = {s.name for s in steps}
        flagged: set[str] = set()
        for idx, step in enumerate(steps, 1):
            if only:
                self._warn_outside(step, selected, flagged, result, dry_run)
            self.report.begin(idx, len(steps), step)
            outcome = self._step(step, issues.get(step.name, []), result,
                                 force or step.name in forced, dry_run)
            self.running_log = None
            result.outcomes[step.name] = outcome
            self.report.end(idx, len(steps), step, outcome)
            if outcome.status == "failed":
                result.halted_by = step.name
                break

        result.elapsed = time.perf_counter() - start
        self.report.footer(result, dry_run)
        return result

    def _warn_outside(self, step: Step, selected: set[str], flagged: set[str],
                      result: RunResult, dry_run: bool) -> None:
        """Say when ``--only`` leaves out an upstream step, such as a gate, that isn't current.

        Checked before each step, not once at the start: a selected step that just ran can
        be what made the step left out go stale.
        """
        p = self.pipeline
        ancestors: set[str] = set()
        queue = [step.name]
        while queue:
            new = [u for u in p.upstream[queue.pop()] if u not in ancestors]
            ancestors.update(new)
            queue += new
        for other in p.steps:
            if other.name not in ancestors or other.name in selected | flagged:
                continue
            fp = checkpoint.fingerprint(p, other, self.state)
            reason = checkpoint.stale_reason(p, other, self.state, fp)
            pending = [u for u in p.upstream[other.name]
                       if (o := result.outcomes.get(u)) and o.status == "would run"]
            if reason is None and dry_run and pending:
                reason = f"after {', '.join(pending)}"
            if reason:
                flagged.add(other.name)
                text = (f"--only left out '{other.name}', which is not up to date ({reason}): "
                        f"'{step.name}' depends on it")
                result.warnings.append(text)
                self.report.warning(text)

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
        if step.clean_outputs and (error := _clean_outputs(p, step)):
            return Outcome(bad, error)
        t0 = time.perf_counter()
        log = p.state_dir / "logs" / f"{step.name}.log"
        self.running_log = log
        ok, info, output = _run_worker(p, step, log, self.report.line if self.report.follow else None)
        took = time.perf_counter() - t0
        if not ok:
            return Outcome(bad, info, took, output, log=log)

        # 4. verify and record outputs
        hashes = {k: self.state.hash_path(p.resolve(rel)) for k, rel in step.outputs.items()}
        missing = [f"{k} ({step.outputs[k]})" for k, h in hashes.items() if h is None]
        if missing:
            return Outcome(bad, f"step did not write output(s): {', '.join(missing)}", took, output,
                           log=log)
        # inputs may have been read lazily; re-fingerprint so a mid-run edit is not masked
        fp_after = checkpoint.fingerprint(p, step, self.state)
        if fp_after == fp:
            self.state.record(step.name, fp, hashes, summary=info, duration=round(took, 3),
                              finished=time.time())
        return Outcome("ran", info, took, output, reason, log)
