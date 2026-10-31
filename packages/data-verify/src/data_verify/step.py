"""The data-pipeline step (`function = "data_verify:verify"`) and the engine behind it."""

from __future__ import annotations

import fnmatch
import importlib
import json
import string
from pathlib import Path
from typing import Any

import numpy as np

from . import checks as _checks  # noqa: F401  (registers every check)
from .core import REGISTRY, Outcome, Verifier, jsonable, path_size, sha256_path
from .spec import Spec, SpecError, parse, suggest

OUTPUT_KEYS = {"report": ".json", "summary": ".md"}
STATUS_ORDER = {"fail": 0, "error": 1, "pass": 2, "skip": 3, "off": 4}


class VerificationFailed(Exception):
    """Raised by the step when a blocking check fails. The report is written first."""


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(x) for k, x in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(x) for x in obj.tolist()]
    return jsonable(obj)


def _specificity(pattern: str) -> int:
    return sum(ch not in "*?[]" for ch in pattern)


def resolve_check(cid: str, spec: Spec, default_sev: str, defaults: dict) -> tuple[str, dict]:
    """An exact id beats every glob; among globs the most specific one (most non-wildcard
    characters) wins, and on a tie the one written last."""
    sev, opt = default_sev, dict(defaults)
    globs = [(_specificity(p), i, o["severity"]) for i, (p, o) in enumerate(spec.checks.items())
             if p != cid and fnmatch.fnmatchcase(cid, p) and "severity" in o]
    if globs:
        sev = max(globs)[2]
    if cid in spec.checks:
        o = spec.checks[cid]
        sev = o.get("severity", sev)
        opt.update({k: x for k, x in o.items() if k != "severity"})
    return sev, opt


def _entry(cid, category, severity, title, out: Outcome) -> dict[str, Any]:
    return {"id": cid + (f".{out.suffix}" if out.suffix else ""), "category": category,
            "severity": severity, "status": out.status, "title": title,
            "message": out.message, "hint": out.hint, "details": _clean(out.details)}


def _template(text: str, spec: Spec, name: str) -> str:
    try:
        return string.Template(text).substitute({k: str(x) for k, x in spec.vars.items()})
    except KeyError as e:
        raise SpecError(f"assert '{name}' uses ${{{e.args[0]}}}, which is not in params.vars "
                        f"({sorted(spec.vars)})") from None
    except ValueError as e:
        raise SpecError(f"assert '{name}': {e} in {text!r}. $ starts a ${{var}} reference; "
                        "write $$ for a literal $") from None


def _run_assertion(v: Verifier, a) -> Outcome:
    n_ex = v.spec.max_examples
    if a.expr:
        expr = _template(a.expr, v.spec, a.name)
        targets = list(v.splits) if a.dataset == "*" else [a.dataset]
        total, bad, per = 0, 0, {}
        ex = []
        for name in targets:
            d = v.dataset(name)
            try:
                ok = d.eval(expr, engine="python")
            except Exception as e:  # noqa: BLE001
                hint = (" Quote column names that are not identifiers in backticks: "
                        "`my column` > 0." if isinstance(e, SyntaxError) else "")
                raise SpecError(f"assert '{a.name}': cannot evaluate {expr!r} on '{name}': "
                                f"{type(e).__name__}: {e}.{hint}") from None
            if np.ndim(ok) == 0:
                ok = np.full(len(d), bool(ok))
            ok = np.asarray(ok, dtype=object)
            viol = ~np.array([x is True or x is np.True_ for x in ok])
            total += len(d)
            bad += int(viol.sum())
            per[name] = int(viol.sum())
            if viol.any() and len(ex) < n_ex:
                ex += d.loc[viol].head(n_ex - len(ex)).to_dict("records")
        details = {"violations": per, "rows": total, "examples": ex}
        if bad:
            return Outcome("fail", f"{bad} of {total} row(s) violate {expr!r}", details,
                           a.hint or a.description)
        return Outcome("pass", f"{total} row(s) satisfy {expr!r}", {"rows": total})
    if a.sql:
        q = _template(a.sql, v.spec, a.name)
        rows = v.sql(q)
        if len(rows):
            return Outcome("fail", f"query returned {len(rows)} violating row(s)",
                           {"violations": len(rows), "examples": rows.head(n_ex).to_dict("records"),
                            "sql": q}, a.hint or a.description)
        return Outcome("pass", "query returned no violating rows")
    q = _template(a.sql_count, v.spec, a.name)
    res = v.sql(q)
    if res.shape[0] != 1 or res.shape[1] < 1:
        raise SpecError(f"assert '{a.name}': sql_count must return one row with one number")
    n = res.iloc[0, 0]
    n = 0 if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
    if n:
        return Outcome("fail", f"{n} violation(s)", {"violations": n, "sql": q},
                       a.hint or a.description)
    return Outcome("pass", "0 violations")


def run(params: dict[str, Any], inputs: dict[str, Path], log=print,
        workdir: Path | None = None) -> dict[str, Any]:
    """Verify the data. Returns the report (a dict); never raises for failed checks."""
    spec = parse(params, inputs, REGISTRY)
    v = Verifier(spec, inputs, log)
    results: list[dict[str, Any]] = []

    for cid, cdef in REGISTRY.items():
        sev, opt = resolve_check(cid, spec, cdef.severity, cdef.options)
        if sev == "off":
            results.append(_entry(cid, cdef.category, "off", cdef.title,
                                  Outcome("off", "turned off in params.checks")))
            continue
        try:
            outs = cdef.fn(v, opt)
        except SpecError:
            raise
        except Exception as e:  # noqa: BLE001  a broken check must not pass silently
            outs = Outcome("error", f"check crashed: {type(e).__name__}: {e}", {},
                           "This is a data-verify bug or unexpected data. The check could not "
                           "verify anything, so it counts as failed.")
        for out in outs if isinstance(outs, list) else [outs]:
            results.append(_entry(cid, cdef.category, sev, cdef.title, out))

    for a in spec.asserts:
        cid = f"assert.{a.name}"
        if a.severity == "off":
            results.append(_entry(cid, a.category, "off", a.description or a.name,
                                  Outcome("off", "severity = off")))
            continue
        try:
            out = _run_assertion(v, a)
        except SpecError:
            raise
        except Exception as e:  # noqa: BLE001
            out = Outcome("error", f"assertion crashed: {type(e).__name__}: {e}")
        results.append(_entry(cid, a.category, a.severity, a.description or a.name, out))

    for ref in spec.custom:
        mod_name, fn_name = ref.split(":")
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
        except (ImportError, AttributeError) as e:
            raise SpecError(f"custom check {ref!r} not found: {e}. The module must be importable "
                            "(under the pipeline's python_path).") from None
        before = len(v.custom_results)
        try:
            ret = fn(v)
            for item in ret or []:
                v.add(**item)
        except SpecError:
            raise
        except Exception as e:  # noqa: BLE001
            v.add(f"{fn_name}", False, f"custom check crashed: {type(e).__name__}: {e}")
        for r in v.custom_results[before:]:
            status = "pass" if r["ok"] else "fail"
            out = Outcome(status, r["message"], r["details"], r["hint"])
            results.append(_entry(f"custom.{r['name']}", r["category"], r["severity"],
                                  r["name"], out))

    return _report(spec, v, results, inputs, workdir)


def _blocking(r: dict, fail_on: str) -> bool:
    if r["status"] not in ("fail", "error") or r["severity"] == "off" or fail_on == "never":
        return False
    return r["severity"] == "error" or fail_on == "warn"


def _rel(path: Path, workdir: Path | None) -> str:
    if workdir is not None:
        try:
            return str(path.resolve().relative_to(Path(workdir).resolve()))
        except ValueError:
            pass
    return str(path)


def _report(spec: Spec, v: Verifier, results, inputs, workdir) -> dict[str, Any]:
    for r in results:
        r["blocking"] = _blocking(r, spec.fail_on)
    counts = {"pass": 0, "fail": 0, "error": 0, "skip": 0, "off": 0}
    for r in results:
        counts[r["status"]] += 1
    warnings = [r["id"] for r in results if r["status"] in ("fail", "error") and not r["blocking"]
                and r["severity"] != "off"]
    blocking = [r["id"] for r in results if r["blocking"]]
    datasets = {}
    for name, path in sorted(inputs.items()):
        size = path_size(path) if path.exists() else 0
        entry = {"path": _rel(path, workdir), "bytes": size}
        if name in spec.pins or size <= spec.max_hash_mb * 2**20:
            entry["sha256"] = v.cached(("sha256", name), lambda p=path: sha256_path(p))
        if name in spec.split_inputs.values():
            entry["role"] = "split"
        elif name == spec.table:
            entry["role"] = "table"
        elif name == spec.source:
            entry["role"] = "source"
        elif name in spec.config_inputs:
            entry["role"] = "config"
        else:
            entry["role"] = "auxiliary"
        datasets[name] = entry
    from . import __version__
    return {
        "tool": "data-verify",
        "version": __version__,
        "passed": not blocking,
        "fail_on": spec.fail_on,
        "blocking": blocking,
        "warnings": warnings,
        "counts": counts,
        "splits": {n: {"rows": len(d), "columns": d.shape[1]} for n, d in v.splits.items()},
        "train_split": v.train_name,
        "roles": _clean({
            "label": spec.labels,
            "task": {lb: v.task(lb) for lb in spec.labels if v.has(lb, v.train_name)},
            "id": spec.id, "group": spec.group, "time": spec.time, "label_time": spec.label_time,
            "weight": spec.weight, "split_column": spec.split_column,
            "features": v.features, "features_mode": "declared" if v.features_declared else "auto",
        }),
        "datasets": datasets,
        "checks": results,
    }


# -- rendering -----------------------------------------------------------------------------

SYMBOL = {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "skip": "skip", "off": "off"}


def console_lines(report: dict) -> list[str]:
    lines = []
    for r in report["checks"]:
        if r["status"] in ("fail", "error"):
            tag = "BLOCK" if r["blocking"] else "WARN"
        else:
            tag = SYMBOL[r["status"]]
        lines.append(f"{tag:<5} {r['id']:<36} {r['message']}")
    return lines


def failure_message(report: dict, report_path: str | None) -> str:
    bad = [r for r in report["checks"] if r["blocking"]]
    lines = [f"data-verify blocked the pipeline: {len(bad)} blocking check(s) failed"]
    for r in bad:
        lines.append(f"  x {r['id']}: {r['message']}")
        if r["hint"]:
            lines.append(f"    fix: {r['hint']}")
    if report_path:
        lines.append(f"report: {report_path}  (details: data-verify explain {report_path})")
    return "\n".join(lines)


def markdown(report: dict) -> str:
    c = report["counts"]
    head = "PASSED" if report["passed"] else "BLOCKED"
    out = [f"# data-verify: {head}", "",
           f"{c['pass']} passed, {len(report['blocking'])} blocking, "
           f"{len(report['warnings'])} warning(s), {c['skip']} skipped, {c['off']} off. "
           f"Train split: `{report['train_split']}`. Splits: "
           + ", ".join(f"`{n}` {s['rows']} rows" for n, s in report["splits"].items()) + ".", ""]
    order = sorted(report["checks"], key=lambda r: (not r["blocking"],
                                                    STATUS_ORDER[r["status"]]))
    out += ["| status | check | result |", "|---|---|---|"]
    for r in order:
        tag = ("BLOCK" if r["blocking"] else "WARN") if r["status"] in ("fail", "error") \
            else r["status"]
        msg = r["message"].replace("|", "\\|").replace("\n", " ")
        out.append(f"| {tag} | `{r['id']}` | {msg} |")
    fixes = [r for r in order if r["status"] in ("fail", "error") and r["hint"]]
    if fixes:
        out += ["", "## How to fix", ""]
        out += [f"- `{r['id']}`: {r['hint']}" for r in fixes]
    return "\n".join(out) + "\n"


def write_outputs(report: dict, outputs: dict[str, Path]) -> None:
    if "report" in outputs:
        outputs["report"].parent.mkdir(parents=True, exist_ok=True)
        outputs["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if "summary" in outputs:
        outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
        outputs["summary"].write_text(markdown(report))


def prepare_outputs(outputs: dict[str, Path]) -> None:
    """Reject unknown output names, then delete the previous run's outputs so a failed run
    never leaves an old verdict behind."""
    unknown = sorted(set(outputs) - set(OUTPUT_KEYS))
    if unknown:
        raise SpecError("unknown output(s) " + ", ".join(
            f"'{k}'{suggest(k, OUTPUT_KEYS)}" for k in unknown)
            + ": data-verify writes 'report' (JSON) and 'summary' (Markdown)")
    for p in outputs.values():
        p.unlink(missing_ok=True)


def verify(ctx) -> dict[str, Any]:
    """data-pipeline step: verify the step's inputs, write `report` (JSON) and `summary`
    (Markdown) if declared, and raise VerificationFailed when a blocking check fails."""
    outputs = {k: Path(p) for k, p in ctx.outputs.items()}
    prepare_outputs(outputs)
    log = getattr(ctx, "log", print)
    inputs = {k: Path(p) for k, p in ctx.inputs.items()}
    report = run(dict(ctx.params), inputs, log, getattr(ctx, "workdir", None))
    write_outputs(report, outputs)
    for line in console_lines(report):   # failures and warnings; the report has the rest
        if line.startswith(("BLOCK", "WARN")):
            log(line)
    c = report["counts"]
    log(f"{c['pass']} passed, {len(report['blocking'])} blocking, {len(report['warnings'])} "
        f"warning(s), {c['skip']} skipped, {c['off']} off")
    if not report["passed"]:
        where = _rel(outputs["report"], getattr(ctx, "workdir", None)) if "report" in outputs \
            else None
        raise VerificationFailed(failure_message(report, where))
    return {"checks": c["pass"] + c["fail"] + c["error"], "passed": c["pass"],
            "warnings": len(report["warnings"]), "skipped": c["skip"]}
