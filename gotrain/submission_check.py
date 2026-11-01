"""Read-only submission readiness checklist.

Reads saved state only: experiment records, best pointers, normalization files,
registry index, agent sessions, review outputs, the frozen manifest and existing
final-evaluation files. It never trains, downloads, calls an LLM, freezes,
promotes, or opens a dataset. Exit codes: 0 = no blocking problems (either ready to
begin final evaluation, or final scores complete), 1 = blocking problems found,
2 = usage error (bad home path or arguments).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gotrain.agent.session import list_sessions
from gotrain.artifacts.store import FilesystemArtifactStore
from gotrain.data.normalization import load_stats
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import ensure_within, iter_jsonl, read_json, sha256_file
from gotrain.util.timing import utc_now_iso

REQUIRED_TASKS = ("m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake")
OK, WARN, FAIL, PENDING, MANUAL = "ok", "warn", "fail", "pending", "manual"


def _item(name: str, status: str, detail: str, **extra: Any) -> dict:
    return {"check": name, "status": status, "detail": detail, **extra}


def check_task_selection(home: Home, store: ExperimentStore, registry: FilesystemArtifactStore, task: str) -> list[dict]:
    items = []
    best = store.best(task)
    if best is None:
        items.append(_item(f"{task}: selected checkpoint", FAIL, "no best pointer; run baseline/agent and select"))
        return items
    try:
        rec = store.read(best["experiment_id"])
    except (FileNotFoundError, ValueError) as exc:
        items.append(_item(f"{task}: selected checkpoint", FAIL, f"pointer {best['experiment_id']} unreadable: {exc}"))
        return items
    problems = []
    if rec.get("status") != "completed":
        problems.append(f"status {rec.get('status')}")
    if rec.get("screening") or rec.get("partial_validation"):
        problems.append("screening or partial-validation run")
    ck_rel = (rec.get("checkpoint") or {}).get("best_path")
    sha_rec = (rec.get("checkpoint") or {}).get("best_sha256")
    actual = None
    if not ck_rel:
        problems.append("no checkpoint recorded")
    else:
        try:
            ck = ensure_within(home.root, home.root / ck_rel)
            if not ck.exists():
                problems.append(f"checkpoint missing at {ck_rel}")
            else:
                actual = sha256_file(ck)
                if actual != sha_rec:
                    problems.append("checkpoint hash on disk differs from the record")
        except ValueError as exc:
            problems.append(str(exc))
    detail = f"{best['experiment_id']} selected by {best.get('selected_by')}, validation {rec.get('metric', {}).get('name')} {rec.get('result', {}).get('best_val_metric')}"
    if best.get("selected_by") == "manual":
        detail += " (manual selection; keep this provenance in the write-up)"
    items.append(_item(f"{task}: selected checkpoint", FAIL if problems else OK, detail + ("; " + "; ".join(problems) if problems else ""), experiment_id=best["experiment_id"], selected_by=best.get("selected_by"), checkpoint_sha256=actual))

    norm = rec.get("normalization") or {}
    npath = Path(norm.get("path", ""))
    if not npath.is_absolute() or not npath.exists():
        alt = home.normalization / f"{task}.json"
        npath = alt if alt.exists() else npath
    if rec.get("source") == "synthetic":
        from gotrain.artifacts.verify import synthetic_train_stats

        ok = synthetic_train_stats(rec).get("stats_hash") == norm.get("stats_hash")
        items.append(_item(f"{task}: normalization", OK if ok else FAIL, "synthetic experiment: statistics refit from the seeded synthetic train split " + ("match" if ok else "do not match") + " the record"))
    elif not npath.exists():
        items.append(_item(f"{task}: normalization", FAIL, f"file not found ({norm.get('path')})"))
    else:
        try:
            stats = load_stats(npath)
            if stats.get("stats_hash") != norm.get("stats_hash"):
                items.append(_item(f"{task}: normalization", FAIL, "hash differs from the experiment record"))
            else:
                items.append(_item(f"{task}: normalization", OK, f"{npath.name} present, hash matches, fitted on {stats.get('n_samples_used')} train samples"))
        except ValueError as exc:
            items.append(_item(f"{task}: normalization", FAIL, str(exc)))

    versions = [v for v in registry.list_versions(task=task) if v["experiment_id"] == best["experiment_id"] and v["checkpoint_sha256"] == sha_rec]
    if not versions:
        items.append(_item(f"{task}: model version", WARN, "no stored registry version matches the selected checkpoint (freeze stores one)"))
    else:
        v = versions[-1]
        full = registry.get_version(v["version_id"]).get("verification") or {}
        if v["verification"] is True:
            scope = "model and preprocessing" if full.get("preprocessing_verified") else "stored reference batch only"
            items.append(_item(f"{task}: model version", OK, f"{v['version_id']} verified ({scope}: reference-batch reproduction, not full accuracy validation){'; promoted' if v['promoted'] else ''}", version_id=v["version_id"]))
        elif v["verification"] is False:
            items.append(_item(f"{task}: model version", FAIL, f"{v['version_id']} verification FAILED: {full.get('error') or full.get('max_abs_diff')}", version_id=v["version_id"]))
        else:
            items.append(_item(f"{task}: model version", WARN, f"{v['version_id']} stored but not verified (python -m gotrain verify-model {v['version_id']})", version_id=v["version_id"]))
    return items


def check_sessions(home: Home, tasks: list[str]) -> list[dict]:
    sessions = list_sessions(home)
    if not sessions:
        return [_item("agent sessions", WARN, "no agent session recorded under agent/sessions")]
    items = []
    finished_anywhere: set[str] = set()
    for s in sessions:
        requested = list(s.get("tasks", []))
        finished = list((s.get("finished_tasks") or {}).keys())
        finished_anywhere |= set(finished)
        unfinished = [t for t in requested if t not in finished]
        status_text = s.get("status")
        detail = f"status '{status_text}', requested {requested}, finished {finished}, launched {s.get('experiments_launched')} experiments, model {s.get('model_id') or 'n/a'}"
        if unfinished:
            detail += f"; unfinished {unfinished}" + (" (status text says completed but coverage is incomplete)" if status_text == "completed" else "")
        items.append(_item(f"session {s['session_id']}", WARN if unfinished else OK, detail, session_id=s["session_id"]))
    store = ExperimentStore(home)
    for t in tasks:
        best = store.best(t)
        if best and best.get("selected_by") == "manual":
            items.append(_item(f"{t}: selection provenance", WARN, "selected manually, not by an agent session; say so in the presentation"))
        elif t not in finished_anywhere:
            items.append(_item(f"{t}: selection provenance", WARN, "no session finished this task" + (f"; pointer set by {best.get('selected_by')}" if best else "")))
    return items


def check_review(home: Home, tasks: list[str]) -> list[dict]:
    root = home.root / "review"
    items = []
    found = sorted(root.glob("*/*/provenance.json")) if root.exists() else []
    if not found:
        supported = [t for t in tasks if t in ("m-pv4ger-seg", "m-nz-cattle")]
        if supported:
            commands = "; ".join(f"python -m gotrain --home {str(home.root)!r} review-build --task {t}" for t in supported)
            return [_item("review page", PENDING, "optional interactive binary-segmentation review not generated in this run home; " + commands)]
        return [_item("review page", MANUAL, "interactive review is optional and currently supports binary segmentation; no solar task is required in this run home") ]
    for prov_path in found:
        try:
            p = read_json(prov_path)
        except ValueError:
            items.append(_item("review page", FAIL, f"{prov_path} unreadable"))
            continue
        data = p.get("data", {})
        model = p.get("model", {})
        page = prov_path.parent / "review.html"
        bench = prov_path.parent / "benchmark.json"
        cov = f"{data.get('n_images_reviewed')} of {data.get('n_images_in_split')} {data.get('split')} images" + (" PARTIAL" if data.get("partial") else "")
        status = OK if (page.exists() and bench.exists() and not data.get("partial") and not p.get("synthetic")) else WARN
        detail = f"{model.get('task')} {model.get('experiment_id')} ({model.get('model_version') or 'no version'}), {cov}, page {'present' if page.exists() else 'missing'}, benchmark {'present' if bench.exists() else 'missing'}"
        if p.get("synthetic"):
            detail += "; SYNTHETIC fixture, not real results"
        items.append(_item(f"review page {prov_path.parent.name}", status, detail, path=str(prov_path.parent)))
    return items


def check_final(home: Home, tasks: list[str]) -> tuple[list[dict], dict]:
    items = []
    manifest_path = home.selection / "manifest.json"
    summary = {"frozen": manifest_path.exists(), "final_complete": False, "final_partial": False, "exposure_without_result": []}
    if manifest_path.exists():
        try:
            from gotrain.final_eval.freeze import load_manifest

            m = load_manifest(manifest_path)
            missing = [t for t in tasks if t not in m.get("tasks", {})]
            items.append(_item("frozen selection", OK if not missing else WARN, f"manifest frozen at {m.get('frozen_at')} for {list(m.get('tasks', {}))}" + (f"; missing {missing}" if missing else "")))
        except ValueError as exc:
            items.append(_item("frozen selection", FAIL, str(exc)))
    else:
        items.append(_item("frozen selection", PENDING, "not frozen yet (python -m gotrain freeze --task all); expected before final evaluation"))
    results_path = home.final_eval / "results.json"
    exposure = list(iter_jsonl(home.final_eval / "exposure.jsonl"))
    exposed_tasks = {e.get("task") for e in exposure if e.get("kind") == "test_split_opened"}
    if results_path.exists():
        try:
            res = read_json(results_path)
        except ValueError:
            items.append(_item("final test scores", FAIL, "final_eval/results.json unreadable"))
            return items, summary
        done = {t: r for t, r in res.get("tasks", {}).items() if r.get("status") == "completed"}
        failed = {t: r for t, r in res.get("tasks", {}).items() if r.get("status") != "completed"}
        missing = [t for t in tasks if t not in done]
        summary["final_complete"] = not missing and not failed
        summary["final_partial"] = bool(done) and bool(missing or failed)
        for t, r in done.items():
            items.append(_item(f"{t}: final test score", OK, f"{r.get('metric', {}).get('name')} {r.get('test_metric')} on {r.get('n_test_samples')} test samples at {r.get('evaluated_at')}"))
        for t, r in failed.items():
            items.append(_item(f"{t}: final test score", FAIL, f"attempt recorded with status {r.get('status')}: {r.get('error')}; review before any retry (automatic retry is blocked after exposure)"))
        for t in missing:
            if t not in failed:
                items.append(_item(f"{t}: final test score", PENDING, "not evaluated yet"))
    else:
        for t in tasks:
            items.append(_item(f"{t}: final test score", PENDING, "no final evaluation yet (expected before the final run)"))
    for t in sorted(exposed_tasks):
        res_tasks = read_json(results_path).get("tasks", {}) if results_path.exists() else {}
        if res_tasks.get(t, {}).get("status") != "completed":
            summary["exposure_without_result"].append(t)
            items.append(_item(f"{t}: test exposure", FAIL, "test split was opened but no completed result exists; preserve logs and review before any further test access"))
    return items, summary


def run_check(home: Home, tasks: list[str] | None = None, presentation_path: Path | None = None) -> dict:
    tasks = list(tasks or REQUIRED_TASKS)
    if not home.root.exists():
        raise FileNotFoundError(f"run home {home.root} does not exist")
    store = ExperimentStore(home)
    registry = FilesystemArtifactStore(home)
    items: list[dict] = []
    for t in tasks:
        items.extend(check_task_selection(home, store, registry, t))
    items.extend(check_sessions(home, tasks))
    items.extend(check_review(home, tasks))
    final_items, final_summary = check_final(home, tasks)
    items.extend(final_items)
    if presentation_path is not None:
        items.append(_item("presentation/demo deliverables", OK if Path(presentation_path).exists() else FAIL, f"{presentation_path} {'present' if Path(presentation_path).exists() else 'missing'}"))
    else:
        items.append(_item("presentation/demo deliverables", MANUAL, "check manually: slides, demo script (DEMO.md, docs/REVIEW_RATIONALE.md), agent logs exported"))
    blocking = [i for i in items if i["status"] == FAIL]
    warnings = [i for i in items if i["status"] == WARN]
    pending = [i for i in items if i["status"] == PENDING]
    if blocking:
        verdict = "NOT READY: blocking problems found"
    elif final_summary["final_complete"]:
        verdict = "final test scores complete for all required tasks; remaining items are manual"
    elif final_summary["final_partial"]:
        verdict = "final evaluation partially complete; finish or review it before submitting"
    else:
        verdict = "ready to begin final evaluation (freeze, then final-eval); NOT ready to submit until test scores exist"
    return {"checked_at": utc_now_iso(), "home": str(home.root), "tasks": tasks, "items": items, "counts": {"ok": sum(i["status"] == OK for i in items), "warn": len(warnings), "fail": len(blocking), "pending": len(pending), "manual": sum(i["status"] == MANUAL for i in items)}, "final": final_summary, "verdict": verdict, "exit_code": 1 if blocking else 0}


def format_report(result: dict) -> str:
    marks = {OK: "[ok]     ", WARN: "[warn]   ", FAIL: "[FAIL]   ", PENDING: "[pending]", MANUAL: "[manual] "}
    lines = [f"GoTrain submission check ({result['checked_at']}) for {result['home']}", ""]
    for it in result["items"]:
        lines.append(f"{marks[it['status']]} {it['check']}: {it['detail']}")
    c = result["counts"]
    lines.append("")
    lines.append(f"ok {c['ok']}, warnings {c['warn']}, pending {c['pending']}, manual {c['manual']}, blocking {c['fail']}")
    lines.append(f"VERDICT: {result['verdict']}")
    return "\n".join(lines)
