"""Post-training delivery: selected models -> verification -> review -> report.

No training, selection, promotion, freeze or test-evaluation operations live here.
Heavy dependencies are lazy so the orchestration can be tested independently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

BINARY_REVIEW_TASKS = frozenset(("m-pv4ger-seg", "m-nz-cattle"))


def _protected_state(home) -> dict:
    paths = [home.selection / "manifest.json"]
    paths += list(home.final_eval.rglob("*")) if home.final_eval.exists() else []
    best = home.experiments / "best"
    paths += list(best.glob("*.json")) if best.exists() else []
    return {str(p.relative_to(home.root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


class Operations:
    def __init__(self, home, device, geobench_root, batch_size, num_workers, log):
        from gotrain.experiments.store import ExperimentStore
        from gotrain.artifacts.store import FilesystemArtifactStore
        self.home, self.device, self.geobench_root = home, device, geobench_root
        self.batch_size, self.num_workers, self.log = batch_size, num_workers, log
        self.experiments = ExperimentStore(home)
        self.registry = FilesystemArtifactStore(home, geobench_root=geobench_root)

    def selection(self, task):
        pointer = self.experiments.best(task)
        if pointer is None:
            raise ValueError(f"No selected model for {task}; finish training and selection first")
        record = self.experiments.read(pointer["experiment_id"])
        if record.get("status") != "completed" or record.get("screening") or record.get("partial_validation"):
            raise ValueError(f"{task}: selected experiment must be completed with full validation")
        frozen = self.home.selection / "manifest.json"
        if frozen.exists():
            entry = json.loads(frozen.read_text()).get("tasks", {}).get(task)
            if entry and (entry["experiment_id"] != pointer["experiment_id"] or entry["checkpoint_sha256"] != record["checkpoint"]["best_sha256"]):
                raise ValueError(f"{task}: current selection differs from frozen selection")
        return {"experiment_id": record["experiment_id"],
                "checkpoint_sha256": record["checkpoint"]["best_sha256"],
                "selected_by": pointer.get("selected_by")}

    def store(self, task, selected):
        matches = [v for v in self.registry.list_versions(task=task)
                   if v["experiment_id"] == selected["experiment_id"]
                   and v["checkpoint_sha256"] == selected["checkpoint_sha256"]]
        if matches:
            frozen = self.home.selection / "manifest.json"
            wanted = None
            if frozen.exists():
                wanted = json.loads(frozen.read_text()).get("tasks", {}).get(task, {}).get("model_version")
            match = next((v for v in matches if v["version_id"] == wanted), None)
            if wanted and match is None:
                raise ValueError(f"{task}: frozen model version is missing")
            return {"version_id": (match or matches[-1])["version_id"], "reused": True}
        if (self.home.selection / "manifest.json").exists():
            frozen = json.loads((self.home.selection / "manifest.json").read_text())
            if task in frozen.get("tasks", {}):
                raise ValueError(f"{task}: cannot recreate a missing frozen model version")
        manifest = self.registry.store_model(selected["experiment_id"], note="Automatic post-training completion")
        return {"version_id": manifest["version_id"], "reused": False}

    def verify(self, version_id):
        from gotrain.artifacts.verify import verify_version
        return verify_version(self.registry, version_id, device=self.device,
                              geobench_root=self.geobench_root)

    def review(self, task, version_id):
        from gotrain.review.model_source import from_bundle
        from gotrain.review.build import build_review, default_out_dir, load_review
        from gotrain.review.page import render_review_page
        source = from_bundle(self.registry.version_dir(version_id), device=self.device)
        out = default_out_dir(self.home, source, None)
        result = build_review(self.home, source, out, device=self.device,
                              batch_size=self.batch_size, num_workers=self.num_workers,
                              geobench_root=self.geobench_root, log=self.log)
        page = render_review_page(load_review(out))
        if result["partial"]:
            raise ValueError("Automatic completion requires a full validation review")
        comparison = result.get("metric_comparison")
        if comparison and not comparison.get("within_tolerance"):
            raise ValueError("Review validation metric differs from recorded metric beyond tolerance")
        return {"page": str(page), "n_images": result["n_images"],
                "partial": result["partial"], "metric_comparison": comparison}

    def report(self):
        from gotrain.report.html import render_dashboard
        return str(render_dashboard(self.home))


def complete_tasks(home, tasks, *, device="cpu", geobench_root=None,
                   batch_size=8, num_workers=0, log=print, operations=None):
    """Finish deliverables for explicit selections; returns a persistent stage summary.

    Failed verification blocks that task's review. Other tasks still run. Unsupported
    review types are explicitly skipped. Reruns reuse existing matching versions.
    """
    tasks = list(dict.fromkeys(tasks))
    if not tasks or batch_size < 1 or num_workers < 0:
        raise ValueError("Provide tasks, positive batch size and nonnegative workers")
    home.ensure()
    protected = _protected_state(home)
    ops = operations or Operations(home, device, geobench_root, batch_size, num_workers, log)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    summary = {"format": "gotrain-completion-v1", "run_id": run_id,
               "tasks": {}, "status": "running", "device": device,
               "scope": "validation/reference inference and deliverables only; no final test evaluation"}
    output = home.root / "completion" / f"{run_id}.json"
    summary["summary_path"] = str(output)
    _save(output, summary)
    for task in tasks:
        row = {"status": "running", "stages": {}}
        summary["tasks"][task] = row
        stage = "selection"
        try:
            log(f"[complete] {task}: checking selected model")
            selected = ops.selection(task)
            row["selection"] = selected
            row["stages"][stage] = {"status": "completed"}
            stage = "storage"
            stored = ops.store(task, selected)
            row["stages"][stage] = {"status": "completed", **stored}
            stage = "verification"
            log(f"[complete] {task}: verifying {stored['version_id']}")
            verified = ops.verify(stored["version_id"])
            if not verified.get("verified"):
                raise ValueError(f"Model verification failed: {verified}")
            row["stages"][stage] = {"status": "completed", "result": verified}
            stage = "review"
            if task in BINARY_REVIEW_TASKS:
                log(f"[complete] {task}: building full validation reviewer")
                row["stages"][stage] = {"status": "completed", **ops.review(task, stored["version_id"])}
            else:
                row["stages"][stage] = {"status": "skipped", "reason": "Interactive reviewer supports solar/cattle binary segmentation; task remains in technical dashboard"}
            stage = "selection_identity"
            if ops.selection(task) != selected:
                raise ValueError("Selection changed during completion; rerun after training/selection stops")
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["stages"][stage] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            log(f"[complete] {task}: FAILED at {stage}: {exc}")
        _save(output, summary)
    try:
        summary["dashboard"] = {"status": "completed", "path": ops.report()}
    except Exception as exc:
        summary["dashboard"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    unchanged = _protected_state(home) == protected
    summary["selection_and_final_evidence_unchanged"] = unchanged
    passed = sum(row["status"] == "completed" for row in summary["tasks"].values())
    summary["status"] = "completed" if passed == len(tasks) and summary["dashboard"]["status"] == "completed" and unchanged else ("partial" if passed else "failed")
    _save(output, summary)
    _save(home.root / "completion/latest.json", summary)
    log(f"[complete] {summary['status']}: {passed}/{len(tasks)} tasks; summary {output}")
    return summary


def cmd_complete(args):
    from gotrain.paths import Home, geobench_dir
    from gotrain.cli_extra import _tasks
    home = Home.from_env(args.home)
    result = complete_tasks(home, _tasks(args.task), device=args.device,
                            geobench_root=geobench_dir(args.geobench_dir),
                            batch_size=args.batch_size, num_workers=args.num_workers)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


def register_completion(sub):
    command = sub.add_parser("complete", help="verify selected models, build supported reviewers and refresh dashboard; never evaluates test data")
    command.add_argument("--task", default="all")
    command.add_argument("--device", default="cpu")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--num-workers", type=int, default=0)
    command.set_defaults(fn=cmd_complete)
