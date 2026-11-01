"""Freeze the model selection into a manifest that the final evaluation consumes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gotrain.config import load_task_config

from gotrain.artifacts.store import FilesystemArtifactStore
from gotrain.data.normalization import load_stats
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import append_jsonl, ensure_within, read_json, sha256_file, sha256_json, write_json
from gotrain.util.timing import utc_now_iso

MANIFEST_NAME = "manifest.json"


def manifest_path(home: Home) -> Path:
    return home.selection / MANIFEST_NAME


def freeze_selection(home: Home, tasks: list[str], overrides: dict[str, str] | None = None, by: str = "manual", replace: bool = False, store_versions: bool = True, log=print, geobench_root: Path | None = None) -> dict:
    store = ExperimentStore(home)
    registry = FilesystemArtifactStore(home, geobench_root=geobench_root)
    path = manifest_path(home)
    if path.exists() and not replace:
        raise FileExistsError(f"{path} exists; a frozen selection is not changed silently (use --replace before any test evaluation)")
    results_path = home.final_eval / "results.json"
    if path.exists() and replace and results_path.exists():
        raise PermissionError("final test results already exist; the selection cannot be replaced after test evaluation (no post-test tuning)")
    exposure = home.final_eval / "exposure.jsonl"
    if exposure.exists() and exposure.stat().st_size:
        raise PermissionError("Test exposure already recorded; refusing to freeze or replace selection")
    entries = {}
    problems = []
    for task in tasks:
        exp_id = (overrides or {}).get(task)
        if exp_id is None:
            best = store.best(task)
            if best is None:
                problems.append(f"{task}: no best pointer; run a baseline or the agent first, or pass --experiment {task}=<id>")
                continue
            exp_id = best["experiment_id"]
            selected_by = best["selected_by"]
            rationale = best.get("rationale", "")
        else:
            selected_by = by
            rationale = "explicit override at freeze time"
        rec = store.read(exp_id)
        if rec["status"] != "completed" or rec.get("screening") or rec.get("partial_validation"):
            problems.append(f"{task}: {exp_id} is not a completed full run")
            continue
        if rec["task"] != task:
            problems.append(f"{task}: {exp_id} belongs to {rec['task']}")
            continue
        ck_path = ensure_within(home.root, home.root / rec["checkpoint"]["best_path"])
        if not ck_path.exists():
            problems.append(f"{task}: checkpoint missing at {ck_path}")
            continue
        sha = sha256_file(ck_path)
        if sha != rec["checkpoint"]["best_sha256"]:
            problems.append(f"{task}: checkpoint hash changed since training")
            continue
        norm = rec.get("normalization", {})
        norm_path = norm.get("path")
        if rec["source"] == "geobench":
            stats = load_stats(Path(norm_path))
            if stats["stats_hash"] != norm.get("stats_hash"):
                problems.append(f"{task}: normalization file changed since training")
                continue
        version_id = None
        if store_versions:
            existing = [v for v in registry.list_versions(task=task) if v["experiment_id"] == exp_id and v["checkpoint_sha256"] == sha]
            if existing:
                version_id = existing[0]["version_id"]
            else:
                try:
                    version_id = registry.store_model(exp_id, note=f"frozen selection ({selected_by})")["version_id"]
                except Exception as exc:
                    raise RuntimeError(f"{task}: could not store the model version for the frozen selection: {exc} (pass store_versions=False to freeze without the registry)") from exc
            if version_id:
                registry.promote_model(version_id, rationale=f"frozen selection for final evaluation ({selected_by})", by="freeze")
        entries[task] = {
            "experiment_id": exp_id,
            "task_type": rec["task_type"],
            "task_config": asdict(load_task_config(task)) if rec["source"] == "geobench" else None,
            "source": rec["source"],
            "selected_by": selected_by,
            "selection_rationale": rationale,
            "metric": rec["metric"],
            "best_val_metric": rec["result"]["best_val_metric"],
            "checkpoint_path": rec["checkpoint"]["best_path"],
            "checkpoint_sha256": sha,
            "normalization_path": norm_path,
            "normalization_hash": norm.get("stats_hash"),
            "model": rec["model"],
            "run_config": rec["run_config"],
            "seed": rec["seed"],
            "model_version": version_id,
        }
    if problems:
        raise ValueError("cannot freeze: " + "; ".join(problems))
    manifest = {"frozen_at": utc_now_iso(), "frozen_by": by, "tasks": entries}
    manifest["manifest_hash"] = sha256_json(manifest)
    write_json(path, manifest)
    append_jsonl(home.selection / "events.jsonl", {"at": manifest["frozen_at"], "kind": "freeze", "replace": bool(replace), "tasks": {t: e["experiment_id"] for t, e in entries.items()}, "manifest_hash": manifest["manifest_hash"]})
    for task, e in entries.items():
        store.log_event(e["experiment_id"], "frozen_for_final_eval", task=task, manifest_hash=manifest["manifest_hash"])
    return manifest


def load_manifest(path: Path) -> dict:
    manifest = read_json(path)
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if sha256_json(body) != manifest.get("manifest_hash"):
        raise ValueError(f"selection manifest {path} was edited after freezing (hash mismatch)")
    return manifest
