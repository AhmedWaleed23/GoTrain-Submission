"""Where the reviewed model comes from.

Native mode (SageMaker): the task's selected experiment pointer in the run home, or
an explicit experiment id; checkpoint and normalization are the experiment's own,
hash-checked. Bundle mode (laptop): a portable directory holding manifest.json,
checkpoint.pt, normalization.json and reference.npz copied from the registry; the
checkpoint SHA-256 and the normalization hash are verified against the manifest and
the model must reproduce the stored reference logits before any review inference.
Nothing here writes to the run home or the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gotrain.artifacts.verify import load_model_from_record, model_from_version
from gotrain.data.normalization import load_stats
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import ensure_within, read_json, sha256_file

BUNDLE_FILES = ("manifest.json", "checkpoint.pt", "normalization.json")


@dataclass
class ModelSource:
    mode: str
    task: str
    task_type: str
    experiment_id: str
    model_version: str | None
    checkpoint_path: Path
    checkpoint_sha256: str
    normalization_path: Path
    normalization_hash: str
    normalization_stats: dict
    input_size: int | None
    architecture: dict
    run_config: dict
    validation_metric: dict
    model: torch.nn.Module
    selected_by: str | None = None
    reference_check: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict:
        return {
            "task": self.task,
            "experiment_id": self.experiment_id,
            "model_version": self.model_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "normalization_hash": self.normalization_hash,
            "input_size": self.input_size,
            "architecture": self.architecture.get("name"),
        }


def _input_size_from_run_config(run_config: dict) -> int | None:
    value = run_config.get("input_size")
    return int(value) if value else None


def from_bundle(bundle_dir: Path, device: str = "cpu", check_reference: bool = True) -> ModelSource:
    bundle_dir = Path(bundle_dir).resolve()
    missing = [name for name in BUNDLE_FILES if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"bundle {bundle_dir} is missing {missing}")
    for name in BUNDLE_FILES + ("reference.npz",):
        ensure_within(bundle_dir, bundle_dir / name)
    manifest = read_json(bundle_dir / "manifest.json")
    for key in ("version_id", "task", "task_type", "experiment_id", "checkpoint_sha256", "architecture", "run_config", "preprocessing", "metric"):
        if key not in manifest:
            raise ValueError(f"bundle manifest lacks {key!r}")
    ck_path = bundle_dir / "checkpoint.pt"
    actual = sha256_file(ck_path)
    if actual != manifest["checkpoint_sha256"]:
        raise ValueError(f"bundle checkpoint hash {actual[:12]} does not match manifest {manifest['checkpoint_sha256'][:12]}")
    stats = load_stats(bundle_dir / "normalization.json")
    expected_norm = manifest["preprocessing"].get("normalization_stats_hash")
    if expected_norm and stats.get("stats_hash") != expected_norm:
        raise ValueError("bundle normalization hash does not match the manifest")
    model = model_from_version(bundle_dir, manifest, device=device)
    selected_by = None
    for candidate in (bundle_dir / "selected_results.json", bundle_dir.parent / "selected_results.json"):
        if candidate.exists():
            try:
                sel = read_json(candidate).get(manifest["task"], {}).get("selection", {})
                if sel.get("experiment_id") == manifest["experiment_id"]:
                    selected_by = sel.get("selected_by")
            except (ValueError, AttributeError):
                selected_by = None
            break
    src = ModelSource(
        mode="bundle",
        task=manifest["task"],
        task_type=manifest["task_type"],
        experiment_id=manifest["experiment_id"],
        model_version=manifest["version_id"],
        checkpoint_path=ck_path,
        checkpoint_sha256=actual,
        normalization_path=bundle_dir / "normalization.json",
        normalization_hash=stats["stats_hash"],
        normalization_stats=stats,
        input_size=_input_size_from_run_config(manifest["run_config"]),
        architecture=manifest["architecture"],
        run_config=manifest["run_config"],
        validation_metric={**manifest["metric"], "diagnostics": manifest.get("validation_diagnostics")},
        model=model,
        selected_by=selected_by,
        extra={"bundle_dir": str(bundle_dir), "source_provenance": {"storage_uri": manifest.get("storage_uri"), "normalization_source": manifest["preprocessing"].get("normalization_source")}},
    )
    if check_reference:
        src.reference_check = check_reference_outputs(src, bundle_dir / "reference.npz", manifest.get("reference", {}), device)
        if not src.reference_check["within_tolerance"]:
            raise ValueError(f"bundle model does not reproduce its stored reference logits: {src.reference_check}")
    return src


def from_home(home: Home, task: str, device: str = "cpu", experiment_id: str | None = None, check_reference: bool = True) -> ModelSource:
    store = ExperimentStore(home)
    selected_by = None
    if experiment_id is None:
        best = store.best(task)
        if best is None:
            raise FileNotFoundError(f"no selected experiment for {task} in {home.root}; pass --experiment or run the agent/baseline first")
        experiment_id = best["experiment_id"]
        selected_by = best.get("selected_by")
    rec = store.read(experiment_id)
    if rec["task"] != task:
        raise ValueError(f"{experiment_id} belongs to task {rec['task']}, not {task}")
    if rec["status"] != "completed" or not rec.get("checkpoint", {}).get("best_path"):
        raise ValueError(f"{experiment_id} has no completed best checkpoint")
    if rec.get("screening") or rec.get("partial_validation"):
        raise ValueError(f"{experiment_id} is a screening or partial-validation run; review the selected full run instead")
    ck_path = ensure_within(home.root, home.root / rec["checkpoint"]["best_path"])
    actual = sha256_file(ck_path)
    if actual != rec["checkpoint"]["best_sha256"]:
        raise ValueError("checkpoint on disk does not match the experiment record")
    if rec["source"] == "synthetic":
        from gotrain.artifacts.verify import synthetic_train_stats

        stats = synthetic_train_stats(rec)
        norm_path = Path("in-memory-synthetic")
    else:
        norm_path = Path(rec["normalization"]["path"])
        if not norm_path.is_absolute() or not norm_path.exists():
            candidate = home.normalization / f"{task}.json"
            if candidate.exists():
                norm_path = candidate
        if not norm_path.exists():
            raise FileNotFoundError(f"normalization file {rec['normalization']['path']} not found (and no {home.normalization / (task + '.json')})")
        stats = load_stats(norm_path)
    if stats.get("stats_hash") != rec["normalization"]["stats_hash"]:
        raise ValueError("normalization statistics do not match the experiment record")
    model = load_model_from_record(home, rec, device=device)
    version_id = None
    try:
        from gotrain.artifacts.store import FilesystemArtifactStore

        for v in FilesystemArtifactStore(home).list_versions(task=task):
            if v["experiment_id"] == experiment_id and v["checkpoint_sha256"] == actual:
                version_id = v["version_id"]
    except Exception:
        version_id = None
    src = ModelSource(
        mode="home",
        task=task,
        task_type=rec["task_type"],
        experiment_id=experiment_id,
        model_version=version_id,
        checkpoint_path=ck_path,
        checkpoint_sha256=actual,
        normalization_path=norm_path,
        normalization_hash=stats["stats_hash"],
        normalization_stats=stats,
        input_size=_input_size_from_run_config(rec["run_config"]),
        architecture=rec["model"],
        run_config=rec["run_config"],
        validation_metric={**rec["metric"], "value": rec["result"]["best_val_metric"], "diagnostics": {k: v for k, v in (rec["result"].get("val_metrics_at_best") or {}).items() if k in ("miou", "foreground_iou", "pixel_accuracy", "n_classes_present")}},
        model=model,
        selected_by=selected_by,
        extra={"home": str(home.root)},
    )
    if check_reference and version_id:
        from gotrain.artifacts.store import FilesystemArtifactStore

        vdir = FilesystemArtifactStore(home).version_dir(version_id)
        ref = vdir / "reference.npz"
        if ref.exists():
            src.reference_check = check_reference_outputs(src, ref, read_json(vdir / "manifest.json").get("reference", {}), device)
    return src


def check_reference_outputs(src: ModelSource, reference_path: Path, reference_meta: dict, device: str) -> dict:
    ref = np.load(reference_path)
    inputs = torch.from_numpy(ref["inputs"]).to(device)
    tol = reference_meta.get("tolerance", {"atol": 1e-3, "rtol": 1e-3})
    with torch.no_grad():
        logits = src.model(inputs).float().cpu().numpy()
    diff = np.abs(logits - ref["logits"])
    return {
        "file": str(reference_path),
        "n": int(inputs.shape[0]),
        "max_abs_diff": float(diff.max()),
        "within_tolerance": bool(np.allclose(logits, ref["logits"], atol=tol.get("atol", 1e-3), rtol=tol.get("rtol", 1e-3))),
        "argmax_agreement": float((logits.argmax(axis=1) == ref["logits"].argmax(axis=1)).mean()),
        "tolerance": tol,
        "device": device,
        "scope": "stored reference batch only (4 validation inputs), not full-split accuracy",
    }
