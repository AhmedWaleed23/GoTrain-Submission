"""Artifact store: datasets and model versions with a stable backend interface.

Filesystem backend (implemented and tested):
  <home>/registry/models.json        version index + promoted pointers (atomic writes)
  <home>/registry/events.jsonl       append-only history
  <home>/registry/models/<version>/  checkpoint.pt (COPY of the experiment's best.pt),
                                     manifest.json, normalization.json (copy),
                                     reference.npz (fixed train/valid batch + logits)
  <home>/registry/datasets.json      dataset registrations (reference or copy)

What is a copy, a reference, or a cache:
  store_model      copies the checkpoint (versions must survive experiment cleanup)
  retrieve_model   returns the stored path after a checksum check; with dest it copies
  store_dataset    mode="reference" records identity and location without copying;
                   mode="copy" duplicates the dataset directory under registry/datasets
  retrieve_dataset returns the registered location when its identity still matches
                   (no download, no duplication); with dest it copies out
S3 backend: see s3.py (same interface, optional, gated on credentials).
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.ids import validate_id, validate_task_name
from gotrain.util.io import append_jsonl, ensure_within, read_json, sha256_file, sha256_json, write_json
from gotrain.util.timing import utc_now_iso

REFERENCE_N = 4
REFERENCE_SPLIT = "valid"
TOLERANCE = {"atol": 1e-3, "rtol": 1e-3, "note": "fp32 eval on either device; AMP is not used for verification"}


class ArtifactStore(ABC):
    backend: str = "abstract"

    @abstractmethod
    def store_dataset(self, task: str, source_dir: Path, mode: str = "reference") -> dict: ...

    @abstractmethod
    def retrieve_dataset(self, task: str, dest: Path | None = None) -> dict: ...

    @abstractmethod
    def store_model(self, experiment_id: str, note: str = "", parent_version: str | None = None) -> dict: ...

    @abstractmethod
    def retrieve_model(self, version_id: str, dest: Path | None = None) -> dict: ...

    @abstractmethod
    def list_versions(self, task: str | None = None) -> list[dict]: ...

    @abstractmethod
    def promote_model(self, version_id: str, rationale: str, by: str = "manual") -> dict: ...

    @abstractmethod
    def get_version(self, version_id: str) -> dict: ...


def _validate_version_id(version_id: str) -> str:
    validate_id(version_id, "version id")
    if not version_id.startswith("mv-"):
        raise ValueError(f"invalid version id {version_id!r}")
    return version_id


class FilesystemArtifactStore(ArtifactStore):
    backend = "filesystem"

    def __init__(self, home: Home, geobench_root: Path | None = None) -> None:
        self.home = home.ensure()
        self.geobench_root = geobench_root
        self.root = home.registry
        self.models_dir = self.root / "models"
        self.datasets_dir = self.root / "datasets"
        self.index_path = self.root / "models.json"
        self.datasets_path = self.root / "datasets.json"
        self.events_path = self.root / "events.jsonl"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = ExperimentStore(home)

    def _index(self, reconcile: bool = True) -> dict:
        if self.index_path.exists():
            try:
                idx = read_json(self.index_path)
            except ValueError:
                idx = {"versions": {}, "promoted": {}, "corrupt_index_recovered_at": utc_now_iso()}
        else:
            idx = {"versions": {}, "promoted": {}}
        changed = False
        for d in sorted(self.models_dir.iterdir()) if (reconcile and self.models_dir.exists()) else []:
            manifest = d / "manifest.json"
            if d.is_dir() and manifest.exists() and d.name not in idx["versions"]:
                try:
                    idx["versions"][d.name] = read_json(manifest)
                    idx["versions"][d.name]["recovered_from_manifest"] = True
                    changed = True
                except ValueError:
                    continue
        if changed:
            self._write_index(idx)
        return idx

    def _write_index(self, idx: dict) -> None:
        idx["updated_at"] = utc_now_iso()
        write_json(self.index_path, idx)

    def _event(self, kind: str, **data: Any) -> None:
        append_jsonl(self.events_path, {"at": utc_now_iso(), "kind": kind, **data})

    def _version_dir(self, version_id: str) -> Path:
        _validate_version_id(version_id)
        return ensure_within(self.root, self.models_dir / version_id)

    def _next_version_id(self, task: str, sha: str) -> str:
        n = 1 + sum(1 for v in self._index()["versions"].values() if v.get("task") == task)
        return f"mv-{task}-{n:04d}-{sha[:8]}"

    def store_model(self, experiment_id: str, note: str = "", parent_version: str | None = None) -> dict:
        rec = self.experiments.read(experiment_id)
        if rec.get("status") != "completed" or not rec.get("checkpoint", {}).get("best_path"):
            raise ValueError(f"{experiment_id} has no completed best checkpoint")
        if parent_version:
            _validate_version_id(parent_version)
            if parent_version not in self._index()["versions"]:
                raise ValueError(f"parent version {parent_version} does not exist")
        src = ensure_within(self.home.root, self.home.root / rec["checkpoint"]["best_path"])
        sha = sha256_file(src)
        if sha != rec["checkpoint"]["best_sha256"]:
            raise ValueError(f"checkpoint on disk ({sha[:12]}) does not match the experiment record ({rec['checkpoint']['best_sha256'][:12]})")
        task = rec["task"]
        if rec["source"] == "geobench":
            validate_task_name(task)
        for v in self._index()["versions"].values():
            if v["checkpoint_sha256"] == sha and v["experiment_id"] == experiment_id:
                raise ValueError(f"this checkpoint is already stored as {v['version_id']}; versions are immutable")
        version_id = self._next_version_id(task, sha)
        vdir = self._version_dir(version_id)
        if vdir.exists():
            raise FileExistsError(f"version {version_id} already exists; refusing to overwrite")
        tmp = vdir.with_name(vdir.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        shutil.copyfile(src, tmp / "checkpoint.pt")
        copied_sha = sha256_file(tmp / "checkpoint.pt")
        if copied_sha != sha:
            shutil.rmtree(tmp)
            raise RuntimeError("checkpoint copy is corrupted (hash mismatch after copy)")
        norm_path = rec.get("normalization", {}).get("path")
        norm_copy = None
        if norm_path and Path(norm_path).exists():
            shutil.copyfile(norm_path, tmp / "normalization.json")
            norm_copy = "normalization.json"
        reference = _build_reference(self.home, rec, tmp / "reference.npz", geobench_root=self.geobench_root)
        manifest = {
            "version_id": version_id,
            "backend": self.backend,
            "task": task,
            "task_type": rec["task_type"],
            "source": rec["source"],
            "experiment_id": experiment_id,
            "parent_version": parent_version,
            "checkpoint_sha256": sha,
            "checkpoint_bytes": (tmp / "checkpoint.pt").stat().st_size,
            "storage_uri": f"file://{vdir / 'checkpoint.pt'}",
            "metric": {**rec["metric"], "value": rec["result"]["best_val_metric"], "best_epoch": rec["result"]["best_epoch"]},
            "validation_diagnostics": {k: v for k, v in (rec["result"].get("val_metrics_at_best") or {}).items() if k in ("accuracy", "macro_f1", "miou", "pixel_accuracy", "foreground_iou", "n_classes_present")},
            "architecture": rec["model"],
            "preprocessing": {
                "normalization_stats_hash": rec.get("normalization", {}).get("stats_hash"),
                "normalization_file": norm_copy,
                "normalization_source": norm_path or "in-memory (synthetic)",
                "input_size": rec.get("dataset", {}).get("input_size"),
                "band_names": rec.get("dataset", {}).get("band_names"),
                "channels_stacked_in_stored_order": True,
            },
            "run_config": rec["run_config"],
            "seed": rec["seed"],
            "pretrained": rec.get("pretrained"),
            "environment": {"torch": rec.get("environment", {}).get("torch"), "python": rec.get("environment", {}).get("python"), "gpu": rec.get("environment", {}).get("gpu")},
            "training_wall_time_seconds": rec.get("compute", {}).get("wall_time_seconds"),
            "reference": reference,
            "note": note,
            "created_at": utc_now_iso(),
            "verification": None,
        }
        write_json(tmp / "manifest.json", manifest)
        tmp.rename(vdir)
        idx = self._index(reconcile=False)
        if version_id in idx["versions"]:
            raise FileExistsError(f"version {version_id} already indexed")
        idx["versions"][version_id] = manifest
        self._write_index(idx)
        self._event("store_model", version_id=version_id, experiment_id=experiment_id, task=task, sha256=sha)
        self.experiments.log_event(experiment_id, "model_version_stored", version_id=version_id)
        return manifest

    def get_version(self, version_id: str) -> dict:
        _validate_version_id(version_id)
        idx = self._index()
        if version_id not in idx["versions"]:
            raise FileNotFoundError(f"no model version {version_id}")
        return idx["versions"][version_id]

    def version_dir(self, version_id: str) -> Path:
        return self._version_dir(version_id)

    def retrieve_model(self, version_id: str, dest: Path | None = None) -> dict:
        manifest = self.get_version(version_id)
        vdir = self._version_dir(version_id)
        ck = vdir / "checkpoint.pt"
        if not ck.exists():
            raise FileNotFoundError(f"checkpoint missing for {version_id} at {ck}")
        actual = sha256_file(ck)
        if actual != manifest["checkpoint_sha256"]:
            self._event("retrieve_model_failed", version_id=version_id, reason="sha256 mismatch", expected=manifest["checkpoint_sha256"], actual=actual)
            raise ValueError(f"{version_id}: stored checkpoint hash {actual[:12]} does not match manifest {manifest['checkpoint_sha256'][:12]}; refusing to use it")
        out_path = ck
        copied = False
        if dest is not None:
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("checkpoint.pt", "manifest.json", "normalization.json", "reference.npz"):
                if (vdir / name).exists():
                    shutil.copyfile(vdir / name, dest / name)
            out_path = dest / "checkpoint.pt"
            if sha256_file(out_path) != manifest["checkpoint_sha256"]:
                raise RuntimeError("copy corrupted during retrieval")
            copied = True
        self._event("retrieve_model", version_id=version_id, dest=str(dest) if dest else None, verified_sha256=actual)
        return {"version_id": version_id, "checkpoint_path": str(out_path), "version_dir": str(vdir), "sha256": actual, "sha256_verified": True, "copied": copied, "manifest": manifest}

    def list_versions(self, task: str | None = None) -> list[dict]:
        idx = self._index()
        rows = []
        for vid, m in idx["versions"].items():
            if task and m.get("task") != task:
                continue
            rows.append(
                {
                    "version_id": vid,
                    "task": m.get("task"),
                    "experiment_id": m.get("experiment_id"),
                    "metric": m.get("metric", {}).get("name"),
                    "value": m.get("metric", {}).get("value"),
                    "architecture": m.get("architecture", {}).get("name"),
                    "created_at": m.get("created_at"),
                    "checkpoint_sha256": m.get("checkpoint_sha256"),
                    "parent_version": m.get("parent_version"),
                    "promoted": idx["promoted"].get(m.get("task"), {}).get("version_id") == vid,
                    "verification": (m.get("verification") or {}).get("verified"),
                    "training_wall_time_seconds": m.get("training_wall_time_seconds"),
                    "note": m.get("note", ""),
                }
            )
        rows.sort(key=lambda r: r["version_id"])
        return rows

    def promoted(self, task: str) -> dict | None:
        return self._index()["promoted"].get(task)

    def promote_model(self, version_id: str, rationale: str, by: str = "manual") -> dict:
        manifest = self.get_version(version_id)
        idx = self._index()
        previous = idx["promoted"].get(manifest["task"])
        pointer = {"version_id": version_id, "task": manifest["task"], "by": by, "rationale": rationale, "at": utc_now_iso(), "previous": previous.get("version_id") if previous else None}
        idx["promoted"][manifest["task"]] = pointer
        self._write_index(idx)
        self._event("promote_model", **pointer)
        return pointer

    def record_verification(self, version_id: str, result: dict) -> None:
        idx = self._index()
        if version_id not in idx["versions"]:
            raise FileNotFoundError(version_id)
        idx["versions"][version_id]["verification"] = result
        self._write_index(idx)
        vdir = self._version_dir(version_id)
        manifest = read_json(vdir / "manifest.json")
        manifest["verification"] = result
        write_json(vdir / "manifest.json", manifest)
        self._event("verify_model", version_id=version_id, verified=result.get("verified"), max_abs_diff=result.get("max_abs_diff"))

    def _datasets(self) -> dict:
        if self.datasets_path.exists():
            return read_json(self.datasets_path)
        return {"datasets": {}}

    def store_dataset(self, task: str, source_dir: Path, mode: str = "reference") -> dict:
        validate_task_name(task)
        if mode not in ("reference", "copy"):
            raise ValueError("mode must be 'reference' or 'copy'")
        source_dir = Path(source_dir).resolve()
        if not (source_dir / "task_specs.pkl").exists():
            raise FileNotFoundError(f"{source_dir} is not a GEO-Bench task directory (no task_specs.pkl)")
        identity = dataset_identity(source_dir)
        location = source_dir
        if mode == "copy":
            location = ensure_within(self.root, self.datasets_dir / task)
            if location.exists():
                existing = dataset_identity(location)
                if existing["identity_hash"] == identity["identity_hash"]:
                    mode = "copy (already present, reused)"
                else:
                    raise FileExistsError(f"{location} holds a different dataset copy; remove it first")
            else:
                shutil.copytree(source_dir, location)
                if dataset_identity(location)["identity_hash"] != identity["identity_hash"]:
                    shutil.rmtree(location)
                    raise RuntimeError("dataset copy mismatch after copying")
        record = {"task": task, "mode": mode, "location": str(location), "source": str(source_dir), "identity": identity, "registered_at": utc_now_iso(), "backend": self.backend}
        data = self._datasets()
        data["datasets"][task] = record
        write_json(self.datasets_path, data)
        self._event("store_dataset", task=task, mode=mode, location=str(location), identity_hash=identity["identity_hash"])
        return record

    def retrieve_dataset(self, task: str, dest: Path | None = None) -> dict:
        validate_task_name(task)
        data = self._datasets()
        record = data["datasets"].get(task)
        if record is None:
            raise FileNotFoundError(f"dataset {task} is not registered; run: python -m gotrain download --task {task}")
        location = Path(record["location"])
        if not location.exists():
            raise FileNotFoundError(f"registered location {location} no longer exists")
        current = dataset_identity(location)
        if current["identity_hash"] != record["identity"]["identity_hash"]:
            raise ValueError(f"dataset {task} at {location} changed since registration (identity hash mismatch)")
        out = {"task": task, "location": str(location), "identity_verified": True, "copied": False, "mode": record["mode"]}
        if dest is not None and Path(dest).resolve() != location:
            dest = Path(dest)
            if dest.exists():
                if dataset_identity(dest)["identity_hash"] == current["identity_hash"]:
                    out.update({"location": str(dest), "copied": False, "note": "destination already holds an identical copy"})
                    return out
                raise FileExistsError(f"{dest} exists with different content")
            shutil.copytree(location, dest)
            out.update({"location": str(dest), "copied": True})
        self._event("retrieve_dataset", task=task, location=out["location"], copied=out["copied"])
        return out


def dataset_identity(dataset_dir: Path) -> dict:
    """Cheap but specific identity: partition file hash, task spec hash, sample count and total size."""
    dataset_dir = Path(dataset_dir)
    partition = dataset_dir / "default_partition.json"
    specs = dataset_dir / "task_specs.pkl"
    hdf5 = sorted(p.name for p in dataset_dir.glob("*.hdf5"))
    total = sum((dataset_dir / n).stat().st_size for n in hdf5)
    payload = {"partition_sha256": sha256_file(partition) if partition.exists() else None, "task_specs_sha256": sha256_file(specs) if specs.exists() else None, "n_hdf5": len(hdf5), "hdf5_bytes": total, "first_last": hdf5[:1] + hdf5[-1:]}
    payload["identity_hash"] = sha256_json(payload)
    return payload


def _build_reference(home: Home, rec: dict, out_path: Path, geobench_root: Path | None = None) -> dict:
    """Fixed validation batch, normalized exactly like training, with fp32 logits."""
    from gotrain.artifacts.verify import load_model_from_record, reference_inputs

    model = load_model_from_record(home, rec, device="cpu")
    inputs, indices, split = reference_inputs(home, rec, n=REFERENCE_N, geobench_root=geobench_root)
    with torch.no_grad():
        logits = model(inputs).float().cpu().numpy()
    np.savez_compressed(out_path, inputs=inputs.numpy().astype(np.float32), logits=logits.astype(np.float32), indices=np.asarray(indices))
    return {"file": "reference.npz", "split": split, "indices": list(indices), "n": int(inputs.shape[0]), "tolerance": TOLERANCE, "input_sha256": sha256_json(inputs.numpy().astype(np.float32).round(5).tolist()[:1]), "logits_argmax": logits.reshape(logits.shape[0], logits.shape[1], -1).argmax(axis=1).mean(axis=1).round(3).tolist()}


def read_manifest(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)
