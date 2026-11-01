"""Rebuild inference from a stored version and check it reproduces the reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gotrain.config import RunConfig, load_task_config
from gotrain.data.normalization import Normalizer, load_stats
from gotrain.data.transforms import TransformedDataset
from gotrain.models.factory import build_model
from gotrain.paths import Home, geobench_dir
from gotrain.train.checkpoint import load_checkpoint
from gotrain.util.io import ensure_within, read_json, sha256_file
from gotrain.util.timing import utc_now_iso


def load_model_from_record(home: Home, rec: dict, device: str = "cpu") -> torch.nn.Module:
    arch = rec["model"]
    model = build_model(arch["name"], arch["in_channels"], arch["n_classes"], rec["task_type"])
    path = ensure_within(home.root, home.root / rec["checkpoint"]["best_path"])
    ck = load_checkpoint(path, map_location=device, expected_sha256=rec["checkpoint"]["best_sha256"])
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


def _valid_dataset(home: Home, rec: dict, geobench_root: Path | None = None):
    run = RunConfig.from_dict(rec["run_config"])
    if rec["source"] == "synthetic":
        from gotrain.data.synthetic import synthetic_splits

        train, valid = synthetic_splits(rec["task_type"], seed=run.seed)
        from gotrain.data.normalization import fit_band_stats

        stats = fit_band_stats(train)
    else:
        from gotrain.data.geobench_adapter import open_split

        task_cfg = load_task_config(run.task)
        valid = open_split(task_cfg, "valid", geobench_root or geobench_dir())
        stats = load_stats(Path(rec["normalization"]["path"]))
        if stats.get("stats_hash") != rec["normalization"]["stats_hash"]:
            raise ValueError("normalization file hash differs from the experiment record")
    return TransformedDataset(valid, Normalizer.from_stats(stats), rec["task_type"], augment=False, input_size=run.input_size)


def synthetic_train_stats(rec: dict) -> dict:
    """Normalization statistics of a synthetic experiment (refit from its seeded train split)."""
    from gotrain.data.normalization import fit_band_stats
    from gotrain.data.synthetic import synthetic_splits

    train, _ = synthetic_splits(rec["task_type"], seed=rec["run_config"]["seed"])
    return fit_band_stats(train)


def reference_inputs(home: Home, rec: dict, n: int = 4, geobench_root: Path | None = None):
    ds = _valid_dataset(home, rec, geobench_root)
    n = min(n, len(ds))
    indices = list(range(n))
    inputs = torch.stack([ds[i][0] for i in indices])
    return inputs, indices, "valid"


def model_from_version(version_dir: Path, manifest: dict, device: str = "cpu") -> torch.nn.Module:
    arch = manifest["architecture"]
    model = build_model(arch["name"], arch["in_channels"], arch["n_classes"], manifest["task_type"])
    ck = load_checkpoint(version_dir / "checkpoint.pt", map_location=device, expected_sha256=manifest["checkpoint_sha256"])
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


def verify_version(store, version_id: str, device: str = "cpu", check_data_identity: bool = True, geobench_root: Path | None = None) -> dict:
    """Store -> retrieve -> rebuild -> predict on the stored reference batch -> compare.

    Returns a dict with verified=True only if the checksum matches, the logits agree
    within tolerance and the argmax predictions agree exactly. If the dataset is
    available and check_data_identity is set, the reference inputs are re-derived from
    the validation split through the stored normalization and compared too.
    """
    result = {"version_id": version_id, "at": utc_now_iso(), "device": device, "verified": False}
    try:
        retrieved = store.retrieve_model(version_id)
    except (ValueError, FileNotFoundError) as exc:
        result["error"] = f"retrieve failed: {exc}"
        store.record_verification(version_id, result)
        return result
    manifest = retrieved["manifest"]
    vdir = Path(retrieved["version_dir"])
    result["sha256_ok"] = True
    ref_path = vdir / manifest["reference"]["file"]
    if not ref_path.exists():
        result["error"] = "reference.npz missing"
        store.record_verification(version_id, result)
        return result
    ref = np.load(ref_path)
    inputs = torch.from_numpy(ref["inputs"]).to(device)
    try:
        model = model_from_version(vdir, manifest, device=device)
    except Exception as exc:
        result["error"] = f"rebuild failed: {type(exc).__name__}: {exc}"
        store.record_verification(version_id, result)
        return result
    with torch.no_grad():
        logits = model(inputs).float().cpu().numpy()
    tol = manifest["reference"]["tolerance"]
    diff = np.abs(logits - ref["logits"])
    close = np.allclose(logits, ref["logits"], atol=tol["atol"], rtol=tol["rtol"])
    argmax_agree = float((logits.argmax(axis=1) == ref["logits"].argmax(axis=1)).mean())
    result.update({"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "within_tolerance": bool(close), "argmax_agreement": argmax_agree, "tolerance": tol, "n_reference": int(inputs.shape[0])})
    if check_data_identity:
        try:
            rec = store.experiments.read(manifest["experiment_id"])
            fresh, _, _ = reference_inputs(store.home, rec, n=int(inputs.shape[0]), geobench_root=geobench_root or getattr(store, 'geobench_root', None))
            data_diff = float(np.abs(fresh.numpy() - ref["inputs"]).max())
            result["preprocessing_identity"] = {"checked": True, "max_abs_diff": data_diff, "ok": data_diff <= 1e-5}
        except Exception as exc:
            result["preprocessing_identity"] = {"checked": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}
    identity = result.get("preprocessing_identity", {})
    result["model_reproduced"] = bool(close and argmax_agree == 1.0)
    result["preprocessing_verified"] = bool(identity.get("checked") and identity.get("ok"))
    result["verified"] = result["model_reproduced"] and (result["preprocessing_verified"] if check_data_identity else True)
    if check_data_identity and not identity.get("checked"):
        result["note"] = "preprocessing identity could not be re-derived (dataset unavailable); only the stored reference batch was checked"
    store.record_verification(version_id, result)
    return result


def version_manifest(version_dir: Path) -> dict:
    return read_json(version_dir / "manifest.json")


def checkpoint_hash(path: Path) -> str:
    return sha256_file(path)
