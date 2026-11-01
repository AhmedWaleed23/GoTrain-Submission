"""Selective, verified download of single GEO-Bench tasks from Hugging Face.

The official ``geobench-download`` fetches the whole 17.5 GB benchmark. This fetches
one zip, verifies its SHA-256 against the value Hugging Face reports for the LFS
object, extracts it, checks the extracted layout and writes a small manifest.
Downloading is never evaluation: nothing here opens a sample.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from gotrain.config import TaskConfig
from gotrain.data.geobench_adapter import dataset_dir, dataset_present
from gotrain.util.io import sha256_file, write_json
from gotrain.util.timing import utc_now_iso

HF_REPO = "recursix/geo-bench-1.0"
MANIFEST_NAME = ".gotrain_download.json"


def remote_file_info(filename: str) -> dict:
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(HF_REPO, [filename], repo_type="dataset")
    if not infos:
        raise FileNotFoundError(f"{filename} not found in {HF_REPO}")
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    sha = None
    if lfs is not None:
        sha = getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)
    return {"filename": filename, "size": getattr(info, "size", None), "sha256": sha}


def _extracted_ok(task_cfg: TaskConfig, root: Path) -> tuple[bool, str]:
    d = dataset_dir(task_cfg, root)
    if not dataset_present(task_cfg, root):
        return False, "task_specs.pkl or default_partition.json missing"
    with open(d / "default_partition.json") as fh:
        partition = json.load(fh)
    expected = sum(len(v) for v in partition.values())
    n_hdf5 = sum(1 for _ in d.glob("*.hdf5"))
    if n_hdf5 < expected:
        return False, f"{n_hdf5} hdf5 files but partition lists {expected} samples"
    return True, f"{n_hdf5} samples, splits {{{', '.join(f'{k}: {len(v)}' for k, v in partition.items())}}}"


def download_task(task_cfg: TaskConfig, root: Path, delete_zip: bool = False, verify_remote_hash: bool = True, log=print) -> dict:
    root = Path(root)
    target = dataset_dir(task_cfg, root)
    ok, detail = _extracted_ok(task_cfg, root)
    if ok:
        log(f"{task_cfg.name}: already present at {target} ({detail}); nothing downloaded")
        return {"task": task_cfg.name, "status": "already_present", "dataset_dir": str(target), "detail": detail}

    from huggingface_hub import hf_hub_download

    expected = remote_file_info(task_cfg.hf_filename) if verify_remote_hash else {"sha256": None, "size": None}
    log(f"{task_cfg.name}: downloading {task_cfg.hf_filename} ({expected.get('size') or '?'} bytes) to {root}")
    zip_path = Path(hf_hub_download(HF_REPO, task_cfg.hf_filename, repo_type="dataset", local_dir=str(root)))
    actual_sha = sha256_file(zip_path)
    if expected.get("sha256") and actual_sha != expected["sha256"]:
        raise ValueError(f"{zip_path}: sha256 {actual_sha} does not match Hugging Face value {expected['sha256']}")
    log(f"{task_cfg.name}: sha256 {actual_sha[:16]}... verified; extracting")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(zip_path.parent)
    ok, detail = _extracted_ok(task_cfg, root)
    if not ok:
        raise RuntimeError(f"{task_cfg.name}: extraction incomplete: {detail}")
    manifest = {
        "task": task_cfg.name,
        "hf_repo": HF_REPO,
        "hf_filename": task_cfg.hf_filename,
        "zip_sha256": actual_sha,
        "zip_size": zip_path.stat().st_size,
        "downloaded_at": utc_now_iso(),
        "detail": detail,
    }
    write_json(target / MANIFEST_NAME, manifest)
    if delete_zip:
        zip_path.unlink()
        log(f"{task_cfg.name}: removed {zip_path.name}")
    log(f"{task_cfg.name}: ready at {target} ({detail})")
    return {"task": task_cfg.name, "status": "downloaded", "dataset_dir": str(target), **manifest}
