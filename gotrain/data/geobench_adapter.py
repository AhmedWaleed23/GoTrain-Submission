"""GEO-Bench 1.0 adapter: HDF5 samples -> tensors, with metadata validation.

Channels are stacked in the order stored in each sample (``bands_order``), which is
the only reliable order for m-eurosat (its stored band names are wrong, see
configs/tasks/m-eurosat.json). The adapter checks class counts, channel counts and
patch sizes against the task config and refuses to proceed on a mismatch rather
than silently adapting.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch

from gotrain.config import TaskConfig
from gotrain.data.base import DataMeta, FinalEvalToken, SplitDataset, check_split_access


def dataset_dir(task_cfg: TaskConfig, root: Path) -> Path:
    return Path(root) / task_cfg.benchmark / task_cfg.name


def dataset_present(task_cfg: TaskConfig, root: Path) -> bool:
    d = dataset_dir(task_cfg, root)
    return (d / "task_specs.pkl").exists() and (d / "default_partition.json").exists()


def _load_specs(d: Path):
    import geobench

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return geobench.load_task_specs(d)


class GeobenchSplit(SplitDataset):
    def __init__(self, task_cfg: TaskConfig, split: str, root: Path, token: FinalEvalToken | None = None, limit: int | None = None) -> None:
        check_split_access(split, token)
        self.task_cfg = task_cfg
        self.split = split
        self.dir = dataset_dir(task_cfg, root)
        if not dataset_present(task_cfg, root):
            raise FileNotFoundError(
                f"dataset {task_cfg.name} not found at {self.dir}; run: python -m gotrain download --task {task_cfg.name}"
            )
        specs = _load_specs(self.dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ds = specs.get_dataset(split=split, band_names=None)
        partition = self.ds.active_partition.partition_dict
        self.sample_names = list(partition[split])
        if limit is not None:
            self.sample_names = self.sample_names[: int(limit)]
        label_type = specs.label_type
        n_classes = int(label_type.n_classes)
        class_names = list(label_type.class_names) if label_type.class_names else [str(i) for i in range(n_classes)]
        task_type = "segmentation" if type(label_type).__name__ == "SegmentationClasses" else "classification"
        problems = []
        if task_type != task_cfg.task_type:
            problems.append(f"dataset label type is {task_type}, config says {task_cfg.task_type}")
        if n_classes != task_cfg.n_classes:
            problems.append(f"dataset has {n_classes} classes, config says {task_cfg.n_classes}")
        band_names = [b.name for b in specs.bands_info]
        if len(band_names) != task_cfg.in_channels:
            problems.append(f"dataset has {len(band_names)} bands, config says {task_cfg.in_channels}")
        patch = list(specs.patch_size) if specs.patch_size else None
        if patch and patch != list(task_cfg.patch_size):
            problems.append(f"dataset patch size {patch}, config says {task_cfg.patch_size}")
        if problems:
            raise ValueError(f"{task_cfg.name}: metadata mismatch: " + "; ".join(problems))
        self.meta = DataMeta(
            task=task_cfg.name,
            task_type=task_type,
            n_classes=n_classes,
            class_names=class_names,
            in_channels=len(band_names),
            band_names=band_names,
            height=int(patch[0]) if patch else -1,
            width=int(patch[1]) if patch else -1,
            rgb_indices=list(task_cfg.rgb_indices),
            source="geobench",
            split_sizes={k: len(v) for k, v in partition.items()},
            ignore_index=task_cfg.ignore_index,
            extra={
                "benchmark": task_cfg.benchmark,
                "dataset_dir": str(self.dir),
                "partition": self.ds.active_partition_name,
                "spatial_resolution_m": float(specs.spatial_resolution) if specs.spatial_resolution is not None else None,
                "band_order_note": task_cfg.band_order_note,
            },
        )

    def __len__(self) -> int:
        return len(self.sample_names)

    def load_sample(self, idx: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.ds.get_sample(self.sample_names[idx])

    def __getitem__(self, idx: int):
        sample = self.load_sample(idx)
        bands = sample.bands
        if len(bands) != self.meta.in_channels:
            raise ValueError(f"sample {sample.sample_name} has {len(bands)} bands, expected {self.meta.in_channels}")
        x = np.stack([b.data for b in bands], axis=0).astype(np.float32)
        if self.meta.task_type == "classification":
            y = torch.tensor(int(sample.label), dtype=torch.int64)
        else:
            y = torch.from_numpy(np.asarray(sample.label.data).astype(np.int64))
            if y.shape != x.shape[1:]:
                raise ValueError(f"sample {sample.sample_name}: mask shape {tuple(y.shape)} != image shape {tuple(x.shape[1:])}")
        return torch.from_numpy(x), y


def open_split(task_cfg: TaskConfig, split: str, root: Path, token: FinalEvalToken | None = None, limit: int | None = None) -> GeobenchSplit:
    return GeobenchSplit(task_cfg, split, root, token=token, limit=limit)


def inspect_task(task_cfg: TaskConfig, root: Path, max_label_samples: int = 300, max_value_samples: int = 32) -> dict:
    """Metadata, label validation and value ranges from TRAIN and VALID only."""
    report = {"task": task_cfg.name, "present": dataset_present(task_cfg, root), "dataset_dir": str(dataset_dir(task_cfg, root))}
    if not report["present"]:
        report["hint"] = f"python -m gotrain download --task {task_cfg.name}"
        return report
    train = open_split(task_cfg, "train", root)
    valid = open_split(task_cfg, "valid", root)
    meta = train.meta
    report["meta"] = meta.to_dict()
    report["split_sizes"] = meta.split_sizes
    report["train_valid_overlap"] = len(set(train.sample_names) & set(valid.sample_names))
    t0 = time.monotonic()
    x0, y0 = train[0]
    report["first_sample"] = {
        "image_shape": list(x0.shape),
        "image_dtype_stored": str(train.load_sample(0).bands[0].data.dtype),
        "label_shape": list(y0.shape),
        "load_seconds": round(time.monotonic() - t0, 4),
    }
    splits_report = {}
    for name, ds in (("train", train), ("valid", valid)):
        n = min(len(ds), max_label_samples)
        counts = np.zeros(meta.n_classes, dtype=np.int64)
        bad = 0
        shapes = set()
        for i in range(n):
            x, y = ds[i]
            shapes.add(tuple(x.shape))
            if meta.task_type == "classification":
                v = int(y)
                if 0 <= v < meta.n_classes:
                    counts[v] += 1
                else:
                    bad += 1
            else:
                vals = y.numpy()
                inside = (vals >= 0) & (vals < meta.n_classes)
                if meta.ignore_index is not None:
                    inside |= vals == meta.ignore_index
                bad += int((~inside).sum())
                counts += np.bincount(vals[inside & (vals < meta.n_classes)].ravel(), minlength=meta.n_classes)[: meta.n_classes]
        total = counts.sum()
        splits_report[name] = {
            "samples_checked": n,
            "samples_in_split": len(ds),
            "label_counts": {meta.class_names[i]: int(counts[i]) for i in range(meta.n_classes)},
            "label_fractions": {meta.class_names[i]: (float(counts[i] / total) if total else 0.0) for i in range(meta.n_classes)},
            "invalid_labels": int(bad),
            "image_shapes": sorted(list(s) for s in shapes),
        }
    report["splits"] = splits_report
    m = min(len(train), max_value_samples)
    stack = torch.stack([train[i][0] for i in range(m)])
    report["train_value_ranges"] = {
        "samples": m,
        "min_per_band": [float(v) for v in stack.amin(dim=(0, 2, 3))],
        "max_per_band": [float(v) for v in stack.amax(dim=(0, 2, 3))],
        "mean_per_band": [float(v) for v in stack.mean(dim=(0, 2, 3))],
        "all_finite": bool(torch.isfinite(stack).all()),
    }
    report["warnings"] = []
    for name, rep in splits_report.items():
        if rep["invalid_labels"]:
            report["warnings"].append(f"{name}: {rep['invalid_labels']} invalid label values")
        if len(rep["image_shapes"]) > 1:
            report["warnings"].append(f"{name}: varying image shapes {rep['image_shapes']}")
    if report["train_valid_overlap"]:
        report["warnings"].append("train/valid overlap detected")
    return report
