"""Build the review artifacts for one segmentation task from real validation inference.

Pipeline per image (validation split only, no augmentation, fp32, no gradients):
  raw bands -> resize to run_config.input_size (bilinear) -> normalize (train stats)
  -> model -> logits upsampled bilinearly to the native mask size -> softmax
  -> per-pixel normalized entropy, argmax prediction
Per image we keep: priority score (labels never used), mean entropy, predicted
foreground fraction, and, for the retrospective benchmark only, foreground IoU and
misclassified pixel count against the native-resolution ground truth.
"""

from __future__ import annotations

import io
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from gotrain.config import TaskConfig, load_task_config
from gotrain.data.base import SPLITS
from gotrain.data.normalization import Normalizer, display_rgb
from gotrain.data.transforms import logits_to_mask_resolution, resize_pair
from gotrain.paths import Home, geobench_dir
from gotrain.review.metrics import BUDGET_FRACTIONS, RANDOM_QUEUES, RANDOM_SEED, compare_to_random, cumulative_capture_curve, error_capture, foreground_iou, misclassified_pixels, random_baseline
from gotrain.review.model_source import ModelSource
from gotrain.review.signals import TOP_FRACTION, image_priority, normalized_entropy, rank_queue, signal_description
from gotrain.train.metrics import ConfusionAccumulator, segmentation_metrics
from gotrain.util.env import capture_environment
from gotrain.util.io import dumps_json, read_json, sha256_json, write_json
from gotrain.util.timing import utc_now_iso

REVIEW_SPLIT = "valid"
ENTROPY_LEVELS = 32
FORMAT_VERSION = "gotrain-review-v1"


def assert_review_split(split: str) -> None:
    """Called before any dataset is opened. Only the validation split is reviewable."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split != REVIEW_SPLIT:
        raise PermissionError(f"review inference is allowed on the {REVIEW_SPLIT!r} split only, not {split!r}")


def _encode_rgb(rgb: np.ndarray, size: int) -> tuple[bytes, str]:
    try:
        from PIL import Image

        im = Image.fromarray(rgb)
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return buf.getvalue(), "jpeg"
    except ImportError:
        from gotrain.report.png import encode_png

        if rgb.shape[0] != size:
            step = max(1, int(round(rgb.shape[0] / size)))
            rgb = rgb[::step, ::step]
        return encode_png(rgb), "png"


def _encode_gray(arr: np.ndarray) -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, "PNG", optimize=True)
        return buf.getvalue()
    except ImportError:
        from gotrain.report.png import encode_png

        return encode_png(np.stack([arr.astype(np.uint8)] * 3, axis=-1))


def _downsample_mean(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    if h == size and w == size:
        return arr
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=(size, size), mode="area")[0, 0].numpy()


class ReviewDataset(torch.utils.data.Dataset):
    """Validation items as the model expects them, plus the native mask and the index."""

    def __init__(self, base, normalizer: Normalizer, input_size: int | None, indices: list[int]) -> None:
        self.base = base
        self.normalizer = normalizer
        self.input_size = input_size
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x, y = self.base[idx]
        x = x.to(torch.float32)
        if self.input_size is not None:
            x, y = resize_pair(x, y, self.input_size, "segmentation", resize_mask=False)
        return self.normalizer(x), y, idx


def _sample_ids(base, indices: list[int]) -> list[str]:
    names = getattr(base, "sample_names", None)
    return [str(names[i]) if names is not None else f"idx{i:05d}" for i in indices]


def build_review(
    home: Home,
    source: ModelSource,
    out_dir: Path,
    device: str = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    limit: int | None = None,
    geobench_root: Path | None = None,
    thumb_size: int = 256,
    log: Callable[[str], None] = print,
    force: bool = False,
    valid_dataset=None,
    task_cfg: TaskConfig | None = None,
) -> dict:
    assert_review_split(REVIEW_SPLIT)
    if source.task_type != "segmentation":
        raise ValueError("the review workflow supports segmentation tasks only")
    out_dir = Path(out_dir)
    if valid_dataset is None:
        from gotrain.data.geobench_adapter import open_split

        task_cfg = task_cfg or load_task_config(source.task)
        valid_dataset = open_split(task_cfg, REVIEW_SPLIT, geobench_root or geobench_dir())
    meta = valid_dataset.meta
    n_total = len(valid_dataset)
    n = min(n_total, int(limit)) if limit else n_total
    indices = list(range(n))
    partial = n < n_total
    identity = {**source.identity(), "split": REVIEW_SPLIT, "n_images": n, "n_images_in_split": n_total, "partial": partial, "dataset_source": meta.source}
    identity_hash = sha256_json(identity)
    prov_path = out_dir / "provenance.json"
    if out_dir.exists() and any(out_dir.iterdir()):
        if prov_path.exists() and read_json(prov_path).get("identity_hash") == identity_hash:
            log(f"rebuilding review output for the same model/split in {out_dir}")
        elif force:
            log(f"overwriting a review output with a different identity in {out_dir} (--force)")
        else:
            raise FileExistsError(f"{out_dir} already holds a review for a different model, split or limit; choose another --out or pass --force")
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    normalizer = Normalizer.from_stats(source.normalization_stats)
    ds = ReviewDataset(valid_dataset, normalizer, source.input_size, indices)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = source.model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_classes = int(source.architecture["n_classes"])
    class_names = list(meta.class_names) if meta.class_names else [str(i) for i in range(n_classes)]
    foreground = 1 if n_classes == 2 else None
    confusion = ConfusionAccumulator(n_classes, ignore_index=meta.ignore_index)
    sample_ids = _sample_ids(valid_dataset, indices)
    items: list[dict] = []
    t0 = time.monotonic()
    done = 0
    both_empty = 0
    image_format = "jpeg"
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            logits = model(x).float()
            logits = logits_to_mask_resolution(logits, y.to(logits.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_np = y.numpy()
            confusion.update(torch.from_numpy(preds), y)
            for b in range(preds.shape[0]):
                i = int(idx[b])
                sid = sample_ids[indices.index(i)]
                ent = normalized_entropy(probs[b], axis=0)
                pred = preds[b].astype(np.uint8)
                gt = y_np[b].astype(np.int64)
                priority = image_priority(ent, TOP_FRACTION)
                errors = misclassified_pixels(pred, gt)
                if foreground is not None:
                    fg_iou, empty = foreground_iou(pred, gt, foreground)
                else:
                    fg_iou, empty = (None, False)
                both_empty += int(empty)
                raw_x, _ = valid_dataset[i]
                rgb = display_rgb(raw_x, meta.rgb_indices)
                rgb_bytes, image_format = _encode_rgb(rgb, thumb_size)
                stem = f"{len(items):04d}"
                (cache / f"{stem}_rgb.{image_format}").write_bytes(rgb_bytes)
                (cache / f"{stem}_pred.png").write_bytes(_encode_gray((pred > 0).astype(np.uint8) * 255 if n_classes == 2 else pred))
                (cache / f"{stem}_gt.png").write_bytes(_encode_gray((gt > 0).astype(np.uint8) * 255 if n_classes == 2 else gt.astype(np.uint8)))
                ent_small = _downsample_mean(ent, min(ent.shape[0], 160))
                ent_q = np.floor(np.clip(ent_small, 0, 1) * (ENTROPY_LEVELS - 1) + 0.5) * (255 // (ENTROPY_LEVELS - 1))
                (cache / f"{stem}_ent.png").write_bytes(_encode_gray(ent_q))
                items.append(
                    {
                        "stem": stem,
                        "sample_id": sid,
                        "index": i,
                        "priority": round(float(priority), 6),
                        "mean_entropy": round(float(ent.mean()), 6),
                        "uncertain_pixel_fraction": round(float((ent > 0.5).mean()), 6),
                        "predicted_foreground_fraction": round(float((pred > 0).mean()), 6),
                        "true_foreground_fraction": round(float((gt > 0).mean()), 6),
                        "foreground_iou": None if fg_iou is None else round(float(fg_iou), 6),
                        "both_empty": bool(empty),
                        "misclassified_pixels": int(errors),
                        "pixels": int(gt.size),
                        "mask_shape": [int(v) for v in gt.shape],
                    }
                )
            done += preds.shape[0]
            if done % (batch_size * 5) == 0 or done == n:
                log(f"review inference: {done}/{n} images ({time.monotonic() - t0:.0f}s)")
    elapsed = time.monotonic() - t0

    scores = {it["sample_id"]: it["priority"] for it in items}
    ranked = rank_queue(scores)
    rank_of = {sid: r + 1 for r, sid in enumerate(ranked)}
    for it in items:
        it["rank"] = rank_of[it["sample_id"]]
    errors_by_id = {it["sample_id"]: it["misclassified_pixels"] for it in items}
    ranked_capture = error_capture(ranked, errors_by_id, BUDGET_FRACTIONS)
    random = random_baseline(sorted(errors_by_id), errors_by_id, BUDGET_FRACTIONS, RANDOM_QUEUES, RANDOM_SEED)
    curve = cumulative_capture_curve(ranked, errors_by_id)
    split_metrics = segmentation_metrics(confusion.matrix(), class_names)
    recorded = source.validation_metric.get("value")
    comparison = None
    if recorded is not None and not partial:
        comparison = {"recorded_val_miou": float(recorded), "review_val_miou": float(split_metrics["miou"]), "abs_diff": abs(float(recorded) - float(split_metrics["miou"])), "tolerance": 5e-3}
        comparison["within_tolerance"] = comparison["abs_diff"] <= comparison["tolerance"]
        comparison["note"] = "recorded value came from training-time validation (possibly AMP on CUDA); the review recomputes in fp32"
    benchmark = {
        "signal": signal_description(),
        "ground_truth_used_for": "retrospective evaluation only; never for ranking",
        "budget_fractions": list(BUDGET_FRACTIONS),
        "ranked": ranked_capture,
        "random": random,
        "comparison": compare_to_random(ranked_capture, random),
        "cumulative_capture_curve": curve,
        "both_empty_images": both_empty,
        "foreground_iou_convention": "IoU of the foreground class per image; 1.0 when both target and prediction are empty (counted separately)",
        "caveat": "exploratory validation evidence on this dataset; not a deployment guarantee and not a user study of time saved",
        "partial": partial,
    }
    provenance = {
        "format": FORMAT_VERSION,
        "built_at": utc_now_iso(),
        "identity": identity,
        "identity_hash": identity_hash,
        "model": {
            "mode": source.mode,
            "task": source.task,
            "experiment_id": source.experiment_id,
            "model_version": source.model_version,
            "selected_by": source.selected_by,
            "checkpoint_path": str(source.checkpoint_path),
            "checkpoint_sha256": source.checkpoint_sha256,
            "normalization_path": str(source.normalization_path),
            "normalization_hash": source.normalization_hash,
            "architecture": source.architecture,
            "input_size": source.input_size,
            "run_config": source.run_config,
            "validation_metric": source.validation_metric,
            "reference_check": source.reference_check,
            "extra": source.extra,
        },
        "data": {"split": REVIEW_SPLIT, "dataset_source": meta.source, "n_images_in_split": n_total, "n_images_reviewed": n, "partial": partial, "class_names": class_names, "rgb_indices": list(meta.rgb_indices), "native_mask_size": items[0]["mask_shape"] if items else None},
        "inference": {"device": device, "precision": "fp32", "batch_size": batch_size, "augmentation": False, "logits_upsampled_to_native": True, "seconds": round(elapsed, 2), "seconds_per_image": round(elapsed / max(1, n), 4)},
        "thumbnails": {"rgb_size": thumb_size, "rgb_format": image_format, "entropy_size": min(items[0]["mask_shape"][0], 160) if items else None, "entropy_levels": ENTROPY_LEVELS, "masks_native": True},
        "split_metrics_fp32": {k: v for k, v in split_metrics.items() if k != "conventions"},
        "metric_comparison": comparison,
        "environment": capture_environment(),
        "synthetic": meta.source == "synthetic",
    }
    write_json(out_dir / "items.json", {"format": FORMAT_VERSION, "identity_hash": identity_hash, "items": items})
    write_json(out_dir / "queue.json", {"format": FORMAT_VERSION, "identity_hash": identity_hash, "ranked_sample_ids": ranked, "signal": signal_description()})
    write_json(out_dir / "benchmark.json", benchmark)
    write_json(prov_path, provenance)
    log(f"review build done: {n}/{n_total} images{' (PARTIAL)' if partial else ''}, {elapsed:.0f}s, outputs in {out_dir}")
    return {"out_dir": str(out_dir), "n_images": n, "n_images_in_split": n_total, "partial": partial, "seconds": round(elapsed, 2), "identity_hash": identity_hash, "benchmark": benchmark["comparison"], "split_miou_fp32": split_metrics["miou"], "metric_comparison": comparison}


def default_out_dir(home: Home, source: ModelSource, limit: int | None) -> Path:
    tag = source.model_version or source.experiment_id
    suffix = f"_limit{int(limit)}" if limit else ""
    return home.root / "review" / source.task / f"{tag}{suffix}"


def load_review(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    prov = read_json(out_dir / "provenance.json")
    items = read_json(out_dir / "items.json")
    queue = read_json(out_dir / "queue.json")
    bench = read_json(out_dir / "benchmark.json")
    if not (prov["identity_hash"] == items["identity_hash"] == queue["identity_hash"]):
        raise ValueError(f"review output {out_dir} is inconsistent (identity hashes differ)")
    return {"provenance": prov, "items": items["items"], "queue": queue["ranked_sample_ids"], "benchmark": bench, "dir": out_dir}


def summarize(review: dict) -> str:
    prov = review["provenance"]
    comp = review["benchmark"]["comparison"]
    lines = [f"review of {prov['model']['task']} with {prov['model']['experiment_id']} ({prov['model']['model_version'] or 'no registry version'})", f"images: {prov['data']['n_images_reviewed']}/{prov['data']['n_images_in_split']}{' PARTIAL' if prov['data']['partial'] else ''}, device {prov['inference']['device']}, {prov['inference']['seconds']}s"]
    if comp["applicable"]:
        for key, row in comp["budgets"].items():
            lines.append(f"  budget {key} ({row['images']} images): ranked {row['ranked_proportion']:.3f} vs random mean {row['random_mean']:.3f} [p5 {row['random_p5']:.3f}, p95 {row['random_p95']:.3f}]")
    else:
        lines.append("  no misclassified pixels: error-capture benchmark not applicable")
    return "\n".join(lines)


def dumps(obj) -> str:
    return dumps_json(obj)
