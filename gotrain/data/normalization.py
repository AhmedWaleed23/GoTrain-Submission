"""Per-band normalization statistics fitted on the TRAIN split only and persisted.

The persisted JSON records which split, how many samples, and a hash of the sample
order so an experiment record can prove which statistics it used. Display contrast
stretching for previews is a separate function and never feeds training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gotrain.util.io import read_json, sha256_json, write_json


def fit_band_stats(dataset, max_samples: int | None = None, log=None) -> dict:
    if getattr(dataset, "split", "train") != "train":
        raise ValueError("normalization statistics must be fitted on the train split only")
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    s = None
    ss = None
    count = 0
    for i in range(n):
        x, _ = dataset[i]
        x = x.to(torch.float64)
        c = x.shape[0]
        flat = x.reshape(c, -1)
        if s is None:
            s = torch.zeros(c, dtype=torch.float64)
            ss = torch.zeros(c, dtype=torch.float64)
        s += flat.sum(dim=1)
        ss += (flat * flat).sum(dim=1)
        count += flat.shape[1]
        if log and (i + 1) % 500 == 0:
            log(f"normalization: {i + 1}/{n} samples")
    if s is None:
        raise ValueError("empty dataset")
    mean = s / count
    var = ss / count - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    meta = dataset.meta
    stats = {
        "task": meta.task,
        "split": "train",
        "n_samples_used": n,
        "n_samples_in_split": len(dataset),
        "band_names": list(meta.band_names),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "sample_order_hash": sha256_json(getattr(dataset, "sample_names", list(range(len(dataset))))[:n]),
        "source": meta.source,
    }
    stats["stats_hash"] = sha256_json(stats)
    return stats


def save_stats(stats: dict, path: Path) -> str:
    stats = dict(stats)
    stats["stats_hash"] = sha256_json({k: v for k, v in stats.items() if k != "stats_hash"})
    write_json(path, stats)
    return stats["stats_hash"]


def load_stats(path: Path) -> dict:
    stats = read_json(path)
    expected = sha256_json({k: v for k, v in stats.items() if k != "stats_hash"})
    if stats.get("stats_hash") != expected:
        raise ValueError(f"normalization file {path} is corrupted or edited (hash mismatch)")
    return stats


class Normalizer:
    def __init__(self, mean, std) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)

    @staticmethod
    def from_stats(stats: dict) -> "Normalizer":
        return Normalizer(stats["mean"], stats["std"])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(f"normalizer has {self.mean.shape[0]} bands, image has {x.shape[0]}")
        return (x - self.mean) / self.std


def display_rgb(image: torch.Tensor | np.ndarray, rgb_indices, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch for previews only. Returns uint8 [H, W, 3]."""
    arr = image.numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    rgb = np.stack([arr[i] for i in rgb_indices], axis=-1).astype(np.float32)
    low, high = np.percentile(rgb, [low_pct, high_pct])
    rgb = (rgb - low) / max(float(high - low), 1e-6)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
