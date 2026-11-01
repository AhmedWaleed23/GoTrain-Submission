"""Uncertainty signal and queue ranking.

The signal is fixed and simple: per-pixel predictive entropy of the softmax output,
normalized to [0, 1] by log(n_classes), aggregated per image as the mean of its
highest-entropy 10 percent of pixels. Ties are broken by sample id so the order is
stable. Labels never enter this module.
"""

from __future__ import annotations

import math

import numpy as np

TOP_FRACTION = 0.10
SIGNAL_NAME = "top10pct_mean_normalized_entropy"


def normalized_entropy(probs: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Entropy of a probability distribution along ``axis`` divided by log(n_classes).

    probs: array with class probabilities along ``axis`` (sums to 1). Returns an array
    without that axis, values in [0, 1]: 0 for a one-hot prediction, 1 for uniform.
    """
    p = np.asarray(probs, dtype=np.float64)
    n_classes = p.shape[axis]
    if n_classes < 2:
        raise ValueError("entropy needs at least two classes")
    sums = p.sum(axis=axis)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError("probabilities along the class axis must sum to 1")
    ent = -(p * np.log(np.clip(p, eps, 1.0))).sum(axis=axis)
    return np.clip(ent / math.log(n_classes), 0.0, 1.0)


def image_priority(entropy_map: np.ndarray, top_fraction: float = TOP_FRACTION) -> float:
    """Mean of the highest ``top_fraction`` of pixel entropies (at least one pixel)."""
    flat = np.asarray(entropy_map, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("empty entropy map")
    k = max(1, int(math.ceil(flat.size * top_fraction)))
    top = np.partition(flat, flat.size - k)[flat.size - k :]
    return float(top.mean())


def rank_queue(scores: dict[str, float]) -> list[str]:
    """Sample ids ordered by descending priority, ties broken by ascending sample id."""
    return [sid for sid, _ in sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))]


def signal_description() -> dict:
    return {
        "name": SIGNAL_NAME,
        "pixel_signal": "normalized predictive entropy of the softmax output (entropy / log(n_classes))",
        "image_aggregate": f"mean of the highest-entropy {int(TOP_FRACTION * 100)} percent of pixels",
        "tie_break": "ascending sample id",
        "uses_labels": False,
        "caveat": "an uncertainty heuristic, not calibrated correctness and not an explanation of the model",
    }
