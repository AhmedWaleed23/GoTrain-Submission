"""Retrospective usefulness of a review queue. Labels are used here only, after the
queue has been ranked without them."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

BUDGET_FRACTIONS = (0.10, 0.20, 0.30)
RANDOM_QUEUES = 200
RANDOM_SEED = 42


def foreground_iou(pred: np.ndarray, target: np.ndarray, foreground: int = 1) -> tuple[float, bool]:
    """IoU of the foreground class. Returns (iou, both_empty). Both empty counts as 1.0."""
    p = np.asarray(pred) == foreground
    t = np.asarray(target) == foreground
    union = int(np.logical_or(p, t).sum())
    if union == 0:
        return 1.0, True
    return float(np.logical_and(p, t).sum() / union), False


def misclassified_pixels(pred: np.ndarray, target: np.ndarray) -> int:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    return int((np.asarray(pred) != np.asarray(target)).sum())


def budget_sizes(n: int, fractions: Sequence[float] = BUDGET_FRACTIONS) -> dict[str, int]:
    return {f"{int(round(f * 100))}%": int(math.ceil(n * f)) for f in fractions}


def error_capture(ranked_ids: Sequence[str], errors: dict[str, int], fractions: Sequence[float] = BUDGET_FRACTIONS) -> dict:
    """Proportion of all misclassified pixels contained in the first ceil(N*f) images."""
    n = len(ranked_ids)
    total = int(sum(int(errors[s]) for s in ranked_ids))
    out = {"n_images": n, "total_misclassified_pixels": total, "applicable": total > 0, "budgets": {}}
    sizes = budget_sizes(n, fractions)
    for key, k in sizes.items():
        captured = int(sum(int(errors[s]) for s in ranked_ids[:k]))
        out["budgets"][key] = {"images": k, "captured_pixels": captured, "proportion": (captured / total) if total > 0 else None}
    return out


def cumulative_capture_curve(ranked_ids: Sequence[str], errors: dict[str, int]) -> list[float] | None:
    total = sum(int(errors[s]) for s in ranked_ids)
    if total <= 0:
        return None
    acc = 0
    curve = []
    for s in ranked_ids:
        acc += int(errors[s])
        curve.append(acc / total)
    return curve


def random_baseline(ids: Sequence[str], errors: dict[str, int], fractions: Sequence[float] = BUDGET_FRACTIONS, n_queues: int = RANDOM_QUEUES, seed: int = RANDOM_SEED) -> dict:
    """Error capture of uniformly random review orders (without replacement), fixed seed.

    Reports mean and the 5th/95th percentiles over the simulated queues per budget.
    These are random-baseline simulation bounds, not a confidence interval.
    """
    ids = list(ids)
    n = len(ids)
    total = int(sum(int(errors[s]) for s in ids))
    out = {"n_queues": n_queues, "seed": seed, "applicable": total > 0, "budgets": {}}
    sizes = budget_sizes(n, fractions)
    if total <= 0 or n == 0:
        for key, k in sizes.items():
            out["budgets"][key] = {"images": k, "mean": None, "p5": None, "p95": None}
        return out
    rng = np.random.default_rng(seed)
    err = np.array([int(errors[s]) for s in ids], dtype=np.float64)
    samples = {key: [] for key in sizes}
    for _ in range(n_queues):
        perm = rng.permutation(n)
        cum = np.cumsum(err[perm])
        for key, k in sizes.items():
            samples[key].append(float(cum[k - 1] / total))
    for key, k in sizes.items():
        arr = np.array(samples[key])
        out["budgets"][key] = {"images": k, "mean": float(arr.mean()), "p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95))}
    return out


def compare_to_random(ranked: dict, random: dict) -> dict:
    """Side-by-side table; 'beats_random_mean' is descriptive, not a significance claim."""
    rows = {}
    for key, r in ranked["budgets"].items():
        b = random["budgets"].get(key, {})
        rows[key] = {
            "images": r["images"],
            "ranked_proportion": r["proportion"],
            "random_mean": b.get("mean"),
            "random_p5": b.get("p5"),
            "random_p95": b.get("p95"),
            "beats_random_mean": (r["proportion"] > b["mean"]) if (r["proportion"] is not None and b.get("mean") is not None) else None,
            "above_random_p95": (r["proportion"] > b["p95"]) if (r["proportion"] is not None and b.get("p95") is not None) else None,
        }
    return {"applicable": ranked["applicable"], "budgets": rows}
