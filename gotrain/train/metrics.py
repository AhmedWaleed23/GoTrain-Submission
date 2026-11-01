"""Confusion-matrix based metrics.

Conventions (frozen, see docs/PLAN.md):
- Classification selection metric ``accuracy``: overall accuracy over the full split,
  equal to torchmetrics ``Accuracy(task="multiclass")`` as used by geo-bench-experiments.
- Segmentation selection metric ``miou``: per-class IoU from a confusion matrix
  accumulated over the whole split, averaged with equal weight over classes that
  appear in the target or the prediction at least once. Classes absent from both are
  excluded. This equals torchmetrics ``JaccardIndex(task="multiclass", average="macro")``
  (verified against torchmetrics 1.9.0 in tests). No ignore index unless the task
  config sets one.
Diagnostics (not used for selection): macro-F1, per-class precision/recall/F1,
per-class IoU, pixel accuracy, mIoU excluding named classes.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

SELECTION_METRIC = {"classification": "accuracy", "segmentation": "miou"}
SELECTION_DIRECTION = "max"


class ConfusionAccumulator:
    """Accumulates a K x K confusion matrix (rows = target, cols = prediction)."""

    def __init__(self, n_classes: int, ignore_index: int | None = None) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        self.ignore_index = ignore_index
        self._matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int64)
        self.n_ignored = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
        pred = pred.reshape(-1).to(torch.int64)
        target = target.reshape(-1).to(torch.int64)
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            self.n_ignored += int((~keep).sum().item())
            pred = pred[keep]
            target = target[keep]
        if target.numel() == 0:
            return
        if int(target.min()) < 0 or int(target.max()) >= self.n_classes:
            raise ValueError(
                f"target labels outside [0, {self.n_classes}): min={int(target.min())} max={int(target.max())}"
            )
        if int(pred.min()) < 0 or int(pred.max()) >= self.n_classes:
            raise ValueError(f"predictions outside [0, {self.n_classes})")
        idx = target * self.n_classes + pred
        counts = torch.bincount(idx, minlength=self.n_classes * self.n_classes)
        self._matrix += counts.reshape(self.n_classes, self.n_classes).cpu()

    def matrix(self) -> np.ndarray:
        return self._matrix.numpy().copy()

    @property
    def total(self) -> int:
        return int(self._matrix.sum().item())


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def classification_metrics(confmat: np.ndarray, class_names: Sequence[str] | None = None) -> dict:
    cm = np.asarray(confmat, dtype=np.int64)
    k = cm.shape[0]
    names = list(class_names) if class_names else [str(i) for i in range(k)]
    total = int(cm.sum())
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    per_class = []
    f1s = []
    for i in range(k):
        precision = _safe_div(tp[i], tp[i] + fp[i])
        recall = _safe_div(tp[i], tp[i] + fn[i])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1s.append(f1)
        per_class.append(
            {
                "class": names[i],
                "support": int(cm[i].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": _safe_div(tp.sum(), total),
        "macro_f1": float(np.mean(f1s)) if k else 0.0,
        "n_samples": total,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "conventions": {
            "accuracy": "overall accuracy over the full split (torchmetrics multiclass Accuracy)",
            "macro_f1": "unweighted mean of per-class F1; absent classes count as 0 (diagnostic only)",
        },
    }


def segmentation_metrics(
    confmat: np.ndarray,
    class_names: Sequence[str] | None = None,
    exclude_from_diagnostic_miou: Sequence[int] = (),
) -> dict:
    cm = np.asarray(confmat, dtype=np.int64)
    k = cm.shape[0]
    names = list(class_names) if class_names else [str(i) for i in range(k)]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    present = denom > 0
    iou = np.where(present, tp / np.maximum(denom, 1), np.nan)

    per_class = []
    for i in range(k):
        per_class.append(
            {
                "class": names[i],
                "target_pixels": int(cm[i].sum()),
                "predicted_pixels": int(cm[:, i].sum()),
                "iou": None if math.isnan(iou[i]) else float(iou[i]),
                "present": bool(present[i]),
            }
        )
    miou = float(np.nanmean(iou)) if present.any() else 0.0

    excluded = [int(i) for i in exclude_from_diagnostic_miou if 0 <= int(i) < k]
    keep = present.copy()
    for i in excluded:
        keep[i] = False
    miou_excl = float(np.nanmean(iou[keep])) if keep.any() else 0.0

    out = {
        "miou": miou,
        "n_classes_present": int(present.sum()),
        "pixel_accuracy": _safe_div(tp.sum(), cm.sum()),
        "n_pixels": int(cm.sum()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "conventions": {
            "miou": "macro mean of per-class IoU from the split-level confusion matrix; classes absent from both target and prediction are excluded (torchmetrics multiclass JaccardIndex, average=macro)",
            "pixel_accuracy": "diagnostic only; misleading on imbalanced masks",
        },
    }
    if k == 2:
        out["foreground_iou"] = per_class[1]["iou"]
    if excluded:
        out["miou_excluding"] = {"classes": [names[i] for i in excluded], "value": miou_excl}
    return out


def compute_metrics(
    task_type: str,
    confmat: np.ndarray,
    class_names: Sequence[str] | None = None,
    exclude_from_diagnostic_miou: Sequence[int] = (),
) -> dict:
    if task_type == "classification":
        return classification_metrics(confmat, class_names)
    if task_type == "segmentation":
        return segmentation_metrics(confmat, class_names, exclude_from_diagnostic_miou)
    raise ValueError(f"unknown task type {task_type!r}")
