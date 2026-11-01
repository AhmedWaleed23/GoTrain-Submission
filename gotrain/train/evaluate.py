"""Deterministic evaluation over a whole split with a confusion matrix."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from gotrain.data.transforms import logits_to_mask_resolution
from gotrain.train.metrics import ConfusionAccumulator, compute_metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    task_type: str,
    n_classes: int,
    class_names,
    device: str,
    ignore_index: int | None = None,
    exclude_from_diagnostic_miou=(),
    use_amp: bool = False,
    loss_fn=None,
) -> dict:
    model.eval()
    acc = ConfusionAccumulator(n_classes, ignore_index=ignore_index)
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device == "cuda"):
            logits = model(x)
        logits = logits_to_mask_resolution(logits.float(), y)
        if loss_fn is not None:
            loss = loss_fn(logits, y)
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                n_batches += 1
        pred = logits.argmax(dim=1)
        acc.update(pred, y)
    metrics = compute_metrics(task_type, acc.matrix(), class_names, exclude_from_diagnostic_miou)
    metrics["loss"] = total_loss / n_batches if n_batches else None
    metrics["n_ignored"] = acc.n_ignored
    return metrics
