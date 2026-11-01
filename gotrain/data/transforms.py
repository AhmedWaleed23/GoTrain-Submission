"""Normalization, deterministic resizing and paired augmentation.

Augmentations are flips and 90-degree rotations only, applied identically to image
and mask with no interpolation, so masks stay exact. Resizing uses bilinear for
images and nearest-neighbour for masks. Validation is deterministic (no augmentation,
no shuffling).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from gotrain.data.normalization import Normalizer


def resize_pair(x: torch.Tensor, y: torch.Tensor, size: int, task_type: str, resize_mask: bool = True):
    """Resize the image; resize the mask (nearest) only when resize_mask is set. Validation
    and test keep masks at native resolution so metrics are comparable across input sizes."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x, y
    x = F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
    if task_type == "segmentation" and resize_mask:
        y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=(size, size), mode="nearest").squeeze(0).squeeze(0).to(torch.int64)
    return x, y


def logits_to_mask_resolution(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Upsample segmentation logits (bilinear) to the target mask size when they differ."""
    if logits.ndim == 4 and target.ndim == 3 and logits.shape[-2:] != target.shape[-2:]:
        return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def augment_pair(x: torch.Tensor, y: torch.Tensor, task_type: str, gen: torch.Generator):
    k = int(torch.randint(0, 4, (1,), generator=gen))
    flip_h = bool(torch.rand(1, generator=gen) < 0.5)
    flip_v = bool(torch.rand(1, generator=gen) < 0.5)
    seg = task_type == "segmentation"
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
        if seg:
            y = torch.rot90(y, k, dims=(-2, -1))
    if flip_h:
        x = torch.flip(x, dims=(-1,))
        if seg:
            y = torch.flip(y, dims=(-1,))
    if flip_v:
        x = torch.flip(x, dims=(-2,))
        if seg:
            y = torch.flip(y, dims=(-2,))
    return x.contiguous(), y.contiguous()


class TransformedDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        normalizer: Normalizer,
        task_type: str,
        augment: bool = False,
        input_size: int | None = None,
        seed: int = 0,
        indices: list[int] | None = None,
    ) -> None:
        self.base = base
        self.normalizer = normalizer
        self.task_type = task_type
        self.augment = augment
        self.input_size = input_size
        self.seed = seed
        self.indices = indices
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.base)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx] if self.indices is not None else idx
        x, y = self.base[base_idx]
        x = x.to(torch.float32)
        if self.input_size is not None:
            x, y = resize_pair(x, y, self.input_size, self.task_type, resize_mask=self.augment)
        x = self.normalizer(x)
        if self.augment:
            gen = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch * 7919 + base_idx)
            x, y = augment_pair(x, y, self.task_type, gen)
        return x, y
