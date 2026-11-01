"""Synthetic datasets for CPU tests and clearly-labelled fixtures.

Images carry a learnable signal (class-dependent brightness and shapes) so a few
epochs of training measurably reduce the loss. They never stand in for real
scores: every record produced from them is tagged source="synthetic".
"""

from __future__ import annotations

import numpy as np
import torch

from gotrain.data.base import DataMeta, SplitDataset, check_split_access


def _rng(seed: int, split: str) -> np.random.Generator:
    offset = {"train": 0, "valid": 10_000, "test": 20_000}[split]
    return np.random.default_rng(seed + offset)


class SyntheticClassification(SplitDataset):
    def __init__(
        self,
        split: str = "train",
        n: int = 64,
        in_channels: int = 4,
        size: int = 32,
        n_classes: int = 3,
        seed: int = 0,
        token=None,
    ) -> None:
        check_split_access(split, token)
        self.split = split
        rng = _rng(seed, split)
        self.labels = rng.integers(0, n_classes, size=n)
        base = rng.normal(0, 1, size=(n, in_channels, size, size)).astype(np.float32)
        for i, y in enumerate(self.labels):
            base[i] += 3.0 * (y + 1) * np.linspace(0, 1, in_channels, dtype=np.float32)[:, None, None]
            base[i, :, : size // 2, : size // 2] += 2.0 * y
        self.images = base * 100.0 + 500.0
        self.meta = DataMeta(
            task="synthetic-cls",
            task_type="classification",
            n_classes=n_classes,
            class_names=[f"class_{i}" for i in range(n_classes)],
            in_channels=in_channels,
            band_names=[f"band_{i}" for i in range(in_channels)],
            height=size,
            width=size,
            rgb_indices=[0, 1, 2] if in_channels >= 3 else [0, 0, 0],
            source="synthetic",
            split_sizes={"train": n, "valid": n, "test": n},
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), torch.tensor(int(self.labels[idx]), dtype=torch.int64)


class SyntheticSegmentation(SplitDataset):
    def __init__(
        self,
        split: str = "train",
        n: int = 32,
        in_channels: int = 3,
        size: int = 32,
        n_classes: int = 3,
        seed: int = 0,
        token=None,
    ) -> None:
        check_split_access(split, token)
        self.split = split
        rng = _rng(seed, split)
        images = rng.normal(0, 1, size=(n, in_channels, size, size)).astype(np.float32)
        masks = np.zeros((n, size, size), dtype=np.int64)
        for i in range(n):
            for cls in range(1, n_classes):
                h = rng.integers(size // 6, size // 2)
                w = rng.integers(size // 6, size // 2)
                r = rng.integers(0, size - h)
                c = rng.integers(0, size - w)
                masks[i, r : r + h, c : c + w] = cls
            for cls in range(1, n_classes):
                sel = masks[i] == cls
                images[i, :, sel] += 2.5 * cls
        self.images = images * 40.0 + 128.0
        self.masks = masks
        self.meta = DataMeta(
            task="synthetic-seg",
            task_type="segmentation",
            n_classes=n_classes,
            class_names=["background"] + [f"object_{i}" for i in range(1, n_classes)],
            in_channels=in_channels,
            band_names=[f"band_{i}" for i in range(in_channels)],
            height=size,
            width=size,
            rgb_indices=[0, 1, 2] if in_channels >= 3 else [0, 0, 0],
            source="synthetic",
            split_sizes={"train": n, "valid": n, "test": n},
        )

    def __len__(self) -> int:
        return len(self.masks)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), torch.from_numpy(self.masks[idx])


def synthetic_splits(task_type: str, seed: int = 0, **kwargs) -> tuple[SplitDataset, SplitDataset]:
    cls = SyntheticClassification if task_type == "classification" else SyntheticSegmentation
    return cls("train", seed=seed, **kwargs), cls("valid", seed=seed, **kwargs)
