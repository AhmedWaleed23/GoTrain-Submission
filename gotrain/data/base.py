"""Common dataset interface shared by the geobench adapter and synthetic data.

Every dataset returns ``(image, label)`` where image is a float32 tensor [C, H, W]
of raw (un-normalized) values and label is an int64 scalar (classification) or an
int64 mask [H, W] (segmentation). Normalization and augmentation are applied by
``gotrain.data.transforms`` on top.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from torch.utils.data import Dataset

SPLITS = ("train", "valid", "test")


@dataclass
class DataMeta:
    task: str
    task_type: str
    n_classes: int
    class_names: list[str]
    in_channels: int
    band_names: list[str]
    height: int
    width: int
    rgb_indices: list[int]
    source: str
    split_sizes: dict[str, int] = field(default_factory=dict)
    ignore_index: int | None = None
    dtype: str = "float32"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestSplitAccessError(PermissionError):
    __test__ = False


class FinalEvalToken:
    """Only ``gotrain.final_eval`` creates one of these. Holding it is the sole way to
    open the test split. Agent tools, smoke, baseline, inspect and report never do."""

    __slots__ = ("purpose",)

    def __init__(self, purpose: str) -> None:
        self.purpose = purpose


def check_split_access(split: str, token: FinalEvalToken | None) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "test" and not isinstance(token, FinalEvalToken):
        raise TestSplitAccessError(
            "the test split is reserved for the one-shot final evaluation; "
            "use split='valid' for all development, tuning and diagnostics"
        )


class SplitDataset(Dataset):
    """Marker base class; subclasses implement __len__ and __getitem__."""

    meta: DataMeta
    split: str
