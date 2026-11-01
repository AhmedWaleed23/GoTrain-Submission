"""Model allow-list and construction. Channel and class counts come from task metadata."""

from __future__ import annotations

from torch import nn

from gotrain.models.classification import ResNetSmall, SmallCNN
from gotrain.models.segmentation import UNet

MODEL_SPECS = {
    "cnn_small": {
        "task_type": "classification",
        "description": "4-stage plain CNN, width 32, ~0.6M params; fast baseline",
    },
    "resnet_small": {
        "task_type": "classification",
        "description": "ResNet-18 layout at width 32 (~2.8M params), from scratch; stronger, slower",
    },
    "unet_small": {
        "task_type": "segmentation",
        "description": "U-Net base width 16, depth 4 (~1.9M params); fast baseline",
    },
    "unet_medium": {
        "task_type": "segmentation",
        "description": "U-Net base width 32, depth 4 (~7.8M params); stronger, roughly 3x the compute",
    },
}


def models_for(task_type: str) -> list[str]:
    return [name for name, spec in MODEL_SPECS.items() if spec["task_type"] == task_type]


def build_model(name: str, in_channels: int, n_classes: int, task_type: str) -> nn.Module:
    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown model {name!r}; allowed: {sorted(MODEL_SPECS)}")
    if spec["task_type"] != task_type:
        raise ValueError(f"model {name!r} is for {spec['task_type']}, task is {task_type}")
    if name == "cnn_small":
        return SmallCNN(in_channels, n_classes, width=32)
    if name == "resnet_small":
        return ResNetSmall(in_channels, n_classes, width=32)
    if name == "unet_small":
        return UNet(in_channels, n_classes, base_width=16, depth=4)
    if name == "unet_medium":
        return UNet(in_channels, n_classes, base_width=32, depth=4)
    raise AssertionError(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
