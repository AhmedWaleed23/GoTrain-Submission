"""From-scratch CNN classifiers that accept any channel count and image size."""

from __future__ import annotations

import torch
from torch import nn


def conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """Three pooled conv stages, global average pooling, linear head."""

    def __init__(self, in_channels: int, n_classes: int, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn_relu(in_channels, w),
            conv_bn_relu(w, w),
            nn.MaxPool2d(2),
            conv_bn_relu(w, 2 * w),
            conv_bn_relu(2 * w, 2 * w),
            nn.MaxPool2d(2),
            conv_bn_relu(2 * w, 4 * w),
            conv_bn_relu(4 * w, 4 * w),
            nn.MaxPool2d(2),
            conv_bn_relu(4 * w, 8 * w),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNetSmall(nn.Module):
    """ResNet-18 style network at reduced width, trained from scratch."""

    def __init__(self, in_channels: int, n_classes: int, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        w = width
        self.stem = conv_bn_relu(in_channels, w)
        self.stages = nn.Sequential(
            ResidualBlock(w, w, 1),
            ResidualBlock(w, w, 1),
            ResidualBlock(w, 2 * w, 2),
            ResidualBlock(2 * w, 2 * w, 1),
            ResidualBlock(2 * w, 4 * w, 2),
            ResidualBlock(4 * w, 4 * w, 1),
            ResidualBlock(4 * w, 8 * w, 2),
            ResidualBlock(8 * w, 8 * w, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stages(self.stem(x))))
