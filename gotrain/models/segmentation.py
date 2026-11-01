"""Compact U-Net encoder-decoder for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """U-Net with ``depth`` downsamplings. Inputs of any size are padded to a multiple
    of 2**depth and the logits are cropped back, so masks keep their original shape."""

    def __init__(self, in_channels: int, n_classes: int, base_width: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        widths = [base_width * (2**i) for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        cin = in_channels
        for w in widths[:-1]:
            self.encoders.append(DoubleConv(cin, w))
            cin = w
        self.bottleneck = DoubleConv(widths[-2], widths[-1])
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.decoders.append(DoubleConv(widths[i - 1] * 2, widths[i - 1]))
        self.head = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2**self.depth
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect" if min(h, w) > max(pad_h, pad_w) else "constant")
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        logits = self.head(x)
        if pad_h or pad_w:
            logits = logits[..., :h, :w]
        return logits
