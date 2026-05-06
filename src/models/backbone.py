from __future__ import annotations

import torch
from torch import nn


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallBackbone(nn.Module):
    stride = 16

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        mid1 = max(32, out_channels // 4)
        mid2 = max(48, out_channels // 2)
        self.out_channels = out_channels
        self.body = nn.Sequential(
            ConvBlock(3, mid1, stride=2),
            ConvBlock(mid1, mid1, stride=1),
            ConvBlock(mid1, mid2, stride=2),
            ConvBlock(mid2, mid2, stride=1),
            ConvBlock(mid2, out_channels, stride=2),
            ConvBlock(out_channels, out_channels, stride=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.body(images)
