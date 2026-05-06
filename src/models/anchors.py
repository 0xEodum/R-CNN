from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchorGenerator:
    sizes: tuple[int, ...] = (32, 64, 128)
    ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    stride: int = 16

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.sizes) * len(self.ratios)

    def base_anchors(self, device: torch.device) -> torch.Tensor:
        anchors: list[list[float]] = []
        for size in self.sizes:
            area = float(size * size)
            for ratio in self.ratios:
                width = (area / ratio) ** 0.5
                height = width * ratio
                anchors.append([-0.5 * width, -0.5 * height, 0.5 * width, 0.5 * height])
        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def grid_anchors(self, feature_size: tuple[int, int], device: torch.device) -> torch.Tensor:
        feature_h, feature_w = feature_size
        shifts_x = torch.arange(feature_w, dtype=torch.float32, device=device) * self.stride + self.stride / 2
        shifts_y = torch.arange(feature_h, dtype=torch.float32, device=device) * self.stride + self.stride / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack(
            (shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)),
            dim=1,
        )
        anchors = self.base_anchors(device).reshape(1, -1, 4) + shifts.reshape(-1, 1, 4)
        return anchors.reshape(-1, 4)
