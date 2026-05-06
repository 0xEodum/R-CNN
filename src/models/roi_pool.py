from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RoIAlignPool(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (7, 7), spatial_scale: float = 1.0 / 16.0) -> None:
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features: torch.Tensor, proposals: list[torch.Tensor]) -> torch.Tensor:
        roi_boxes: list[torch.Tensor] = []
        roi_batch_indices: list[torch.Tensor] = []
        for batch_idx, boxes in enumerate(proposals):
            if boxes.numel() == 0:
                continue
            boxes = boxes.to(device=features.device, dtype=features.dtype)
            roi_boxes.append(boxes)
            roi_batch_indices.append(torch.full((boxes.shape[0],), batch_idx, dtype=torch.long, device=features.device))

        if not roi_boxes:
            return features.new_zeros((0, features.shape[1], self.output_size[0], self.output_size[1]))

        boxes = torch.cat(roi_boxes, dim=0) * self.spatial_scale
        batch_indices = torch.cat(roi_batch_indices, dim=0)
        selected_features = features[batch_indices]
        grid = self._make_sampling_grid(boxes, features.shape[-2], features.shape[-1])
        return F.grid_sample(selected_features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _make_sampling_grid(self, boxes: torch.Tensor, feature_h: int, feature_w: int) -> torch.Tensor:
        pooled_h, pooled_w = self.output_size
        dtype = boxes.dtype
        device = boxes.device

        steps_y = (torch.arange(pooled_h, dtype=dtype, device=device) + 0.5) / pooled_h
        steps_x = (torch.arange(pooled_w, dtype=dtype, device=device) + 0.5) / pooled_w
        y1, x1 = boxes[:, 1], boxes[:, 0]
        y2, x2 = boxes[:, 3], boxes[:, 2]

        sample_y = y1[:, None] + (y2 - y1).clamp(min=1.0)[:, None] * steps_y[None, :]
        sample_x = x1[:, None] + (x2 - x1).clamp(min=1.0)[:, None] * steps_x[None, :]
        grid_y, grid_x = torch.meshgrid(sample_y[0], sample_x[0], indexing="ij")
        base_shape = (boxes.shape[0], pooled_h, pooled_w)
        grid_y = sample_y[:, :, None].expand(base_shape)
        grid_x = sample_x[:, None, :].expand(base_shape)

        norm_x = (grid_x / max(feature_w - 1, 1)) * 2.0 - 1.0
        norm_y = (grid_y / max(feature_h - 1, 1)) * 2.0 - 1.0
        return torch.stack((norm_x, norm_y), dim=-1)
