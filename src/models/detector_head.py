from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.box_ops import box_iou, encode_boxes


class DetectorHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        pooled_size: tuple[int, int] = (7, 7),
        hidden_dim: int = 256,
        num_classes: int = 2,
        fg_iou_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        input_dim = in_channels * pooled_size[0] * pooled_size[1]
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

    def forward(self, pooled_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.layers(pooled_features)
        class_logits = self.cls_score(hidden)
        box_deltas = self.bbox_pred(hidden).view(pooled_features.shape[0], self.num_classes, 4)
        return class_logits, box_deltas

    def compute_loss(
        self,
        class_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        labels, regression_targets = self._assign_targets(proposals, targets, class_logits.device)
        if labels.numel() == 0:
            return {
                "detector_cls": class_logits.sum() * 0.0,
                "detector_box_reg": box_deltas.sum() * 0.0,
            }

        cls_loss = F.cross_entropy(class_logits, labels)
        positive = torch.where(labels > 0)[0]
        if positive.numel() > 0:
            box_loss = F.smooth_l1_loss(
                box_deltas[positive, labels[positive]],
                regression_targets[positive],
                beta=1.0 / 9.0,
                reduction="sum",
            ) / labels.numel()
        else:
            box_loss = box_deltas.sum() * 0.0
        return {"detector_cls": cls_loss, "detector_box_reg": box_loss}

    def _assign_targets(
        self,
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_labels: list[torch.Tensor] = []
        all_regression_targets: list[torch.Tensor] = []
        for image_proposals, target in zip(proposals, targets, strict=True):
            image_proposals = image_proposals.to(device=device, dtype=torch.float32)
            gt_boxes = target["boxes"].to(device=device, dtype=torch.float32)
            if image_proposals.numel() == 0:
                continue
            if gt_boxes.numel() == 0:
                all_labels.append(torch.zeros((image_proposals.shape[0],), dtype=torch.long, device=device))
                all_regression_targets.append(torch.zeros_like(image_proposals))
                continue

            quality = box_iou(image_proposals, gt_boxes)
            matched_vals, matched_idxs = quality.max(dim=1)
            labels = (matched_vals >= self.fg_iou_thresh).to(dtype=torch.long)
            matched_boxes = gt_boxes[matched_idxs]
            regression_targets = encode_boxes(matched_boxes, image_proposals)
            all_labels.append(labels)
            all_regression_targets.append(regression_targets)

        if not all_labels:
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0, 4), dtype=torch.float32, device=device)
        return torch.cat(all_labels, dim=0), torch.cat(all_regression_targets, dim=0)
