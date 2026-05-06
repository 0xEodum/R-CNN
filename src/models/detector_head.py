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
        bg_iou_thresh: float = 0.5,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.25,
        class_weights: tuple[float, ...] | None = None,
        balanced_positive_classes: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= bg_iou_thresh <= fg_iou_thresh <= 1.0:
            raise ValueError("Expected 0.0 <= bg_iou_thresh <= fg_iou_thresh <= 1.0")
        if batch_size_per_image <= 0:
            raise ValueError("batch_size_per_image must be positive")
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be between 0.0 and 1.0")
        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.balanced_positive_classes = balanced_positive_classes
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError("class_weights length must match num_classes")
        weights_tensor = torch.tensor(class_weights or (), dtype=torch.float32)
        self.register_buffer("class_weights", weights_tensor, persistent=False)
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
        sampled = self._sample_rois(labels, [proposal.shape[0] for proposal in proposals])
        if sampled.numel() == 0:
            return {
                "detector_cls": class_logits.sum() * 0.0,
                "detector_box_reg": box_deltas.sum() * 0.0,
            }

        sampled_labels = labels[sampled]
        class_weights = self.class_weights.to(device=class_logits.device) if self.class_weights.numel() > 0 else None
        cls_loss = F.cross_entropy(class_logits[sampled], sampled_labels, weight=class_weights)
        positive = torch.where(sampled_labels > 0)[0]
        if positive.numel() > 0:
            positive_indices = sampled[positive]
            box_loss = F.smooth_l1_loss(
                box_deltas[positive_indices, sampled_labels[positive]],
                regression_targets[positive_indices],
                beta=1.0 / 9.0,
                reduction="sum",
            ) / sampled_labels.numel()
        else:
            box_loss = box_deltas.sum() * 0.0
        return {"detector_cls": cls_loss, "detector_box_reg": box_loss}

    def _sample_rois(self, labels: torch.Tensor, proposal_counts: list[int]) -> torch.Tensor:
        sampled_indices: list[torch.Tensor] = []
        start = 0
        for count in proposal_counts:
            end = start + count
            image_labels = labels[start:end]
            positive = torch.where(image_labels > 0)[0] + start
            negative = torch.where(image_labels == 0)[0] + start
            num_pos = min(int(self.batch_size_per_image * self.positive_fraction), positive.numel())
            num_neg = min(self.batch_size_per_image - num_pos, negative.numel())

            if num_pos > 0:
                if self.balanced_positive_classes:
                    sampled_indices.append(self._sample_balanced_positive_rois(labels, positive, num_pos))
                else:
                    sampled_indices.append(positive[torch.randperm(positive.numel(), device=labels.device)[:num_pos]])
            if num_neg > 0:
                sampled_indices.append(negative[torch.randperm(negative.numel(), device=labels.device)[:num_neg]])
            start = end

        if not sampled_indices:
            return torch.empty((0,), dtype=torch.long, device=labels.device)
        return torch.cat(sampled_indices, dim=0)

    @staticmethod
    def _sample_balanced_positive_rois(labels: torch.Tensor, positive: torch.Tensor, num_pos: int) -> torch.Tensor:
        positive_labels = labels[positive]
        classes = positive_labels.unique(sorted=True)
        if classes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=labels.device)

        per_class = max(1, num_pos // int(classes.numel()))
        selected: list[torch.Tensor] = []
        selected_mask = torch.zeros((positive.shape[0],), dtype=torch.bool, device=labels.device)
        for class_label in classes:
            class_positions = torch.where(positive_labels == class_label)[0]
            quota = min(per_class, class_positions.numel())
            choice = class_positions[torch.randperm(class_positions.numel(), device=labels.device)[:quota]]
            selected_mask[choice] = True
            selected.append(positive[choice])

        remaining_slots = num_pos - sum(item.numel() for item in selected)
        if remaining_slots > 0:
            remaining = positive[~selected_mask]
            if remaining.numel() > 0:
                fill = remaining[torch.randperm(remaining.numel(), device=labels.device)[:remaining_slots]]
                selected.append(fill)

        return torch.cat(selected, dim=0)[:num_pos]

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

            gt_labels = target.get("labels")
            if gt_labels is None:
                gt_labels = torch.ones((gt_boxes.shape[0],), dtype=torch.long, device=device)
            else:
                gt_labels = gt_labels.to(device=device, dtype=torch.long)
            if torch.any(gt_labels <= 0) or torch.any(gt_labels >= self.num_classes):
                raise ValueError(f"Target labels must be in [1, {self.num_classes - 1}] for this detector")

            quality = box_iou(image_proposals, gt_boxes)
            matched_vals, matched_idxs = quality.max(dim=1)
            labels = torch.full((image_proposals.shape[0],), -1, dtype=torch.long, device=device)
            labels[matched_vals < self.bg_iou_thresh] = 0
            foreground = matched_vals >= self.fg_iou_thresh
            labels[foreground] = gt_labels[matched_idxs[foreground]]
            matched_boxes = gt_boxes[matched_idxs]
            regression_targets = encode_boxes(matched_boxes, image_proposals)
            all_labels.append(labels)
            all_regression_targets.append(regression_targets)

        if not all_labels:
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0, 4), dtype=torch.float32, device=device)
        return torch.cat(all_labels, dim=0), torch.cat(all_regression_targets, dim=0)
