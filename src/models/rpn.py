from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.anchors import AnchorGenerator
from src.models.box_ops import (
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    nms,
    remove_small_boxes,
)


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        for layer in (self.conv, self.objectness, self.bbox_pred):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.conv(features))
        return self.objectness(hidden), self.bbox_pred(hidden)


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        anchor_generator: AnchorGenerator | None = None,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_top_n: int = 1000,
        post_nms_top_n: int = 200,
        nms_thresh: float = 0.7,
        min_box_size: float = 1.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator or AnchorGenerator()
        self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location)
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size

    def forward(
        self,
        features: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None,
        image_size: tuple[int, int],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        objectness_map, bbox_map = self.head(features)
        batch_size, _, feature_h, feature_w = objectness_map.shape
        anchors = self.anchor_generator.grid_anchors((feature_h, feature_w), features.device)
        objectness = objectness_map.permute(0, 2, 3, 1).reshape(batch_size, -1)
        bbox_deltas = bbox_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        proposals = self._filter_proposals(anchors, objectness.detach(), bbox_deltas.detach(), image_size)
        losses: dict[str, torch.Tensor] = {}
        if targets is not None:
            labels, matched_gt_boxes = self._assign_targets_to_anchors(anchors, targets)
            losses = self._compute_loss(objectness, bbox_deltas, labels, matched_gt_boxes, anchors)
        return proposals, losses

    def _assign_targets_to_anchors(
        self,
        anchors: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        labels_per_image: list[torch.Tensor] = []
        matched_boxes_per_image: list[torch.Tensor] = []
        for target in targets:
            gt_boxes = target["boxes"].to(device=anchors.device, dtype=anchors.dtype)
            if gt_boxes.numel() == 0:
                labels_per_image.append(torch.zeros((anchors.shape[0],), dtype=torch.float32, device=anchors.device))
                matched_boxes_per_image.append(torch.zeros_like(anchors))
                continue

            match_quality = box_iou(anchors, gt_boxes)
            matched_vals, matched_idxs = match_quality.max(dim=1)
            labels = torch.full((anchors.shape[0],), -1.0, dtype=torch.float32, device=anchors.device)
            labels[matched_vals < self.bg_iou_thresh] = 0.0
            labels[matched_vals >= self.fg_iou_thresh] = 1.0

            highest_quality_anchor = match_quality.argmax(dim=0)
            labels[highest_quality_anchor] = 1.0
            matched_boxes_per_image.append(gt_boxes[matched_idxs])
            labels_per_image.append(labels)
        return labels_per_image, matched_boxes_per_image

    def _compute_loss(
        self,
        objectness: torch.Tensor,
        bbox_deltas: torch.Tensor,
        labels: list[torch.Tensor],
        matched_gt_boxes: list[torch.Tensor],
        anchors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sampled_masks: list[torch.Tensor] = []
        positive_masks: list[torch.Tensor] = []
        for labels_per_image in labels:
            positive = torch.where(labels_per_image == 1)[0]
            negative = torch.where(labels_per_image == 0)[0]
            num_pos = min(int(self.batch_size_per_image * self.positive_fraction), positive.numel())
            num_neg = min(self.batch_size_per_image - num_pos, negative.numel())

            pos_idx = positive[torch.randperm(positive.numel(), device=positive.device)[:num_pos]]
            neg_idx = negative[torch.randperm(negative.numel(), device=negative.device)[:num_neg]]
            sampled = torch.zeros_like(labels_per_image, dtype=torch.bool)
            sampled[pos_idx] = True
            sampled[neg_idx] = True
            sampled_masks.append(sampled)
            positive_masks.append(labels_per_image == 1)

        sampled_mask = torch.stack(sampled_masks, dim=0)
        labels_tensor = torch.stack(labels, dim=0)
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_mask], labels_tensor[sampled_mask])

        box_losses: list[torch.Tensor] = []
        for image_idx, positive_mask in enumerate(positive_masks):
            if positive_mask.any():
                regression_targets = encode_boxes(matched_gt_boxes[image_idx][positive_mask], anchors[positive_mask])
                box_losses.append(
                    F.smooth_l1_loss(
                        bbox_deltas[image_idx][positive_mask],
                        regression_targets,
                        beta=1.0 / 9.0,
                        reduction="sum",
                    )
                )
        if box_losses:
            box_loss = torch.stack(box_losses).sum() / max(1, sampled_mask.sum())
        else:
            box_loss = bbox_deltas.sum() * 0.0
        return {"rpn_objectness": objectness_loss, "rpn_box_reg": box_loss}

    def _filter_proposals(
        self,
        anchors: torch.Tensor,
        objectness: torch.Tensor,
        bbox_deltas: torch.Tensor,
        image_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        proposals: list[torch.Tensor] = []
        scores = objectness.sigmoid()
        for image_scores, image_deltas in zip(scores, bbox_deltas, strict=True):
            boxes = decode_boxes(image_deltas, anchors)
            boxes = clip_boxes_to_image(boxes, image_size)
            keep = remove_small_boxes(boxes, self.min_box_size)
            boxes = boxes[keep]
            image_scores = image_scores[keep]
            top_n = min(self.pre_nms_top_n, image_scores.numel())
            if top_n > 0:
                order = image_scores.topk(top_n).indices
                boxes = boxes[order]
                image_scores = image_scores[order]
                keep_after_nms = nms(boxes, image_scores, self.nms_thresh)[: self.post_nms_top_n]
                boxes = boxes[keep_after_nms]
            proposals.append(_pad_or_trim_boxes(boxes, self.post_nms_top_n))
        return proposals


def _pad_or_trim_boxes(boxes: torch.Tensor, count: int) -> torch.Tensor:
    if boxes.shape[0] >= count:
        return boxes[:count]
    if boxes.shape[0] == 0:
        return torch.zeros((count, 4), dtype=torch.float32, device=boxes.device)
    pad = boxes[-1:].expand(count - boxes.shape[0], 4)
    return torch.cat([boxes, pad], dim=0)
