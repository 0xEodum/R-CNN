from __future__ import annotations

import torch
from torch import nn

from src.models.anchors import AnchorGenerator
from src.models.backbone import SmallBackbone
from src.models.box_ops import clip_boxes_to_image, decode_boxes, nms
from src.models.detector_head import DetectorHead
from src.models.roi_pool import RoIAlignPool
from src.models.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        backbone_channels: int = 128,
        hidden_dim: int = 256,
        rpn_pre_nms_top_n: int = 600,
        rpn_post_nms_top_n: int = 100,
        anchor_sizes: tuple[int, ...] = (16, 32, 64),
        score_thresh: float = 0.05,
        detections_per_image: int = 50,
    ) -> None:
        super().__init__()
        self.backbone = SmallBackbone(out_channels=backbone_channels)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, stride=self.backbone.stride)
        self.rpn = RegionProposalNetwork(
            in_channels=backbone_channels,
            anchor_generator=anchor_generator,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
        )
        pooled_size = (7, 7)
        self.roi_pool = RoIAlignPool(output_size=pooled_size, spatial_scale=1.0 / self.backbone.stride)
        self.detector_head = DetectorHead(
            in_channels=backbone_channels,
            pooled_size=pooled_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.detections_per_image = detections_per_image

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        image_size = (images.shape[-2], images.shape[-1])
        features = self.backbone(images)
        proposals, rpn_losses = self.rpn(features, targets=targets, image_size=image_size)

        training_proposals = proposals
        if self.training:
            if targets is None:
                raise ValueError("targets must be provided when FasterRCNN is in training mode")
            training_proposals = self._append_ground_truth(proposals, targets, images.device)

        pooled = self.roi_pool(features, training_proposals)
        class_logits, box_deltas = self.detector_head(pooled)

        if self.training:
            detector_losses = self.detector_head.compute_loss(class_logits, box_deltas, training_proposals, targets or [])
            return {**rpn_losses, **detector_losses}

        return self._postprocess_predictions(class_logits, box_deltas, training_proposals, image_size)

    @staticmethod
    def _append_ground_truth(
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        output: list[torch.Tensor] = []
        for image_proposals, target in zip(proposals, targets, strict=True):
            gt_boxes = target["boxes"].to(device=device, dtype=image_proposals.dtype)
            output.append(torch.cat([image_proposals, gt_boxes], dim=0))
        return output

    def _postprocess_predictions(
        self,
        class_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        proposals: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        probabilities = class_logits.softmax(dim=1)
        predictions: list[dict[str, torch.Tensor]] = []
        start = 0
        for image_proposals in proposals:
            count = image_proposals.shape[0]
            end = start + count
            image_scores = probabilities[start:end, 1]
            image_boxes = decode_boxes(box_deltas[start:end, 1], image_proposals)
            image_boxes = clip_boxes_to_image(image_boxes, image_size)
            keep = torch.where(image_scores >= self.score_thresh)[0]
            image_boxes = image_boxes[keep]
            image_scores = image_scores[keep]
            if image_scores.numel() > 0:
                keep_nms = nms(image_boxes, image_scores, iou_threshold=0.5)[: self.detections_per_image]
                image_boxes = image_boxes[keep_nms]
                image_scores = image_scores[keep_nms]
            predictions.append(
                {
                    "boxes": image_boxes,
                    "scores": image_scores,
                    "labels": torch.ones((image_scores.shape[0],), dtype=torch.int64, device=image_scores.device),
                }
            )
            start = end
        return predictions
