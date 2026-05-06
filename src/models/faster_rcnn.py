from __future__ import annotations

import torch
from torch import nn

from src.models.anchors import AnchorGenerator
from src.models.backbone import SmallBackbone
from src.models.box_ops import clip_boxes_to_image, decode_boxes, nms, soft_nms
from src.models.detector_head import DetectorHead
from src.models.roi_pool import RoIAlignPool
from src.models.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        backbone_channels: int = 128,
        backbone_stride: int = 16,
        hidden_dim: int = 256,
        rpn_pre_nms_top_n: int = 600,
        rpn_post_nms_top_n: int = 100,
        anchor_sizes: tuple[int, ...] = (16, 32, 64),
        score_thresh: float = 0.05,
        detections_per_image: int = 50,
        postprocess_nms: str = "hard",
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        detector_fg_iou_thresh: float = 0.5,
        detector_bg_iou_thresh: float = 0.5,
        detector_batch_size_per_image: int = 256,
        detector_positive_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if postprocess_nms not in {"hard", "soft"}:
            raise ValueError("postprocess_nms must be 'hard' or 'soft'")
        self.backbone = SmallBackbone(out_channels=backbone_channels, output_stride=backbone_stride)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, stride=self.backbone.stride)
        self.rpn = RegionProposalNetwork(
            in_channels=backbone_channels,
            anchor_generator=anchor_generator,
            fg_iou_thresh=rpn_fg_iou_thresh,
            bg_iou_thresh=rpn_bg_iou_thresh,
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
            fg_iou_thresh=detector_fg_iou_thresh,
            bg_iou_thresh=detector_bg_iou_thresh,
            batch_size_per_image=detector_batch_size_per_image,
            positive_fraction=detector_positive_fraction,
        )
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.detections_per_image = detections_per_image
        self.postprocess_nms = postprocess_nms

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
            boxes_per_class: list[torch.Tensor] = []
            scores_per_class: list[torch.Tensor] = []
            labels_per_class: list[torch.Tensor] = []
            for class_index in range(1, self.num_classes):
                image_scores = probabilities[start:end, class_index]
                image_boxes = decode_boxes(box_deltas[start:end, class_index], image_proposals)
                image_boxes = clip_boxes_to_image(image_boxes, image_size)
                keep = torch.where(image_scores >= self.score_thresh)[0]
                image_boxes = image_boxes[keep]
                image_scores = image_scores[keep]
                if image_scores.numel() == 0:
                    continue
                if self.postprocess_nms == "soft":
                    keep_nms, decayed_scores = soft_nms(
                        image_boxes,
                        image_scores,
                        iou_threshold=0.5,
                        score_threshold=self.score_thresh,
                    )
                    image_scores = decayed_scores[keep_nms]
                else:
                    keep_nms = nms(image_boxes, image_scores, iou_threshold=0.5)
                    image_scores = image_scores[keep_nms]
                image_boxes = image_boxes[keep_nms]
                boxes_per_class.append(image_boxes)
                scores_per_class.append(image_scores)
                labels_per_class.append(
                    torch.full((image_scores.shape[0],), class_index, dtype=torch.int64, device=image_scores.device)
                )
            if boxes_per_class:
                image_boxes = torch.cat(boxes_per_class, dim=0)
                image_scores = torch.cat(scores_per_class, dim=0)
                image_labels = torch.cat(labels_per_class, dim=0)
                order = image_scores.argsort(descending=True)[: self.detections_per_image]
                image_boxes = image_boxes[order]
                image_scores = image_scores[order]
                image_labels = image_labels[order]
            else:
                image_boxes = torch.empty((0, 4), dtype=image_proposals.dtype, device=image_proposals.device)
                image_scores = torch.empty((0,), dtype=probabilities.dtype, device=probabilities.device)
                image_labels = torch.empty((0,), dtype=torch.int64, device=probabilities.device)
            predictions.append(
                {
                    "boxes": image_boxes,
                    "scores": image_scores,
                    "labels": image_labels,
                }
            )
            start = end
        return predictions
