from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.box_ops import box_iou


@dataclass(frozen=True)
class DetectionStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: DetectionStats) -> DetectionStats:
        return DetectionStats(tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn)


def match_image_detections(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> DetectionStats:
    pred_boxes = prediction["boxes"].detach().cpu()
    pred_scores = prediction["scores"].detach().cpu()
    pred_labels = prediction.get("labels", torch.ones((pred_boxes.shape[0],), dtype=torch.int64)).detach().cpu()
    target_boxes = target["boxes"].detach().cpu()
    target_labels = target.get("labels", torch.ones((target_boxes.shape[0],), dtype=torch.int64)).detach().cpu()

    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    if target_boxes.numel() == 0:
        return DetectionStats(tp=0, fp=int(pred_boxes.shape[0]), fn=0)
    if pred_boxes.numel() == 0:
        return DetectionStats(tp=0, fp=0, fn=int(target_boxes.shape[0]))

    order = pred_scores.argsort(descending=True)
    ious = box_iou(pred_boxes[order], target_boxes)
    matched_targets = torch.zeros((target_boxes.shape[0],), dtype=torch.bool)
    tp = 0
    fp = 0
    for row in range(ious.shape[0]):
        label_matches = target_labels == pred_labels[order[row]]
        class_ious = torch.where(label_matches, ious[row], torch.zeros_like(ious[row]))
        best_iou, best_idx = class_ious.max(dim=0)
        if best_iou >= iou_threshold and not matched_targets[best_idx]:
            matched_targets[best_idx] = True
            tp += 1
        else:
            fp += 1
    fn = int((~matched_targets).sum().item())
    return DetectionStats(tp=tp, fp=fp, fn=fn)


def evaluate_predictions(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, float]:
    stats = DetectionStats()
    for prediction, target in zip(predictions, targets, strict=True):
        stats += match_image_detections(prediction, target, iou_threshold=iou_threshold, score_threshold=score_threshold)
    precision = stats.tp / max(1, stats.tp + stats.fp)
    recall = stats.tp / max(1, stats.tp + stats.fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": float(stats.tp),
        "fp": float(stats.fp),
        "fn": float(stats.fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
