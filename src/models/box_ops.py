from __future__ import annotations

import torch

try:
    from torchvision.ops import nms as _torchvision_nms
except Exception:
    _torchvision_nms = None


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-6)


def encode_boxes(reference_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    anchor_widths = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    anchor_heights = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    ref_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    ref_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    ref_ctr_x = reference_boxes[:, 0] + 0.5 * ref_widths
    ref_ctr_y = reference_boxes[:, 1] + 0.5 * ref_heights

    dx = (ref_ctr_x - anchor_ctr_x) / anchor_widths
    dy = (ref_ctr_y - anchor_ctr_y) / anchor_heights
    dw = torch.log(ref_widths / anchor_widths)
    dh = torch.log(ref_heights / anchor_heights)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    widths = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    heights = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas.unbind(dim=1)
    dw = dw.clamp(max=4.135)
    dh = dh.clamp(max=4.135)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=1)


def clip_boxes_to_image(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    x1 = boxes[:, 0].clamp(min=0, max=width)
    y1 = boxes[:, 1].clamp(min=0, max=height)
    x2 = boxes[:, 2].clamp(min=0, max=width)
    y2 = boxes[:, 3].clamp(min=0, max=height)
    return torch.stack((x1, y1, x2, y2), dim=1)


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return torch.where((widths >= min_size) & (heights >= min_size))[0]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if _torchvision_nms is not None:
        return _torchvision_nms(boxes, scores, iou_threshold)
    return greedy_nms(boxes, scores, iou_threshold)


def greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]

    return torch.stack(keep).to(dtype=torch.long)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    score_threshold: float,
    sigma: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=boxes.device),
            torch.empty((0,), dtype=scores.dtype, device=scores.device),
        )

    working_scores = scores.clone()
    order = working_scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        decay = torch.ones_like(ious)
        overlap = ious > iou_threshold
        decay[overlap] = torch.exp(-((ious[overlap] * ious[overlap]) / sigma))
        working_scores[remaining] = working_scores[remaining] * decay
        remaining = remaining[working_scores[remaining] >= score_threshold]
        order = remaining[working_scores[remaining].argsort(descending=True)]

    keep_tensor = torch.stack(keep).to(dtype=torch.long)
    return keep_tensor, working_scores
