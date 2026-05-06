import torch

from src.models.box_ops import (
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    greedy_nms,
    nms,
    soft_nms,
)


def test_box_iou_matches_known_overlap() -> None:
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])

    iou = box_iou(boxes1, boxes2)

    assert torch.allclose(iou, torch.tensor([[25.0 / 175.0, 0.0]]))


def test_encode_decode_boxes_round_trip() -> None:
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 30.0, 30.0]])
    boxes = torch.tensor([[1.0, 2.0, 12.0, 16.0], [12.0, 8.0, 28.0, 34.0]])

    deltas = encode_boxes(boxes, anchors)
    decoded = decode_boxes(deltas, anchors)

    assert torch.allclose(decoded, boxes, atol=1e-4)


def test_clip_boxes_to_image_clamps_coordinates() -> None:
    boxes = torch.tensor([[-5.0, 2.0, 30.0, 40.0]])

    clipped = clip_boxes_to_image(boxes, (20, 25))

    assert torch.equal(clipped, torch.tensor([[0.0, 2.0, 25.0, 20.0]]))


def test_nms_keeps_highest_scoring_non_overlapping_boxes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    scores = torch.tensor([0.8, 0.9, 0.7])

    keep = nms(boxes, scores, iou_threshold=0.5)

    assert torch.equal(keep, torch.tensor([1, 2]))


def test_fused_nms_matches_greedy_fallback_on_cpu() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [12.0, 12.0, 20.0, 20.0],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95])

    assert torch.equal(nms(boxes, scores, 0.3), greedy_nms(boxes, scores, 0.3))


def test_soft_nms_keeps_overlapping_boxes_with_decayed_scores() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    scores = torch.tensor([0.9, 0.85, 0.7])

    keep, decayed_scores = soft_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.05)

    assert keep.shape[0] == 3
    assert 1 in keep.tolist()
    assert decayed_scores[keep == 1].item() < 0.85
