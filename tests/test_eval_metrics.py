import torch

from src.eval import DetectionStats, evaluate_predictions, match_image_detections


def test_match_image_detections_counts_tp_fp_fn() -> None:
    prediction = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.1]),
        "labels": torch.ones(3, dtype=torch.int64),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 120.0, 120.0]]),
        "labels": torch.ones(2, dtype=torch.int64),
    }

    stats = match_image_detections(prediction, target, iou_threshold=0.5, score_threshold=0.2)

    assert stats == DetectionStats(tp=1, fp=1, fn=1)


def test_evaluate_predictions_accumulates_precision_recall_f1() -> None:
    predictions = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9])},
        {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,))},
    ]
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])},
        {"boxes": torch.tensor([[5.0, 5.0, 10.0, 10.0]])},
    ]

    metrics = evaluate_predictions(predictions, targets, iou_threshold=0.5, score_threshold=0.1)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 4) == 0.6667


def test_empty_predictions_are_naive_zero_recall() -> None:
    metrics = evaluate_predictions(
        [{"boxes": torch.empty((0, 4)), "scores": torch.empty((0,))}],
        [{"boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]])}],
        iou_threshold=0.5,
        score_threshold=0.1,
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_match_image_detections_requires_matching_class_labels() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([2, 1], dtype=torch.int64),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "labels": torch.tensor([1, 1], dtype=torch.int64),
    }

    stats = match_image_detections(prediction, target, iou_threshold=0.5, score_threshold=0.1)

    assert stats == DetectionStats(tp=1, fp=1, fn=1)
