import torch

from src.models.detector_head import DetectorHead


def test_detector_head_outputs_class_logits_and_box_deltas() -> None:
    head = DetectorHead(in_channels=8, pooled_size=(2, 2), hidden_dim=16, num_classes=2)
    pooled = torch.randn(5, 8, 2, 2)

    class_logits, box_deltas = head(pooled)

    assert class_logits.shape == (5, 2)
    assert box_deltas.shape == (5, 2, 4)


def test_detector_head_losses_backpropagate() -> None:
    head = DetectorHead(in_channels=4, pooled_size=(2, 2), hidden_dim=16, num_classes=2)
    pooled = torch.randn(3, 4, 2, 2, requires_grad=True)
    proposals = [torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [4.0, 4.0, 12.0, 12.0]])]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.ones(1, dtype=torch.int64)}]

    class_logits, box_deltas = head(pooled)
    losses = head.compute_loss(class_logits, box_deltas, proposals, targets)
    total_loss = losses["detector_cls"] + losses["detector_box_reg"]
    total_loss.backward()

    assert set(losses) == {"detector_cls", "detector_box_reg"}
    assert pooled.grad is not None


def test_detector_head_uses_ignore_band_for_ambiguous_rois() -> None:
    head = DetectorHead(
        in_channels=4,
        pooled_size=(2, 2),
        hidden_dim=16,
        num_classes=2,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.3,
    )
    proposals = [
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 7.0, 7.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        )
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.ones(1, dtype=torch.int64)}]

    labels, _ = head._assign_targets(proposals, targets, device=torch.device("cpu"))

    assert torch.equal(labels, torch.tensor([1, -1, 0]))


def test_detector_head_samples_balanced_positive_and_negative_rois() -> None:
    head = DetectorHead(
        in_channels=4,
        pooled_size=(2, 2),
        hidden_dim=16,
        num_classes=2,
        batch_size_per_image=4,
        positive_fraction=0.5,
    )
    labels = torch.tensor([1, 1, 1, 0, 0, 0, -1, -1])

    sampled = head._sample_rois(labels, proposal_counts=[labels.numel()])

    assert sampled.numel() == 4
    assert int((labels[sampled] == 1).sum()) == 2
    assert int((labels[sampled] == 0).sum()) == 2
    assert not torch.any(labels[sampled] == -1)


def test_detector_head_assigns_matched_target_classes_for_multiclass_data() -> None:
    head = DetectorHead(in_channels=4, pooled_size=(2, 2), hidden_dim=16, num_classes=6)
    proposals = [
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [80.0, 80.0, 90.0, 90.0],
            ]
        )
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            "labels": torch.tensor([2, 5], dtype=torch.int64),
        }
    ]

    labels, _ = head._assign_targets(proposals, targets, device=torch.device("cpu"))

    assert torch.equal(labels, torch.tensor([2, 5, 0]))


def test_detector_head_uses_configured_class_weights_for_classification_loss() -> None:
    head = DetectorHead(
        in_channels=4,
        pooled_size=(2, 2),
        hidden_dim=16,
        num_classes=3,
        class_weights=(1.0, 2.0, 4.0),
    )
    assert torch.equal(head.class_weights, torch.tensor([1.0, 2.0, 4.0]))


def test_detector_head_balances_positive_samples_across_classes() -> None:
    head = DetectorHead(
        in_channels=4,
        pooled_size=(2, 2),
        hidden_dim=16,
        num_classes=4,
        batch_size_per_image=6,
        positive_fraction=0.5,
        balanced_positive_classes=True,
    )
    labels = torch.tensor([1, 1, 1, 1, 1, 2, 3, 0, 0, 0])

    sampled = head._sample_rois(labels, proposal_counts=[labels.numel()])

    sampled_labels = labels[sampled]
    assert sampled.numel() == 6
    assert int((sampled_labels == 1).sum()) == 1
    assert int((sampled_labels == 2).sum()) == 1
    assert int((sampled_labels == 3).sum()) == 1
    assert int((sampled_labels == 0).sum()) == 3
