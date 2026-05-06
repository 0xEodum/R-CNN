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
