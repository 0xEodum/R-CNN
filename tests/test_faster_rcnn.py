import torch

from src.models.faster_rcnn import FasterRCNN


def test_faster_rcnn_training_forward_returns_losses_with_gradients() -> None:
    model = FasterRCNN(backbone_channels=16, rpn_pre_nms_top_n=30, rpn_post_nms_top_n=8, hidden_dim=32)
    images = torch.randn(1, 3, 128, 128)
    targets = [{"boxes": torch.tensor([[32.0, 32.0, 80.0, 80.0]]), "labels": torch.ones(1, dtype=torch.int64)}]

    losses = model(images, targets)
    total_loss = sum(losses.values())
    total_loss.backward()

    assert set(losses) == {"rpn_objectness", "rpn_box_reg", "detector_cls", "detector_box_reg"}
    assert total_loss.requires_grad
    assert any(param.grad is not None for param in model.parameters())


def test_faster_rcnn_supports_stride_8_rpn_geometry() -> None:
    model = FasterRCNN(
        backbone_channels=16,
        backbone_stride=8,
        anchor_sizes=(8, 16, 32),
        rpn_pre_nms_top_n=30,
        rpn_post_nms_top_n=8,
        hidden_dim=32,
    )
    images = torch.randn(1, 3, 128, 128)
    targets = [{"boxes": torch.tensor([[32.0, 32.0, 80.0, 80.0]]), "labels": torch.ones(1, dtype=torch.int64)}]

    losses = model(images, targets)

    assert model.backbone.stride == 8
    assert model.rpn.anchor_generator.stride == 8
    assert sum(losses.values()).requires_grad


def test_faster_rcnn_eval_returns_predictions() -> None:
    model = FasterRCNN(backbone_channels=16, rpn_pre_nms_top_n=20, rpn_post_nms_top_n=5, hidden_dim=32)
    model.eval()

    with torch.no_grad():
        predictions = model(torch.randn(1, 3, 128, 128))

    assert len(predictions) == 1
    assert set(predictions[0]) == {"boxes", "scores", "labels"}
    assert predictions[0]["boxes"].shape[1] == 4


def test_faster_rcnn_supports_soft_nms_postprocessing() -> None:
    model = FasterRCNN(
        backbone_channels=16,
        hidden_dim=32,
        rpn_pre_nms_top_n=20,
        rpn_post_nms_top_n=5,
        postprocess_nms="soft",
    )
    model.eval()

    with torch.no_grad():
        predictions = model(torch.randn(1, 3, 128, 128))

    assert model.postprocess_nms == "soft"
    assert len(predictions) == 1
