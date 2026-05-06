import torch

from src.models.backbone import SmallBackbone


def test_small_backbone_returns_stride_16_features() -> None:
    model = SmallBackbone(out_channels=64)
    images = torch.randn(2, 3, 256, 256)

    features = model(images)

    assert features.shape == (2, 64, 16, 16)


def test_small_backbone_allows_backward_pass() -> None:
    model = SmallBackbone(out_channels=32)
    images = torch.randn(1, 3, 128, 128, requires_grad=True)

    loss = model(images).square().mean()
    loss.backward()

    assert images.grad is not None
    assert any(param.grad is not None for param in model.parameters())
