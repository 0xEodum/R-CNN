import torch

from src.models.roi_pool import RoIAlignPool


def test_roi_align_pool_returns_fixed_size_features() -> None:
    pool = RoIAlignPool(output_size=(3, 3), spatial_scale=0.25)
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    proposals = [
        torch.tensor([[0.0, 0.0, 32.0, 32.0], [16.0, 16.0, 48.0, 48.0]]),
        torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
    ]

    pooled = pool(features, proposals)
    pooled.mean().backward()

    assert pooled.shape == (3, 8, 3, 3)
    assert features.grad is not None


def test_roi_align_pool_handles_empty_proposals() -> None:
    pool = RoIAlignPool(output_size=(2, 2), spatial_scale=1.0)
    features = torch.randn(1, 4, 8, 8)

    pooled = pool(features, [torch.empty((0, 4))])

    assert pooled.shape == (0, 4, 2, 2)
