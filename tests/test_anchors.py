import torch

from src.models.anchors import AnchorGenerator


def test_anchor_generator_returns_anchors_for_each_feature_location() -> None:
    generator = AnchorGenerator(sizes=(32,), ratios=(1.0,), stride=16)

    anchors = generator.grid_anchors((2, 3), device=torch.device("cpu"))

    assert anchors.shape == (6, 4)
    assert torch.equal(anchors[0], torch.tensor([-8.0, -8.0, 24.0, 24.0]))
    assert torch.equal(anchors[1], torch.tensor([8.0, -8.0, 40.0, 24.0]))


def test_anchor_generator_supports_multiple_sizes_and_ratios() -> None:
    generator = AnchorGenerator(sizes=(16, 32), ratios=(0.5, 1.0), stride=8)

    anchors = generator.grid_anchors((1, 1), device=torch.device("cpu"))

    assert anchors.shape == (4, 4)
    assert torch.all(anchors[:, 2:] > anchors[:, :2])
