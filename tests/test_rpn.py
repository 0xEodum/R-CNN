import torch

from src.models.anchors import AnchorGenerator
from src.models.rpn import RegionProposalNetwork, RPNHead


def test_rpn_head_outputs_anchor_logits_and_deltas() -> None:
    head = RPNHead(in_channels=16, num_anchors=3)
    features = torch.randn(2, 16, 4, 5)

    objectness, deltas = head(features)

    assert objectness.shape == (2, 3, 4, 5)
    assert deltas.shape == (2, 12, 4, 5)


def test_rpn_returns_proposals_and_trainable_losses() -> None:
    generator = AnchorGenerator(sizes=(32,), ratios=(1.0,), stride=16)
    rpn = RegionProposalNetwork(
        in_channels=8,
        anchor_generator=generator,
        pre_nms_top_n=20,
        post_nms_top_n=5,
        batch_size_per_image=16,
    )
    features = torch.randn(1, 8, 4, 4, requires_grad=True)
    targets = [{"boxes": torch.tensor([[16.0, 16.0, 48.0, 48.0]])}]

    proposals, losses = rpn(features, targets=targets, image_size=(64, 64))
    total_loss = losses["rpn_objectness"] + losses["rpn_box_reg"]
    total_loss.backward()

    assert len(proposals) == 1
    assert proposals[0].shape == (5, 4)
    assert set(losses) == {"rpn_objectness", "rpn_box_reg"}
    assert features.grad is not None
