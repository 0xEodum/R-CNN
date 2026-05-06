import torch

from src.models.faster_rcnn import FasterRCNN
from src.train import build_model


def test_one_optimizer_step_updates_at_least_one_parameter() -> None:
    model = FasterRCNN(backbone_channels=16, rpn_pre_nms_top_n=20, rpn_post_nms_top_n=5, hidden_dim=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    images = torch.randn(1, 3, 128, 128)
    targets = [{"boxes": torch.tensor([[24.0, 24.0, 72.0, 72.0]]), "labels": torch.ones(1, dtype=torch.int64)}]
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    losses = model(images, targets)
    loss = sum(losses.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    changed = [not torch.equal(before[name], param.detach()) for name, param in model.named_parameters()]
    assert any(changed)


def test_build_model_applies_tunable_proposal_parameters() -> None:
    args = type(
        "Args",
        (),
        {
            "backbone_channels": 32,
            "backbone_stride": 8,
            "hidden_dim": 64,
            "rpn_pre_nms_top_n": 111,
            "rpn_post_nms_top_n": 22,
            "anchor_sizes": "16,32",
            "score_thresh": 0.25,
            "detections_per_image": 33,
            "postprocess_nms": "soft",
        },
    )()

    model = build_model(args)

    assert model.backbone.out_channels == 32
    assert model.backbone.stride == 8
    assert model.rpn.pre_nms_top_n == 111
    assert model.rpn.post_nms_top_n == 22
    assert model.rpn.anchor_generator.sizes == (16, 32)
    assert model.score_thresh == 0.25
    assert model.postprocess_nms == "soft"
