from pathlib import Path

import torch

from src.models.faster_rcnn import FasterRCNN
from src.train import save_checkpoint


def test_save_checkpoint_writes_model_config_and_metric(tmp_path: Path) -> None:
    model = FasterRCNN(backbone_channels=16, hidden_dim=32, rpn_pre_nms_top_n=20, rpn_post_nms_top_n=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "last.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=7,
        metric=0.25,
        model_config={
            "backbone_channels": 16,
            "hidden_dim": 32,
            "rpn_pre_nms_top_n": 20,
            "rpn_post_nms_top_n": 5,
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["step"] == 7
    assert checkpoint["metric"] == 0.25
    assert checkpoint["model_config"]["backbone_channels"] == 16
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint
