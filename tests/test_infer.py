from pathlib import Path

import torch
from PIL import Image

from src.infer import draw_detections, load_model_from_checkpoint, scale_boxes
from src.models.faster_rcnn import FasterRCNN


def test_scale_boxes_maps_model_coordinates_to_original_size() -> None:
    boxes = torch.tensor([[0.0, 10.0, 128.0, 256.0]])

    scaled = scale_boxes(boxes, model_size=(256, 256), original_size=(1024, 512))

    assert torch.equal(scaled, torch.tensor([[0.0, 40.0, 256.0, 1024.0]]))


def test_draw_detections_writes_annotated_image(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    predictions = {
        "boxes": torch.tensor([[5.0, 5.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]]),
        "scores": torch.tensor([0.9, 0.01]),
    }
    output_path = tmp_path / "annotated.png"

    draw_detections(image, predictions, output_path=output_path, score_threshold=0.1)

    assert output_path.exists()
    assert Image.open(output_path).size == (64, 64)


def test_load_model_from_checkpoint_restores_config(tmp_path: Path) -> None:
    model = FasterRCNN(
        backbone_channels=16,
        hidden_dim=32,
        rpn_pre_nms_top_n=20,
        rpn_post_nms_top_n=5,
        anchor_sizes=(16, 32),
        score_thresh=0.2,
        detections_per_image=10,
    )
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_config": {
                "backbone_channels": 16,
                "hidden_dim": 32,
                "rpn_pre_nms_top_n": 20,
                "rpn_post_nms_top_n": 5,
                "anchor_sizes": (16, 32),
                "score_thresh": 0.2,
                "detections_per_image": 10,
            },
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    restored = load_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert restored.backbone.out_channels == 16
    assert restored.rpn.anchor_generator.sizes == (16, 32)
    assert restored.training is False
