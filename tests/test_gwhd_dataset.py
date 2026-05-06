from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.gwhd_dataset import (
    GWHDDetectionDataset,
    collate_detection_batch,
    parse_boxes_string,
)


def _write_image(path: Path, color: tuple[int, int, int] = (12, 34, 56)) -> None:
    Image.new("RGB", (1024, 1024), color=color).save(path)


def test_parse_boxes_string_returns_pixel_xyxy_boxes() -> None:
    boxes = parse_boxes_string("10 20 30 40;0 0 1024 1024")

    assert torch.equal(
        boxes,
        torch.tensor([[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 1024.0, 1024.0]]),
    )


def test_parse_boxes_string_treats_no_box_as_empty() -> None:
    boxes = parse_boxes_string("no_box")

    assert boxes.shape == (0, 4)
    assert boxes.dtype == torch.float32


def test_dataset_loads_sorted_rows_and_scales_boxes(tmp_path: Path) -> None:
    data_root = tmp_path / "gwhd"
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True)
    _write_image(images_dir / "b.png")
    _write_image(images_dir / "a.png")
    (data_root / "competition_train.csv").write_text(
        "image_name,BoxesString,domain\n"
        "b.png,100 200 300 400,domain_b\n"
        "a.png,0 0 1024 512;512 512 1024 1024,domain_a\n",
        encoding="utf-8",
    )

    dataset = GWHDDetectionDataset(data_root, split="train", image_size=256)
    image, target = dataset[0]

    assert target["image_id"] == "a.png"
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
    assert torch.allclose(
        target["boxes"],
        torch.tensor([[0.0, 0.0, 256.0, 128.0], [128.0, 128.0, 256.0, 256.0]]),
    )
    assert torch.equal(target["labels"], torch.ones(2, dtype=torch.int64))


def test_dataset_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="competition_train.csv"):
        GWHDDetectionDataset(tmp_path, split="train")


def test_collate_detection_batch_keeps_variable_targets() -> None:
    batch = [
        (torch.zeros(3, 8, 8), {"boxes": torch.zeros(1, 4)}),
        (torch.ones(3, 8, 8), {"boxes": torch.zeros(3, 4)}),
    ]

    images, targets = collate_detection_batch(batch)

    assert images.shape == (2, 3, 8, 8)
    assert [target["boxes"].shape[0] for target in targets] == [1, 3]
