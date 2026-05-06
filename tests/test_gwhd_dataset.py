from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.gwhd_dataset import (
    GWHDDetectionDataset,
    YOLOPolygonDetectionDataset,
    build_detection_dataset,
    collate_detection_batch,
    parse_boxes_string,
    parse_yolo_annotation_line,
    read_yolo_class_names,
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


def test_dataset_can_apply_horizontal_flip_to_image_and_boxes(tmp_path: Path) -> None:
    data_root = tmp_path / "gwhd"
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True)
    _write_image(images_dir / "sample.png")
    (data_root / "competition_train.csv").write_text(
        "image_name,BoxesString,domain\n"
        "sample.png,100 200 300 400,domain_a\n",
        encoding="utf-8",
    )

    dataset = GWHDDetectionDataset(data_root, split="train", image_size=256, hflip_prob=1.0)
    _, target = dataset[0]

    assert torch.allclose(target["boxes"], torch.tensor([[181.0, 50.0, 231.0, 100.0]]))


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


def test_parse_yolo_annotation_line_converts_polygon_to_pixel_box() -> None:
    parsed = parse_yolo_annotation_line("2 0.25 0.50 0.75 0.50 0.75 1.0 0.25 1.0", image_width=640, image_height=640)

    assert parsed is not None
    label, box = parsed
    assert label == 3
    assert torch.equal(box, torch.tensor([160.0, 320.0, 480.0, 640.0]))


def test_yolo_polygon_dataset_loads_split_labels_and_class_names(tmp_path: Path) -> None:
    data_root = tmp_path / "coffee"
    images_dir = data_root / "train" / "images"
    labels_dir = data_root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (640, 640), color=(12, 34, 56)).save(images_dir / "sample.jpg")
    (labels_dir / "sample.txt").write_text(
        "0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"
        "4 0.5 0.5 0.9 0.5 0.9 0.9 0.5 0.9\n",
        encoding="utf-8",
    )
    (data_root / "data.yaml").write_text(
        "nc: 5\nnames: ['dry', 'overripe', 'ripe', 'semi_ripe', 'unripe']\n",
        encoding="utf-8",
    )

    dataset = YOLOPolygonDetectionDataset(data_root, split="train", image_size=320)
    image, target = dataset[0]

    assert read_yolo_class_names(data_root) == ("dry", "overripe", "ripe", "semi_ripe", "unripe")
    assert image.shape == (3, 320, 320)
    assert torch.allclose(target["boxes"], torch.tensor([[32.0, 32.0, 96.0, 128.0], [160.0, 160.0, 288.0, 288.0]]))
    assert torch.equal(target["labels"], torch.tensor([1, 5], dtype=torch.int64))


def test_build_detection_dataset_auto_detects_yolo_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "coffee"
    (data_root / "valid" / "images").mkdir(parents=True)
    (data_root / "valid" / "labels").mkdir(parents=True)
    Image.new("RGB", (640, 640), color=(255, 255, 255)).save(data_root / "valid" / "images" / "sample.jpg")
    (data_root / "valid" / "labels" / "sample.txt").write_text("", encoding="utf-8")
    (data_root / "data.yaml").write_text("nc: 1\nnames: ['dry']\n", encoding="utf-8")

    dataset = build_detection_dataset(data_root, split="val", image_size=640, dataset_format="auto")

    assert isinstance(dataset, YOLOPolygonDetectionDataset)
    assert len(dataset) == 1
