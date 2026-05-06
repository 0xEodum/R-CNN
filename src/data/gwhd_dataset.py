from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


SPLIT_TO_CSV = {
    "train": "competition_train.csv",
    "val": "competition_val.csv",
    "test": "competition_test.csv",
}


def parse_boxes_string(boxes_string: str) -> torch.Tensor:
    value = boxes_string.strip()
    if not value or value == "no_box":
        return torch.empty((0, 4), dtype=torch.float32)

    boxes: list[list[float]] = []
    for box_text in value.split(";"):
        coords = box_text.strip().split()
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates per box, got {len(coords)} in {box_text!r}")
        x1, y1, x2, y2 = (float(coord) for coord in coords)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


@dataclass(frozen=True)
class GWHDRecord:
    image_name: str
    boxes_string: str
    domain: str


class GWHDDetectionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int | None = 512,
        hflip_prob: float = 0.0,
    ) -> None:
        if split not in SPLIT_TO_CSV:
            valid = ", ".join(sorted(SPLIT_TO_CSV))
            raise ValueError(f"Unknown split {split!r}; expected one of: {valid}")
        if not 0.0 <= hflip_prob <= 1.0:
            raise ValueError("hflip_prob must be between 0.0 and 1.0")

        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.csv_path = self.data_root / SPLIT_TO_CSV[split]
        self.image_size = image_size
        self.hflip_prob = hflip_prob

        if not self.csv_path.exists():
            raise FileNotFoundError(f"GWHD annotation CSV not found: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"GWHD images directory not found: {self.images_dir}")

        self.records = self._read_records(self.csv_path)

    @staticmethod
    def _read_records(csv_path: Path) -> list[GWHDRecord]:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"image_name", "BoxesString", "domain"}
            if set(reader.fieldnames or []) < expected:
                raise ValueError(f"GWHD CSV must contain columns: {sorted(expected)}")
            records = [
                GWHDRecord(
                    image_name=row["image_name"],
                    boxes_string=row["BoxesString"],
                    domain=row["domain"],
                )
                for row in reader
            ]
        return sorted(records, key=lambda record: record.image_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image_path = self.images_dir / record.image_name
        if not image_path.exists():
            raise FileNotFoundError(f"GWHD image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            if self.image_size is not None and image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            width, height = image.size
            image_tensor = _image_to_tensor_bytes(image).float().permute(2, 0, 1) / 255.0

        boxes = parse_boxes_string(record.boxes_string)
        if boxes.numel() > 0:
            scale_x = width / original_width
            scale_y = height / original_height
            scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=torch.float32)
            boxes = boxes * scale
            boxes[:, 0::2].clamp_(min=0.0, max=float(width))
            boxes[:, 1::2].clamp_(min=0.0, max=float(height))
        if self.hflip_prob > 0.0 and torch.rand(()) < self.hflip_prob:
            image_tensor = torch.flip(image_tensor, dims=(2,))
            if boxes.numel() > 0:
                flipped_x1 = float(width) - boxes[:, 2]
                flipped_x2 = float(width) - boxes[:, 0]
                boxes = torch.stack((flipped_x1, boxes[:, 1], flipped_x2, boxes[:, 3]), dim=1)

        target = {
            "boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
            "image_id": record.image_name,
            "domain": record.domain,
            "orig_size": torch.tensor([original_height, original_width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        return image_tensor, target


def _image_to_tensor_bytes(image: Image.Image) -> torch.Tensor:
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(image.height, image.width, 3)


def collate_detection_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch, strict=True)
    return torch.stack(list(images), dim=0), list(targets)
