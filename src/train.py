from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.gwhd_dataset import GWHDDetectionDataset, collate_detection_batch
from src.models.faster_rcnn import FasterRCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-batch Faster R-CNN training smoke for GWHD.")
    parser.add_argument("--data-root", type=Path, default=Path("../gwhd_2021"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def move_targets_to_device(targets: list[dict[str, object]], device: torch.device) -> list[dict[str, object]]:
    moved: list[dict[str, object]] = []
    for target in targets:
        moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return moved


def train_one_smoke_step(args: argparse.Namespace) -> dict[str, float]:
    device = resolve_device(args.device)
    dataset = GWHDDetectionDataset(args.data_root, split=args.split, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
    )
    model = FasterRCNN().to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    last_losses: dict[str, torch.Tensor] = {}
    try:
        for step, (images, targets) in enumerate(loader):
            if step >= args.max_steps:
                break
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_losses = {name: value.detach() for name, value in losses.items()}
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM during Faster R-CNN smoke: batch_size={args.batch_size}, "
            f"image_size={args.image_size}, device={device}"
        ) from exc

    return {name: float(value.cpu()) for name, value in last_losses.items()}


def main() -> None:
    args = parse_args()
    losses = train_one_smoke_step(args)
    print({key: round(value, 6) for key, value in losses.items()})


if __name__ == "__main__":
    main()
