from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.gwhd_dataset import build_detection_dataset, collate_detection_batch
from src.infer import load_model_from_checkpoint
from src.train import evaluate_model, resolve_device


def parse_score_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not thresholds:
        raise ValueError("--score-thresholds must contain at least one value")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved detector checkpoint on a GWHD split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("../gwhd_2021"))
    parser.add_argument("--dataset-format", choices=("auto", "gwhd", "yolo"), default="auto")
    parser.add_argument("--split", choices=("train", "val", "valid", "test"), default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--score-thresholds", default="0.3")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    dataset = build_detection_dataset(
        args.data_root,
        split=args.split,
        image_size=args.image_size,
        dataset_format=args.dataset_format,
    )
    num_workers = max(0, int(args.num_workers))
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> list[dict[str, float]]:
    device = resolve_device(args.device)
    model = load_model_from_checkpoint(args.checkpoint, device)
    loader = build_loader(args, device)
    rows: list[dict[str, float]] = []
    for score_threshold in parse_score_thresholds(args.score_thresholds):
        metrics = evaluate_model(
            model,
            loader,
            device,
            max_batches=args.max_batches,
            iou_threshold=args.iou_thresh,
            score_threshold=score_threshold,
        )
        rows.append(
            {
                "score_threshold": score_threshold,
                "precision": metrics["val_precision"],
                "recall": metrics["val_recall"],
                "f1": metrics["val_f1"],
                "tp": metrics["val_tp"],
                "fp": metrics["val_fp"],
                "fn": metrics["val_fn"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["score_threshold", "precision", "recall", "f1", "tp", "fp", "fn"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = evaluate_checkpoint(args)
    for row in rows:
        print(
            "score={score_threshold:.3f} precision={precision:.4f} recall={recall:.4f} "
            "f1={f1:.4f} tp={tp:.0f} fp={fp:.0f} fn={fn:.0f}".format(**row),
            flush=True,
        )
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
