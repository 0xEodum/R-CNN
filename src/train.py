from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.data.gwhd_dataset import GWHDDetectionDataset, collate_detection_batch
from src.eval import evaluate_predictions
from src.models.faster_rcnn import FasterRCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scratch Faster R-CNN model on GWHD.")
    parser.add_argument("--data-root", type=Path, default=Path("../gwhd_2021"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--metrics-csv", type=Path, default=Path("runs/train_metrics.csv"))
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--sync-timing", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/full_scale"))
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=10)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--eval-score-thresh", type=float, default=0.2)
    parser.add_argument("--eval-iou-thresh", type=float, default=0.3)
    parser.add_argument("--backbone-channels", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--rpn-pre-nms-top-n", type=int, default=600)
    parser.add_argument("--rpn-post-nms-top-n", type=int, default=100)
    parser.add_argument("--detections-per-image", type=int, default=100)
    parser.add_argument("--anchor-sizes", default="16,32,64")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
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


def should_use_amp(device: torch.device, enabled: bool) -> bool:
    return enabled and device.type == "cuda"


def sync_device_for_timing(device: torch.device, enabled: bool) -> bool:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)
        return True
    return False


def create_data_loader(dataset: object, args: argparse.Namespace, device: torch.device) -> DataLoader:
    num_workers = max(0, int(args.num_workers))
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": bool(getattr(args, "shuffle", not getattr(args, "no_shuffle", False))),
        "num_workers": num_workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def parse_anchor_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes:
        raise ValueError("--anchor-sizes must contain at least one integer")
    return sizes


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "backbone_channels": args.backbone_channels,
        "hidden_dim": args.hidden_dim,
        "rpn_pre_nms_top_n": args.rpn_pre_nms_top_n,
        "rpn_post_nms_top_n": args.rpn_post_nms_top_n,
        "anchor_sizes": parse_anchor_sizes(args.anchor_sizes),
        "score_thresh": args.score_thresh,
        "detections_per_image": args.detections_per_image,
    }


def build_model(args: argparse.Namespace) -> FasterRCNN:
    return FasterRCNN(**model_config_from_args(args))


def save_checkpoint(
    path: Path,
    *,
    model: FasterRCNN,
    optimizer: torch.optim.Optimizer,
    step: int,
    metric: float,
    model_config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "metric": metric,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def limit_dataset(dataset: object, limit: int) -> object:
    if limit <= 0:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))  # type: ignore[arg-type]


class StepTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._last_step_end = time.perf_counter()
        self._batch_loaded = self._last_step_end
        self._step_done = self._last_step_end
        self._batch_size = 1
        self._latest_metrics = {
            "data_ms": 0.0,
            "train_ms": 0.0,
            "step_ms": 0.0,
            "samples_per_sec": 0.0,
        }

    def mark_batch_loaded(self) -> None:
        self._batch_loaded = time.perf_counter()

    def mark_step_done(self, batch_size: int) -> None:
        self._step_done = time.perf_counter()
        self._batch_size = batch_size
        data_s = max(0.0, self._batch_loaded - self._last_step_end)
        train_s = max(1e-9, self._step_done - self._batch_loaded)
        total_s = max(1e-9, self._step_done - self._last_step_end)
        self._latest_metrics = {
            "data_ms": data_s * 1000.0,
            "train_ms": train_s * 1000.0,
            "step_ms": total_s * 1000.0,
            "samples_per_sec": self._batch_size / total_s,
        }
        self._last_step_end = self._step_done

    def metrics(self) -> dict[str, float]:
        return dict(self._latest_metrics)


class TrainingLogger:
    def __init__(self, csv_path: Path | None, enabled: bool = True) -> None:
        self.csv_path = csv_path
        self.enabled = enabled
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = csv_path.open("w", encoding="utf-8", newline="")

    def log(self, metrics: dict[str, float]) -> None:
        if self.enabled:
            print(self._format_console(metrics), flush=True)
        if self._handle is not None:
            if self._writer is None:
                self._fieldnames = [
                    "step",
                    "lr",
                    "data_ms",
                    "train_ms",
                    "step_ms",
                    "samples_per_sec",
                    "loss",
                    "rpn_objectness",
                    "rpn_box_reg",
                    "detector_cls",
                    "detector_box_reg",
                    "gpu_mem_mb",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                    "val_tp",
                    "val_fp",
                    "val_fn",
                ]
                self._writer = csv.DictWriter(self._handle, fieldnames=self._fieldnames)
                self._writer.writeheader()
            assert self._writer is not None
            assert self._fieldnames is not None
            self._writer.writerow({key: metrics.get(key, "") for key in self._fieldnames})
            self._handle.flush()

    @staticmethod
    def _format_console(metrics: dict[str, float]) -> str:
        loss_parts = [
            f"{name}={value:.4f}"
            for name, value in metrics.items()
            if name not in {"step", "loss", "lr", "samples_per_sec", "data_ms", "train_ms", "step_ms", "gpu_mem_mb"}
        ]
        prefix = (
            f"step={int(metrics['step'])} loss={metrics['loss']:.4f} "
            f"lr={metrics['lr']:.2e} samples/s={metrics['samples_per_sec']:.2f} "
            f"data={metrics['data_ms']:.1f}ms train={metrics['train_ms']:.1f}ms"
        )
        if "gpu_mem_mb" in metrics:
            prefix += f" gpu_mem={metrics['gpu_mem_mb']:.0f}MB"
        return " ".join([prefix, *loss_parts])

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def gpu_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def detached_loss_metrics(losses: dict[str, torch.Tensor], total_loss: torch.Tensor) -> dict[str, float]:
    output = {"loss": float(total_loss.detach().cpu())}
    output.update({name: float(value.detach().cpu()) for name, value in losses.items()})
    return output


@torch.no_grad()
def evaluate_model(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        predictions = model(images)
        metrics = evaluate_predictions(
            predictions,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        total["tp"] += metrics["tp"]
        total["fp"] += metrics["fp"]
        total["fn"] += metrics["fn"]
    precision = total["tp"] / max(1.0, total["tp"] + total["fp"])
    recall = total["tp"] / max(1.0, total["tp"] + total["fn"])
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    if was_training:
        model.train()
    return {
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_tp": total["tp"],
        "val_fp": total["fp"],
        "val_fn": total["fn"],
    }


def train_one_smoke_step(args: argparse.Namespace) -> dict[str, float]:
    device = resolve_device(args.device)
    dataset = GWHDDetectionDataset(args.data_root, split=args.split, image_size=args.image_size)
    dataset = limit_dataset(dataset, args.train_limit)
    loader = create_data_loader(dataset, args=args, device=device)
    val_loader = None
    if args.eval_interval > 0:
        val_dataset = GWHDDetectionDataset(args.data_root, split="val", image_size=args.image_size)
        val_args = copy.copy(args)
        val_args.shuffle = False
        val_loader = create_data_loader(val_dataset, args=val_args, device=device)
    model_config = model_config_from_args(args)
    model = FasterRCNN(**model_config).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=should_use_amp(device, args.amp))
    logger = TrainingLogger(args.metrics_csv, enabled=args.log_interval > 0)
    timer = StepTimer(device)
    best_metric = -1.0
    last_step = 0

    last_losses: dict[str, torch.Tensor] = {}
    try:
        for step, (images, targets) in enumerate(loader):
            if step >= args.max_steps:
                break
            last_step = step + 1
            timer.mark_batch_loaded()
            images = images.to(device, non_blocking=True)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = move_targets_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=should_use_amp(device, args.amp)):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            last_losses = {name: value.detach() for name, value in losses.items()}
            sync_device_for_timing(device, args.sync_timing)
            timer.mark_step_done(batch_size=images.shape[0])

            should_log = args.log_interval > 0 and ((step + 1) % args.log_interval == 0 or step + 1 == args.max_steps)
            eval_metrics: dict[str, float] = {}
            if val_loader is not None and ((step + 1) % args.eval_interval == 0 or step + 1 == args.max_steps):
                eval_metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    max_batches=args.val_max_batches,
                    iou_threshold=args.eval_iou_thresh,
                    score_threshold=args.eval_score_thresh,
                )
                current_metric = eval_metrics["val_f1"]
                save_checkpoint(
                    args.run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    metric=current_metric,
                    model_config=model_config,
                )
                if current_metric > best_metric:
                    best_metric = current_metric
                    save_checkpoint(
                        args.run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step + 1,
                        metric=current_metric,
                        model_config=model_config,
                    )
            if should_log:
                metrics = {"step": float(step + 1), "lr": optimizer.param_groups[0]["lr"]}
                metrics.update(timer.metrics())
                metrics.update(detached_loss_metrics(losses, loss))
                metrics.update(eval_metrics)
                memory_mb = gpu_memory_mb(device)
                if memory_mb is not None:
                    metrics["gpu_mem_mb"] = memory_mb
                logger.log(metrics)
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM during Faster R-CNN smoke: batch_size={args.batch_size}, "
            f"image_size={args.image_size}, num_workers={args.num_workers}, device={device}"
        ) from exc
    finally:
        if last_step > 0:
            save_checkpoint(
                args.run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                step=last_step,
                metric=best_metric,
                model_config=model_config,
            )
        logger.close()

    return {name: float(value.cpu()) for name, value in last_losses.items()}


def main() -> None:
    args = parse_args()
    losses = train_one_smoke_step(args)
    print({key: round(value, 6) for key, value in losses.items()})


if __name__ == "__main__":
    main()
