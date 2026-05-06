from argparse import Namespace
from pathlib import Path

import torch

from src.train import (
    StepTimer,
    TrainingLogger,
    create_data_loader,
    limit_dataset,
    should_use_amp,
    sync_device_for_timing,
)


def test_training_logger_writes_console_and_csv(tmp_path: Path, capsys) -> None:
    logger = TrainingLogger(csv_path=tmp_path / "metrics.csv", enabled=True)
    logger.log(
        {
            "step": 1,
            "loss": 2.5,
            "lr": 0.01,
            "samples_per_sec": 8.0,
            "data_ms": 3.0,
            "train_ms": 12.0,
            "gpu_mem_mb": 128.0,
            "rpn_objectness": 0.7,
        }
    )
    logger.close()

    output = capsys.readouterr().out
    csv_text = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert "step=1" in output
    assert "samples/s=8.00" in output
    assert "gpu_mem=128MB" in output
    assert "rpn_objectness" in csv_text
    assert "2.5" in csv_text


def test_step_timer_tracks_data_and_train_time_without_cuda_sync() -> None:
    timer = StepTimer(device=torch.device("cpu"))

    timer.mark_batch_loaded()
    timer.mark_step_done(batch_size=2)
    metrics = timer.metrics()

    assert metrics["data_ms"] >= 0.0
    assert metrics["train_ms"] >= 0.0
    assert metrics["samples_per_sec"] > 0.0


def test_step_timer_keeps_latest_step_metrics_when_logging_is_sparse() -> None:
    timer = StepTimer(device=torch.device("cpu"))

    timer.mark_batch_loaded()
    timer.mark_step_done(batch_size=1)
    first = timer.metrics()
    timer.mark_batch_loaded()
    timer.mark_step_done(batch_size=1)
    second = timer.metrics()

    assert second["step_ms"] >= 0.0
    assert second["step_ms"] < first["step_ms"] + 1000.0


def test_create_data_loader_enables_worker_options_only_when_valid() -> None:
    dataset = [(torch.zeros(3, 8, 8), {"boxes": torch.empty((0, 4))})]
    args = Namespace(batch_size=1, num_workers=0, prefetch_factor=4, shuffle=False)

    loader = create_data_loader(dataset, args=args, device=torch.device("cpu"))

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.persistent_workers is False


def test_create_data_loader_respects_shuffle_flag() -> None:
    dataset = [(torch.zeros(3, 8, 8), {"boxes": torch.empty((0, 4))}) for _ in range(3)]
    args = Namespace(batch_size=1, num_workers=0, prefetch_factor=2, shuffle=True)

    loader = create_data_loader(dataset, args=args, device=torch.device("cpu"))

    assert loader.sampler.__class__.__name__ == "RandomSampler"


def test_should_use_amp_only_for_cuda_when_enabled() -> None:
    assert should_use_amp(torch.device("cpu"), enabled=True) is False
    assert should_use_amp(torch.device("cuda"), enabled=False) is False
    assert should_use_amp(torch.device("cuda"), enabled=True) is True


def test_sync_device_for_timing_is_noop_on_cpu() -> None:
    assert sync_device_for_timing(torch.device("cpu"), enabled=True) is False
    assert sync_device_for_timing(torch.device("cpu"), enabled=False) is False


def test_limit_dataset_keeps_prefix_when_requested() -> None:
    dataset = [1, 2, 3]

    limited = limit_dataset(dataset, limit=2)
    unchanged = limit_dataset(dataset, limit=0)

    assert len(limited) == 2
    assert len(unchanged) == 3
