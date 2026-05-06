from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.models.faster_rcnn import FasterRCNN
from src.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run wheat-head detection on one image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> FasterRCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = dict(checkpoint["model_config"])
    if "anchor_sizes" in config:
        config["anchor_sizes"] = tuple(config["anchor_sizes"])
    model = FasterRCNN(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def scale_boxes(boxes: torch.Tensor, model_size: tuple[int, int], original_size: tuple[int, int]) -> torch.Tensor:
    model_h, model_w = model_size
    original_h, original_w = original_size
    scale = torch.tensor(
        [original_w / model_w, original_h / model_h, original_w / model_w, original_h / model_h],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    return boxes * scale


def image_to_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    tensor = data.view(image_size, image_size, 3).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)


def draw_detections(
    image: Image.Image,
    prediction: dict[str, torch.Tensor],
    *,
    output_path: Path,
    score_threshold: float,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    for box, score in zip(boxes, scores, strict=True):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 48, 48), width=3)
        draw.text((x1, max(0.0, y1 - 12.0)), f"{float(score):.2f}", fill=(255, 48, 48))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    model = load_model_from_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        tensor = image_to_tensor(image, args.image_size, device)
        prediction = model(tensor)[0]  # type: ignore[index]
        prediction = {
            "boxes": scale_boxes(
                prediction["boxes"],
                model_size=(args.image_size, args.image_size),
                original_size=(image.height, image.width),
            ),
            "scores": prediction["scores"],
            "labels": prediction["labels"],
        }
        draw_detections(image, prediction, output_path=args.output_image, score_threshold=args.score_thresh)

    keep = prediction["scores"] >= args.score_thresh
    result = {
        "image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "detections": [
            {"box": [round(float(v), 3) for v in box.tolist()], "score": round(float(score), 6)}
            for box, score in zip(prediction["boxes"][keep].detach().cpu(), prediction["scores"][keep].detach().cpu(), strict=True)
        ],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run_inference(parse_args())
    print(f"detections={len(result['detections'])}")


if __name__ == "__main__":
    main()
