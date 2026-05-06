# ClassicDetection Project Summary

Updated: 2026-05-06

## Goal

This project implements a Faster R-CNN style wheat-head detector from scratch in PyTorch for the GWHD 2021 dataset. The current sprint goal was to improve recognition accuracy through algorithmic changes, not only parameter tuning, and to preserve enough evidence for fair comparisons between model versions.

## Dataset

Dataset root:

```text
D:\Study_tasks\NN_Img\gwhd_2021
```

Format:

- Images are stored under `images/`.
- Splits are provided as CSV files: `competition_train.csv`, `competition_val.csv`, `competition_test.csv`.
- CSV schema is `image_name,BoxesString,domain`.
- `BoxesString` contains pixel-space `x1 y1 x2 y2` boxes separated by semicolons.
- `BoxesString=no_box` is treated as an empty target.
- This is a single-class detection problem: wheat heads.
- Images are 1024x1024 originally; current best training uses `image_size=256`.

Observed split scale:

| Split | Images | Empty Images | Mean Boxes/Image | Max Boxes |
|---|---:|---:|---:|---:|
| train | 3657 | 50 | 44.76 | 129 |
| val | 1476 | 28 | 30.05 | 179 |
| test | 1382 | 47 | 48.79 | 190 |

## Current Code Structure

Core modules:

- `src/data/gwhd_dataset.py`: GWHD CSV parsing, image loading, target construction, detection collate function.
- `src/models/backbone.py`: compact CNN backbone with configurable output stride.
- `src/models/anchors.py`: anchor grid generation.
- `src/models/rpn.py`: region proposal network, anchor matching, RPN losses, proposal filtering.
- `src/models/roi_pool.py`: RoI Align style pooling using `grid_sample`.
- `src/models/detector_head.py`: classification and box regression head.
- `src/models/faster_rcnn.py`: end-to-end detector wrapper and postprocessing.
- `src/models/box_ops.py`: IoU, encode/decode boxes, clipping, hard NMS, greedy fallback, soft-NMS.
- `src/train.py`: training, validation metrics, logging, checkpointing.
- `src/eval.py`: precision/recall/F1 detection matching.
- `src/infer.py`: checkpoint-based single-image inference and annotated image rendering.

Tests:

```text
43 passed
```

Validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\
```

## Important Implementation Decisions

### 1. Scratch Faster R-CNN, not torchvision detection

The architecture is implemented directly in local modules. `torchvision.models.detection` is not used. Low-level `torchvision.ops.nms` is allowed as a fused performance primitive, with a local greedy fallback in `box_ops.py`.

### 2. Single-class detector

The model uses two classifier outputs:

- class `0`: background
- class `1`: wheat head

GWHD labels are collapsed to foreground label `1`.

### 3. Fixed-size resizing for the current baseline

Current training resizes images to `256x256`. Ground-truth boxes are scaled from original 1024 coordinates into model coordinates. Inference scales predicted boxes back to original image size before rendering.

### 4. Validation protocol

The current comparison metric is detection F1 on the validation split using:

- IoU threshold: `0.3`
- score threshold: `0.3`
- validation window: first `50` validation batches
- batch size: `1`

This is not full COCO mAP. It is a practical, fixed validation protocol used to compare algorithmic iterations quickly and consistently.

### 5. Checkpoint policy

Training writes:

- `last.pt`: last step checkpoint
- `best.pt`: best checkpoint by validation F1
- `metrics.csv`: training/validation metrics

Checkpoints contain:

- model state
- optimizer state
- step
- validation metric
- model config

## Monitoring and Performance Decisions

The trainer logs:

- total loss
- RPN objectness loss
- RPN box regression loss
- detector classification loss
- detector box regression loss
- learning rate
- samples/sec
- data load time
- train step time
- GPU memory
- validation precision/recall/F1 when evaluation runs

DataLoader tuning:

- `num_workers=4`
- `prefetch_factor=2`
- pinned memory on CUDA

Important optimization:

- The original Python-loop NMS path created many tiny operations and misleading GPU activity.
- Switching to fused `torchvision.ops.nms` improved training throughput materially.

## Experiment History

### Initial working model

The first trainable model proved end-to-end correctness: dataset loading, forward pass, backward pass, and optimizer step all worked on GWHD.

### Optimization and monitoring sprint

After replacing Python-loop NMS with fused NMS, short-run throughput improved from roughly `4-5 samples/s` to around `40-60 samples/s` on the measured 256px workload.

AMP and channels-last were tested but were slower on this small, proposal-heavy workload, so they remain optional but are not part of the best baseline.

### Parameter-level training baseline

Best stride-16 full-epoch baseline:

```text
runs\best_256_lr1e3_full_epoch\best.pt
```

Config:

```text
image_size=256
lr=1e-3
backbone_stride=16
anchor_sizes=16,32,64
rpn_post_nms_top_n=200
detections_per_image=150
postprocess_nms=hard
```

Best validation result on the fixed protocol:

```text
step=2000
precision=0.3727
recall=0.5055
F1@IoU0.3=0.4291
```

This is the baseline for the latest algorithmic comparison.

### Algorithmic change: stride-8 proposal geometry

Problem:

At `image_size=256`, many wheat heads become very small in model coordinates. With stride-16 features, anchor centers are placed on a coarse 16-pixel grid, which is a poor match for small, dense objects.

Decision:

Add configurable `backbone_stride=8` and use a denser RPN feature map. This is a mechanism change: the proposal geometry changed, not just the optimizer settings.

Implementation:

- `SmallBackbone(output_stride=8)` returns a stride-8 feature map.
- Faster R-CNN passes the actual backbone stride into the anchor generator and RoI pooling scale.
- Smaller anchors were added for the denser grid: `8,16,32,64`.

Best stride-8 full-epoch run:

```text
runs\stride8_256_lr1e3_full_epoch\best.pt
```

Config:

```text
image_size=256
lr=1e-3
backbone_stride=8
anchor_sizes=8,16,32,64
rpn_post_nms_top_n=300
detections_per_image=180
postprocess_nms=hard
```

Best validation result on the same fixed protocol:

```text
step=3657
precision=0.4618
recall=0.6140
F1@IoU0.3=0.5271
```

Relative improvement over the stride-16 baseline:

```text
(0.5271 - 0.4291) / 0.4291 = 22.8%
```

This clears the requested 10% improvement target.

### Soft-NMS experiment

Soft-NMS was implemented because GWHD is crowded and adjacent heads can be suppressed by hard NMS. It was tested on the stride-8 checkpoint without retraining.

Result:

- Best hard-NMS stride-8 checkpoint at score threshold `0.3`: `F1=0.4582` in the threshold sweep context.
- Soft-NMS did not improve this checkpoint; it peaked around `F1=0.4577`.

Decision:

Soft-NMS remains implemented and tested, but the current best checkpoint uses hard NMS.

## Current Best Model

Checkpoint:

```text
runs\stride8_256_lr1e3_full_epoch\best.pt
```

Metric:

```text
F1@IoU0.3 = 0.5271
precision = 0.4618
recall = 0.6140
validation slice = first 50 validation batches
score threshold = 0.3
```

Rendered inference artifact:

```text
runs\stride8_256_lr1e3_full_epoch\inference_val_sample.png
```

Detection JSON:

```text
runs\stride8_256_lr1e3_full_epoch\inference_val_sample.json
```

The rendered sample produced `52` detections at score threshold `0.3`.

## Reproduction Commands

### Train the current best model

```powershell
.\.venv\Scripts\python.exe -m src.train `
  --data-root ..\gwhd_2021 `
  --image-size 256 `
  --batch-size 1 `
  --max-steps 3657 `
  --num-workers 4 `
  --prefetch-factor 2 `
  --log-interval 250 `
  --eval-interval 500 `
  --val-max-batches 50 `
  --eval-iou-thresh 0.3 `
  --eval-score-thresh 0.3 `
  --run-dir runs\stride8_256_lr1e3_full_epoch `
  --metrics-csv runs\stride8_256_lr1e3_full_epoch\metrics.csv `
  --device auto `
  --sync-timing `
  --backbone-stride 8 `
  --rpn-post-nms-top-n 300 `
  --detections-per-image 180 `
  --anchor-sizes 8,16,32,64 `
  --lr 1e-3
```

### Run inference with the current best checkpoint

```powershell
.\.venv\Scripts\python.exe -m src.infer `
  --checkpoint runs\stride8_256_lr1e3_full_epoch\best.pt `
  --image ..\gwhd_2021\images\e6b6a900e5c54cd5d8b0649768c361512cff1813409319eba26da5c7f47bb2e6.png `
  --output-image runs\stride8_256_lr1e3_full_epoch\inference_val_sample.png `
  --output-json runs\stride8_256_lr1e3_full_epoch\inference_val_sample.json `
  --image-size 256 `
  --score-thresh 0.3 `
  --device auto
```

## Known Limitations

- The metric is a fixed F1 protocol, not full COCO mAP.
- Validation uses a bounded slice of 50 batches for fast iteration.
- Training currently uses batch size 1.
- The model is still a compact scratch detector, not a mature production architecture.
- Some false positives remain visible in rendered samples, especially on wheat-like background textures.
- The current model does not use feature pyramids, data augmentation, focal loss, or multi-scale training.

## Recommended Next Comparisons

To keep comparisons fair:

1. Keep the same validation protocol unless explicitly changing the benchmark.
2. Report precision, recall, and F1 together; improvements can trade precision against recall.
3. Preserve `metrics.csv`, `best.pt`, `last.pt`, and rendered samples for each run.
4. Compare against the current best baseline:

```text
runs\stride8_256_lr1e3_full_epoch\best.pt
F1@IoU0.3 = 0.5271
```

Promising next mechanism changes:

- Add a lightweight FPN or dual-resolution feature path.
- Improve proposal/ROI sampling balance for dense objects.
- Add data augmentation carefully: flips, mild color jitter, and crop/scale policies that preserve box validity.
- Try focal loss for detector classification if false positives remain high.
- Add full validation/mAP reporting once iteration stabilizes.
