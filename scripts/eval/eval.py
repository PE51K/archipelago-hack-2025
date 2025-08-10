#!/usr/bin/env python3
"""
Evaluate a trained YOLOv8 model (with or without custom loss)
============================================================

Example
-------
python eval.py \
    --weights runs/train/8_512_custom_loss_sliced_dataset_slice_size_1536_focal_loss_resume_1/weights/best.pt \
    --data data/merged_sliced/data.yml \
    --imgsz 512 --batch 64 --device 0 --split val --plots
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss, FocalLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
import torch.nn.functional as F
import torch
from ultralytics import YOLO


class CustomFocalLoss(FocalLoss):
    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss


class Customv8DetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bce = CustomFocalLoss(alpha=0.3, gamma=1.2)


class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return Customv8DetectionLoss(self)  # Use our custom loss instead of v8DetectionLoss


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CustomDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


import sys, types
main_mod = sys.modules.setdefault("__main__", types.ModuleType("__main__"))
for _cls in (CustomDetectionModel, Customv8DetectionLoss, CustomFocalLoss):
    setattr(main_mod, _cls.__name__, _cls)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', type=str,  required=True,
                   help='Path to *.pt checkpoint to evaluate')
    p.add_argument('--data',    type=str,  required=True,
                   help='Dataset YAML with train/val/test paths')
    p.add_argument('--imgsz',   type=int,  default=640,
                   help='Inference image size (pixels, square)')
    p.add_argument('--batch',   type=int,  default=32,
                   help='Batch size for evaluation')
    p.add_argument('--device',  type=str,  default='',
                   help='GPU (e.g. "0" or "0,1") or "cpu"')
    p.add_argument('--conf',    type=float, default=0.001,
                   help='Confidence threshold for NMS during val')
    p.add_argument('--iou',     type=float, default=0.7,
                   help='IoU threshold for mAP calculation')
    p.add_argument('--split',   type=str,  default='val',
                   choices=['train', 'val', 'test'],
                   help='Which dataset split to run evaluation on')
    p.add_argument('--half',    action='store_true',
                   help='FP16 inference (GPU only)')
    p.add_argument('--plots',   action='store_true',
                   help='Save PR curve, confusion-matrix, F1-curve, etc.')
    p.add_argument('--save_dir', type=str, default='runs/eval',
                   help='Root directory where results will be stored')
    return p.parse_args()


def main(opt: argparse.Namespace) -> None:
    """Run Ultralytics validation and report metrics."""
    # 1. Load the model
    model = YOLO(opt.weights)

    # 2. Run validation
    metrics = model.val(
        data=opt.data,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        conf=opt.conf,
        iou=opt.iou,
        split=opt.split,
        save_json=True,    # write COCO-format json (good for external tooling)
        save_conf=True,    # save per-box confidences in txts
        half=opt.half,
        project=opt.save_dir,
        name=Path(opt.weights).stem,
        plots=opt.plots
    )

    # 3. Pretty-print the resulting scalar metrics
    print(f"\n‣ Results for {opt.weights}:")
    # Ultralytics ≥ 8.3 returns a DetMetrics object
    if hasattr(metrics, "results_dict"):          # new versions
        mdict = metrics.results_dict
    elif isinstance(metrics, dict):               # legacy versions
        mdict = metrics
    else:
        raise TypeError("Unexpected metrics return type")

    for k, v in mdict.items():
        if isinstance(v, (float, int)):
            print(f"  {k:<20}: {v:.4f}")
        else:                                      # arrays, dicts, etc.
            print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    main(parse_args())
