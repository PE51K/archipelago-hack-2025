#!/usr/bin/env python3
"""
Grid-search optimal confidence & IoU thresholds on a YOLOv8 model
=================================================================

Example
-------
python optimize.py \
    --weights runs/train/8_512_custom_loss_sliced_dataset_slice_size_1536_focal_loss_resume_1/weights/best.pt \
    --data data/merged_sliced/data.yml \
    --imgsz 512 --batch 64 --device 0 \
    --conf_range 0.05 0.95 0.05 \
    --iou_range 0.40 0.95 0.05 \
    --metric f1 --save_csv
"""
from pathlib import Path
import argparse
import csv
from tqdm import tqdm
from ultralytics import YOLO

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
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
# --------------------------------------------------------------------------- #
#                        command-line arguments                               #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    g = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g.add_argument("--weights",  type=str, required=True,  help="Trained *.pt checkpoint")
    g.add_argument("--data",     type=str, required=True,  help="dataset YAML")
    g.add_argument("--imgsz",    type=int, default=640,    help="inference image size (square)")
    g.add_argument("--batch",    type=int, default=32,     help="batch size for val()")
    g.add_argument("--device",   type=str, default="",     help='"0", "0,1", or "cpu"')
    g.add_argument("--split",    type=str, default="val",  choices=["train", "val", "test"])
    g.add_argument("--conf_range", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                   default=(0.05, 0.95, 0.05),
                   help="grid of confidence thresholds")
    g.add_argument("--metric",   type=str, default="f1",
                   choices=["f1", "map50", "map5095", "precision", "recall", "fitness"],
                   help="metric to optimise")
    g.add_argument("--half",     action="store_true", help="FP16 inference")
    g.add_argument("--save_csv", action="store_true", help="write full grid to CSV")
    g.add_argument("--out_dir",  type=str, default="runs/threshold_search",
                   help="directory for CSV / artifacts")
    return g.parse_args()


# --------------------------------------------------------------------------- #
#                 mapping Ultralytics result keys → friendly names           #
# --------------------------------------------------------------------------- #
def get_metric(res, metric_key: str) -> float:
    """
    Return the metric requested from the Ultralytics validation result.

    Ultralytics ≥ 8.3 → DetMetrics object
    Older releases        → plain dict
    """
    # unpack to a simple dict first
    mdict = res.results_dict if hasattr(res, "results_dict") else res

    match metric_key.lower():
        case "precision" | "p":
            return mdict["metrics/p"]
        case "recall" | "r":
            return mdict["metrics/r"]
        case "map50":
            return mdict["metrics/mAP_0.5"]
        case "map5095" | "map":
            return mdict["metrics/mAP_0.5:0.95"]
        case "f1":
            p, r = mdict["metrics/p"], mdict["metrics/r"]
            return 2 * p * r / (p + r + 1e-16)
        case "fitness":
            # attribute exists only on DetMetrics
            print(res.fitness if hasattr(res, "fitness") else mdict["fitness"])
            return res.fitness if hasattr(res, "fitness") else mdict["fitness"]
        case _:
            raise ValueError(f"Unknown metric '{metric_key}'")


# --------------------------------------------------------------------------- #
#                           main search routine                               #
# --------------------------------------------------------------------------- #
def main(opt: argparse.Namespace) -> None:
    conf_start, conf_stop, conf_step = opt.conf_range

    conf_values = [round(c, 3) for c in frange(conf_start, conf_stop, conf_step)]

    model = YOLO(opt.weights)

    best_score, best_conf = -1.0, None
    grid_results = []

    print(f"Searching {len(conf_values)} combinations…")

    for conf in tqdm(conf_values, desc="confidence"):
        res = model.val(
            data=opt.data,
            imgsz=opt.imgsz,
            batch=opt.batch,
            device=opt.device,
            split=opt.split,
            conf=conf,
            half=opt.half,
            save_json=False,
            save_conf=False,
            verbose=False
        )
        score = get_metric(res, opt.metric)
        grid_results.append((conf, score))

        if score > best_score:
            best_score, best_conf = score, conf

    # --------------------------------------------------------------------- #
    #                summary printout + optional CSV write                  #
    # --------------------------------------------------------------------- #
    print("\n╭─ Best thresholds ───────────────────────────────")
    print(f"│ metric : {opt.metric}")
    print(f"│ conf   : {best_conf}")
    print(f"│ score  : {best_score:.5f}")
    print("╰─────────────────────────────────────────────────")

    if opt.save_csv:
        out_dir = Path(opt.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{Path(opt.weights).stem}_grid_{opt.metric}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conf", opt.metric])
            writer.writerows(grid_results)
        print(f"Full grid saved to {csv_path.resolve()}")


def frange(start: float, stop: float, step: float):
    """Floating-point range generator (inclusive of stop)."""
    while start <= stop + 1e-9:
        yield start
        start += step


if __name__ == "__main__":
    main(parse_args())
