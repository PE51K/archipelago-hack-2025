"""
High-res UAV people detector – SAHI tiled inference version
"""

from typing import List, Union
import numpy as np

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.loss import v8DetectionLoss, FocalLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
import torch.nn.functional as F
import torch

# Supress invalid predictor warnings
import logging
logging.getLogger('sahi.models.ultralytics').setLevel(logging.ERROR)


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
    
    
# ── Model --------------------------------------------------------------------
_WEIGHTS = "best.pt"                    # path to your trained weights
detection_model = UltralyticsDetectionModel(      # SAHI 0.23+ API
    model_path=_WEIGHTS,
    confidence_threshold=0.25,
    image_size=512,
    device="cuda:0",                    # set "cpu" if no GPU
)

# ── Tiling parameters (keep them in sync with training) ----------------------
_SLICE = 1536
_OVERLAP = 0.20


def _bbox_to_norm_xywh(x1, y1, x2, y2, img_w: int, img_h: int):
    """Convert absolute coords to YOLO-style normalised xc-yc-w-h."""
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def _infer_single(image: np.ndarray) -> List[dict]:
    """Run SAHI sliced inference on one image."""
    h, w = image.shape[:2]

    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=_SLICE,
        slice_width=_SLICE,
        overlap_height_ratio=_OVERLAP,
        overlap_width_ratio=_OVERLAP,
        verbose=0,
    )

    detections = []
    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_voc_bbox()      # safest accessor
        xc, yc, w_n, h_n = _bbox_to_norm_xywh(x1, y1, x2, y2, w, h)

        detections.append(
            {
                "xc": xc,
                "yc": yc,
                "w": w_n,
                "h": h_n,
                "label": int(obj.category.id) if obj.category else 0,
                "score": obj.score.value,
            }
        )
    return detections


def predict(images: Union[np.ndarray, List[np.ndarray]]) -> List[List[dict]]:
    """
    Args
    ----
    images : a single H×W×C image or a list thereof.

    Returns
    -------
    List[List[dict]] — detection list per image.
    """
    if isinstance(images, np.ndarray):
        images = [images]
    return [_infer_single(img) for img in images]


# ── Quick sanity check -------------------------------------------------------
if __name__ == "__main__":
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)  # black test image
    out = predict(dummy)
    print("Dummy prediction:", out)
    