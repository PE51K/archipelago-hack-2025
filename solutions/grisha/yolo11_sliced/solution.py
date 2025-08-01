"""
High-res UAV people detector – SAHI tiled inference version
"""

from typing import List, Union
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Model --------------------------------------------------------------------
_WEIGHTS = "best.pt"                    # path to your trained weights
detection_model = AutoDetectionModel.from_pretrained(      # SAHI 0.23+ API
    model_type="ultralytics",
    model_path=_WEIGHTS,
    confidence_threshold=0.25,
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
