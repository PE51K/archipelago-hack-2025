# solution.py ------------------------------------------
import numpy as np
from typing import List, Union
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
import pathlib

WEIGHTS = pathlib.Path("artifacts/yolo11m/best.pt")   # <— same path
IMGZ    = 1024
CONF    = 0.12
DEVICE  = 0

model = YOLO(str(WEIGHTS)).fuse()

def _run(img: np.ndarray):
    h, w, _ = img.shape
    res = get_sliced_prediction(
        img, model,
        slice_height=IMGZ, slice_width=IMGZ,
        overlap_height_ratio=0.20, overlap_width_ratio=0.20,
        conf=CONF, device=str(DEVICE), verbose=False
    )
    preds = []
    for op in res.object_prediction_list:
        x1, y1, x2, y2 = op.bbox.to_xyxy()
        preds.append(dict(
            xc=float((x1+x2)/2/w),
            yc=float((y1+y2)/2/h),
            w=float((x2-x1)/w),
            h=float((y2-y1)/h),
            label=0,
            score=float(op.score.value)
        ))
    return preds

def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    if isinstance(images, np.ndarray):
        images = [images]
    return [_run(im) for im in images]
