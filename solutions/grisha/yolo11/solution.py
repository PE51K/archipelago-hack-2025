# solution.py  ────────────────────────────────────────────────
import numpy as np
from typing import List, Union
from pathlib import Path
from ultralytics import YOLO

# checkpoint
WEIGHT = Path("resources/weights/yolo11n/weights/best.pt")

# inference knobs
CONF   = 0.25          # adjust if desired
DEVICE = 0             # "cpu" or CUDA index

# load once
model = YOLO(str(WEIGHT)).fuse()
model.conf = CONF

def _run(img: np.ndarray) -> List[dict]:
    h, w = img.shape[:2]
    preds = model.predict(source=img, device=DEVICE, verbose=False)

    out: List[dict] = []
    for res in preds:
        for box in res.boxes:
            xcn, ycn, wn, hn = box.xywhn[0].tolist()   # normalized
            out.append(
                dict(
                    xc=float(xcn),
                    yc=float(ycn),
                    w=float(wn),
                    h=float(hn),
                    label=0,                            # single-class
                    score=float(box.conf[0])
                )
            )
    return out

def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    if isinstance(images, np.ndarray):
        images = [images]
    return [_run(im) for im in images]
