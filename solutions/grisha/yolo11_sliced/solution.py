import numpy as np
from typing import List, Union
from pathlib import Path
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion


IMGZ, OVERLAP = 1024, 0.20
CONF, WBF_IOU = 0.12, 0.55


model = YOLO("best.pt")


def _run(im: np.ndarray) -> List[dict]:
    h, w, _ = im.shape

    r = get_sliced_prediction(
        im, 
        model,
        slice_height=IMGZ, 
        slice_width=IMGZ,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        conf=CONF, 
        device="0", 
        verbose=False
    )

    bxs, scs = [], []
    for o in r.object_prediction_list:
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        bxs.append([x1/w, y1/h, x2/w, y2/h])
        scs.append(o.score.value)
    if not bxs:
        return []

    bxs, scs, _ = weighted_boxes_fusion(
        [bxs], 
        [scs], 
        [[0]],
        iou_thr=WBF_IOU, 
        skip_box_thr=1e-3
    )

    return [
        dict(
            xc=(x1+x2)/2, 
            yc=(y1+y2)/2,
            w=x2-x1, 
            h=y2-y1,
            label=0, 
            score=s
        ) 
        for (x1, y1, x2, y2), s 
        in zip(bxs, scs)
    ]


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    if isinstance(images, np.ndarray):
        images = [images]
    return [_run(im) for im in images]
