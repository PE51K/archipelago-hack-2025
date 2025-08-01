import numpy as np
from typing import List, Union
from pathlib import Path
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion

# Constants for image slicing and prediction thresholds
IMGZ, OVERLAP = 1024, 0.20  # Slice size and overlap ratio
CONF, WBF_IOU = 0.12, 0.55  # Confidence threshold and WBF IoU threshold

# Load YOLO model weights
model = YOLO("best.pt")


def _run(im: np.ndarray) -> List[dict]:
    """
    Run object detection on a single image using sliced prediction and weighted boxes fusion.

    Args:
        im (np.ndarray): Input image as a NumPy array.

    Returns:
        List[dict]: List of detected objects, each as a dictionary with normalized coordinates,
                    width, height, label, and score.
    """
    h, w, _ = im.shape

    # Perform sliced prediction using SAHI
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
    # Collect bounding boxes and scores from predictions
    for o in r.object_prediction_list:
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        bxs.append([x1/w, y1/h, x2/w, y2/h])  # Normalize coordinates
        scs.append(o.score.value)
    if not bxs:
        return []

    # Fuse overlapping boxes using Weighted Boxes Fusion
    bxs, scs, _ = weighted_boxes_fusion(
        [bxs], 
        [scs], 
        [[0]],
        iou_thr=WBF_IOU, 
        skip_box_thr=1e-3
    )

    # Format results as list of dictionaries
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
    """
    Run object detection on a batch of images.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): List of images or a single image as NumPy arrays.

    Returns:
        List[List[dict]]: List of detection results for each image.
    """
    if isinstance(images, np.ndarray):
        images = [images]
    # Run detection for each image
    return [_run(im) for im in images]
