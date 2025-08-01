import time
import json
"""
This script performs hyperparameter tuning for sliced inference using a YOLO model and weighted boxes fusion (WBF) on a validation dataset.
It uses Optuna for optimization and evaluates predictions using a custom metric.

Modules:
    - time, json: For timing and saving results.
    - pathlib.Path: For file path handling.
    - cv2: For image reading and processing.
    - numpy, pandas: For data manipulation.
    - optuna: For hyperparameter optimization.
    - ultralytics.YOLO: For loading YOLO models.
    - sahi.predict.get_sliced_prediction: For sliced inference.
    - ensemble_boxes.weighted_boxes_fusion: For WBF post-processing.
    - examples.metric: For custom metric evaluation.

Constants:
    VAL_IMGDIR: Path to validation images.
    VAL_GTCSV: Path to ground truth CSV.
    WEIGHT: Path to YOLO model weights.
    IMGZ: Image slice size.
    DEVICE: Device for inference.
    SKIP_BOX_T: WBF skip box threshold.
    BETA: Beta value for F-beta metric.

Functions:
    sliced_wbf_predict(im, model, conf, overlap, wbf_iou):
        Runs sliced prediction on an image, applies WBF, and returns normalized bounding boxes.
    run_validation(model, conf, overlap, wbf_iou):
        Runs inference on all validation images, collects predictions, and returns a DataFrame.
    fbeta(df_pred):
        Computes the F-beta score for predictions using the custom metric.
    objective(trial):
        Optuna objective function for hyperparameter search.

Main:
    Loads YOLO model, runs Optuna optimization, prints and saves best hyperparameters.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import optuna
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion
import examples.metric as metric

# Paths and constants for validation and model
VAL_IMGDIR = Path("data/val/images")
VAL_GTCSV  = Path("solutions/grisha/yolo11_sliced/val_gt.csv")
WEIGHT     = Path("solutions/grisha/yolo11_sliced/finetuned/weights/best.pt")

IMGZ        = 1024           # Slice size for inference
DEVICE      = "0"            # CUDA device
SKIP_BOX_T  = 1e-3           # WBF skip box threshold
BETA        = 1.0            # F-beta metric beta value

def sliced_wbf_predict(im, model, conf, overlap, wbf_iou):
    """
    Runs sliced prediction on an image, applies Weighted Boxes Fusion (WBF),
    and returns normalized bounding boxes.

    Args:
        im (np.ndarray): Input image.
        model: YOLO model instance.
        conf (float): Confidence threshold.
        overlap (float): Overlap ratio for slicing.
        wbf_iou (float): IOU threshold for WBF.

    Returns:
        list[dict]: List of predicted bounding boxes with normalized coordinates.
    """
    h, w, _ = im.shape
    # Run sliced prediction using SAHI
    r = get_sliced_prediction(
        im, 
        model,
        slice_height=IMGZ, 
        slice_width=IMGZ,
        overlap_height_ratio=overlap, 
        overlap_width_ratio=overlap,
        conf=conf, 
        device=DEVICE, 
        verbose=False
    )

    bxs, scs = [], []
    # Collect bounding boxes and scores
    for o in r.object_prediction_list:
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        bxs.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scs.append(o.score.value)

    if not bxs:
        return []
    
    # Apply Weighted Boxes Fusion
    bxs, scs, _ = weighted_boxes_fusion(
        [bxs], [scs], [[0]],
        iou_thr=wbf_iou, skip_box_thr=SKIP_BOX_T
    )

    # Format predictions as dicts
    return [
        dict(
            xc=(x1 + x2) / 2,
            yc=(y1 + y2) / 2,
            w=x2 - x1,
            h=y2 - y1,
            label=0,
            score=s
        ) 
        for (x1, y1, x2, y2), s 
        in zip(bxs, scs)
    ]

def run_validation(model, conf, overlap, wbf_iou):
    """
    Runs inference on all validation images, collects predictions,
    and returns a DataFrame.

    Args:
        model: YOLO model instance.
        conf (float): Confidence threshold.
        overlap (float): Overlap ratio for slicing.
        wbf_iou (float): IOU threshold for WBF.

    Returns:
        pd.DataFrame: DataFrame of predictions for all images.
    """
    rows = []
    for p in VAL_IMGDIR.glob("*.*"):
        img_id = p.stem
        # Read and convert image to RGB
        im     = cv2.cvtColor(cv2.imread(str(p), -1), cv2.COLOR_BGR2RGB)
        start  = time.time()
        preds  = sliced_wbf_predict(im, model, conf, overlap, wbf_iou)
        dt     = round(time.time() - start, 5)

        if preds:
            # Add predictions to rows
            for d in preds:
                d |= dict(
                    image_id=img_id, 
                    time_spent=dt,
                    w_img=im.shape[1], 
                    h_img=im.shape[0]
                )
                rows.append(d)
        else:
            # If no predictions, add empty row
            rows.append(
                dict(
                    image_id=img_id, 
                    xc=None, 
                    yc=None, 
                    w=None, 
                    h=None,
                    label=0, 
                    score=None, 
                    time_spent=dt,
                    w_img=im.shape[1], 
                    h_img=im.shape[0]
                )
            )
            
    # Return predictions as DataFrame
    return pd.DataFrame(
        rows, 
        columns=[
            "image_id",
            "label",
            "xc",
            "yc",
            "w",
            "h",
            "w_img",
            "h_img",
            "score",
            "time_spent"
        ]
    )

def fbeta(df_pred):
    """
    Computes the F-beta score for predictions using the custom metric.

    Args:
        df_pred (pd.DataFrame): DataFrame of predictions.

    Returns:
        float: F-beta score.
    """
    pred_bytes = metric.df_to_bytes(df_pred)
    gt_bytes   = metric.open_df_as_bytes(str(VAL_GTCSV))
    score, *_  = metric.evaluate(pred_bytes, gt_bytes, beta=BETA, parallelize=False)
    return score

def objective(trial):
    """
    Optuna objective function for hyperparameter search.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: F-beta score for current hyperparameters.
    """
    conf     = trial.suggest_float("conf",     0.05, 0.40)
    overlap  = trial.suggest_float("overlap",  0.10, 0.35)
    wbf_iou  = trial.suggest_float("wbf_iou",  0.45, 0.70)
    df_pred  = run_validation(model, conf, overlap, wbf_iou)
    return fbeta(df_pred)

if __name__ == "__main__":
    # Load YOLO model
    model = YOLO(WEIGHT)
    # Create Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    # Print and save best hyperparameters
    best = study.best_params
    print("\nBest hyper-params:", best, "Fβ =", study.best_value)
    Path("best_infer.json").write_text(json.dumps(best, indent=2))
