import time, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import optuna
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion
import examples.metric as metric


VAL_IMGDIR = Path("data/val/images")
VAL_GTCSV  = Path("solutions/grisha/yolo11_sliced/val_gt.csv")
WEIGHT     = Path("solutions/grisha/yolo11_sliced/finetuned/weights/best.pt")


IMGZ        = 1024
DEVICE      = "0"
SKIP_BOX_T  = 1e-3
BETA        = 1.0


def sliced_wbf_predict(im, model, conf, overlap, wbf_iou):
    h, w, _ = im.shape
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
    for o in r.object_prediction_list:
        x1, y1, x2, y2 = o.bbox.to_xyxy()
        bxs.append([x1 / w, y1 / h, x2 / w, y2 / h])
        scs.append(o.score.value)

    if not bxs:
        return []
    
    bxs, scs, _ = weighted_boxes_fusion(
        [bxs], [scs], [[0]],
        iou_thr=wbf_iou, skip_box_thr=SKIP_BOX_T
    )

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
    rows = []
    for p in VAL_IMGDIR.glob("*.*"):
        img_id = p.stem
        im     = cv2.cvtColor(cv2.imread(str(p), -1), cv2.COLOR_BGR2RGB)
        start  = time.time()
        preds  = sliced_wbf_predict(im, model, conf, overlap, wbf_iou)
        dt     = round(time.time() - start, 5)

        if preds:
            for d in preds:
                d |= dict(
                    image_id=img_id, 
                    time_spent=dt,
                    w_img=im.shape[1], 
                    h_img=im.shape[0]
                )
                rows.append(d)

        else:
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
    pred_bytes = metric.df_to_bytes(df_pred)
    gt_bytes   = metric.open_df_as_bytes(str(VAL_GTCSV))
    score, *_  = metric.evaluate(pred_bytes, gt_bytes, beta=BETA, parallelize=False)
    return score


def objective(trial):
    conf     = trial.suggest_float("conf",     0.05, 0.40)
    overlap  = trial.suggest_float("overlap",  0.10, 0.35)
    wbf_iou  = trial.suggest_float("wbf_iou",  0.45, 0.70)
    df_pred  = run_validation(model, conf, overlap, wbf_iou)
    return fbeta(df_pred)


if __name__ == "__main__":
    model = YOLO(WEIGHT)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    best = study.best_params
    print("\nBest hyper-params:", best, "Fβ =", study.best_value)
    Path("best_infer.json").write_text(json.dumps(best, indent=2))
