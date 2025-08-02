from ultralytics import YOLO
from clearml import Task


# Training arguments
args = dict(
    # ────────── Training ──────────
    data="data/merged_sliced/data.yml",
    imgsz=1536,
    epochs=50,
    batch=-1,
    lr0=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    amp=True,
    rect=True,
    device=0,
    project="solutions/grisha/yolo11_sliced",
    name="finetuned",

    # ────────── Data loading ──────────
    workers=16,

    # ────────── Augmentations ──────────
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    erasing=0.0,
    hsv_h=0.015,
    hsv_s=0.70,
    hsv_v=0.40,
    translate=0.05,
    scale=0.60,
    fliplr=0.50,

    # ────────── Validation / logging ──────────
    patience=15,
)



# ClearML logging
Task.init(project_name="archipelago-2025-cv-hack", task_name="YOLO-11n_uav_people_slicing")
Task.current_task().connect(args)


# Training
YOLO("solutions/grisha/yolo11_sliced/model_configs/yolo11n-p2.yaml").train(**args)
