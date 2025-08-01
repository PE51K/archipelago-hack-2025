from ultralytics import YOLO
from clearml import Task
from pathlib  import Path


# Training arguments
args = dict(
    data="data/merged/uav_people.yaml",
    imgsz=1024,
    epochs=150,
    batch=-1,
    lr0=0.01,
    warmup_epochs=3,
    mosaic=1.0,
    copy_paste=1.0,
    mixup=0.10,
    cutmix=0.20,
    erasing=0.10,
    close_mosaic=10,
    box=7,
    cls=5,
    patience=20,
    amp=True,
    device=0,
    project="solutions/grisha/yolo11_sliced",
    name="finetuned",
)


# ClearML logging
Task.init(project_name="archipelago-2025-cv-hack", task_name="YOLO-11n_uav_people")
Task.current_task().connect(args)


# Training
YOLO("solutions/grisha/yolo11_sliced/pretrained/weights/yolo11n.pt").train(**args)
