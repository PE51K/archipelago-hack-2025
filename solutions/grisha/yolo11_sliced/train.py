from ultralytics import YOLO
from clearml import Task
from pathlib  import Path


DATA_YAML = "resources/data/merged/uav_people.yaml"
BASE_W = Path("resources/weights/pretrained/yolo11n/yolo11n.pt")
OUT_DIR = Path("resources/weights/yolo11n")


Task.init(project_name="archipelago-2025-cv-hack", task_name="YOLO-11n_uav_people")

args = dict(
    data=DATA_YAML,
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
    box=7, # Wise-IoU weight
    cls=5, # Focal-CLS weight
    patience=20, # early-stop after 20 idle epochs
    amp=True,
    device=0,
    project=str(OUT_DIR.parent),
    name=OUT_DIR.name
)
Task.current_task().connect(args)

YOLO(str(BASE_W)).train(**args)
