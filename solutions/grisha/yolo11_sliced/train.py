from ultralytics import YOLO
from clearml import Task

# ── 1. ClearML tracking ────────────────────────────────────────────
Task.init(
    project_name="archipelago-2025-cv-hack",
    task_name="YOLOv8n-P2_UAV_people"
)

# ── 2. Paths ───────────────────────────────────────────────────────
DATA_YAML = "data/merged/uav_people_sliced.yaml"   # after SAHI tiling
MODEL_CFG = "models/yolov8n-p2.yaml"               # stride-4 head
PRETRAINED = "yolov8n.pt"                          # ImageNet-&-COCO weights

# ── 3. Training arguments ─────────────────────────────────────────
train_args = dict(
    data=DATA_YAML,
    imgsz=1536,
    epochs=120,
    batch=8,
    lr0=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    erasing=0.0,
    close_mosaic=10,
    rect=True,
    cache=True,
    hsv_h=0.015,
    hsv_s=0.70,
    hsv_v=0.40,
    translate=0.05,
    scale=0.60,
    fliplr=0.50,
    amp=True,
    device=0,
    project="solutions/grisha/yolo11",
    name="uav_people_p2_1536",
)

# Log all hyper-params to ClearML
Task.current_task().connect(train_args)

# ── 4. Train ───────────────────────────────────────────────────────
model = YOLO(PRETRAINED, cfg=MODEL_CFG)   # loads backbone weights + P2 head
model.train(**train_args)
