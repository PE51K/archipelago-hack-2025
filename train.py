# train.py ---------------------------------------------
from ultralytics import YOLO
from clearml import Task
import shutil, pathlib

DATA_YAML = "resources/data/merged/uav_people.yaml"
MODEL     = "yolo11m.pt"
IMG_SIZE  = 1024
EPOCHS    = 150
OUT_DIR   = pathlib.Path("weights/yolo11m")   # <— central weights folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

task = Task.init(project_name="archipelago-2025-cv-hack",
                 task_name   ="yolo11m_uav_people")

train_args = dict(
    data           = DATA_YAML,
    imgsz          = IMG_SIZE,
    epochs         = EPOCHS,
    batch          = -1,
    lr0            = 0.01,
    warmup_epochs  = 3,
    mosaic         = 1.0,
    copy_paste     = True,
    cutmix         = 0.2,
    close_mosaic   = 10,
    device         = 0,
    amp            = True,
    project        = "runs_yolo11",            # keeps Ultralytics logs tidy
    name           = "uav_people_1024"
)
task.connect(train_args)                       # ✅  ClearML logging

model = YOLO(MODEL)
model.train(**train_args)

# ── copy best checkpoint to artefacts/ for Docker use ─────────
best_pt = pathlib.Path(train_args["project"]) / \
          train_args["name"] / "weights" / "best.pt"
shutil.copy2(best_pt, OUT_DIR / "best.pt")     # explicit, predictable path
print(f"✅  Best weights saved ➜ {OUT_DIR/'best.pt'}")
