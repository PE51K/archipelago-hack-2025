from ultralytics import YOLO
from clearml import Task


# Training arguments
args = dict(
    # ────────── Training ──────────
    data="data/merged_sliced/data.yml",
    epochs=50,
    patience=10, # Early stopping on 10 epochs without improvement
    batch=0.95, # Use 95% of GPU mem
    imgsz=1536/2, # Was 1536, 768 for faster training
    device=0,
    workers=16,
    project="solutions/grisha/yolo11_sliced", # Train data will be saved in `project + name` folder
    name="finetuned",
    pretrained=True, # Use pretrained weights if available
    optimizer="AdamW",
    single_cls=True, # We are only detecting people
    rect=True, # Use rectangular training to speed up training
    cos_lr=True, # Use cosine learning rate scheduler
    close_mosaic=10, # Disable mosaic augmentation on last 10 epochs
    resume=False, # Start training from scratch
    amp=True, # Use automatic mixed precision for faster training
    fraction=1.0, # Use 100% of the dataset
    lr0=0.01,
    weight_decay=0.0005,
    warmup_epochs=3, # Warmup lr during first 3 epochs
    dropout=0.2, # Dropout rate for regularization

    # ────────── Augmentations ──────────
    hsv_h=0.015, # Hue augmentation percentage range
    hsv_s=0.70, # Saturation augmentation percentage range
    hsv_v=0.40, # Brightness augmentation percentage range
    degrees=0.0, # No rotation augmentation
    translate=0.05, # Translation augmentation percentage range (keeping it low to avoid losing small objects)
    scale=0.60, # Scale augmentation percentage range
    shear=0.0, # No shear augmentation
    perspective=0.0, # No perspective augmentation
    flipud=0.0, # No vertical flip augmentation
    fliplr=0.50, # Horizontal flip augmentation probability
    bgr=0.0, # No RGB->BGR channel swapping
    mosaic=0.8, # Mosaic augmentation probability (combines 4 images into one)
    mixup=0.0, # Mixup augmentation probability (combines 2 images into one)
    cutmix=0.0, # CutMix augmentation probability (cuts and pastes patches from one image to another)
)



# ClearML logging
Task.init(project_name="archipelago-2025-cv-hack", task_name="YOLO-11n_uav_people_slicing")
Task.current_task().connect(args)


# Training
YOLO("solutions/grisha/yolo11_sliced/model_configs/yolo11n-p2.yaml").train(**args)
