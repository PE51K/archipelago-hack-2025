# UAV Object Detection Solution (YOLOv8n)

## Solution Description

This solution implements YOLOv8n model for real-time object detection in UAV imagery. The model was trained for only 10 epochs.

## Technical Details

- **Base model**: YOLOv8n (Ultralytics)
- **Training**:
  - Epochs: 10
  - Optimizer: SGD
  - Image size: 640x640
  - Batch size: 16
  - Augmentations: Default YOLOv8
- **Docker image**: `your_dockerhub/yolo_solution:latest`


```
yolo_solution/
├── solution.py          # Основной скрипт для инференса
├── metadata.json       # Метаданные модели
├── weights/            
│   └── best.pt         # Веса обученной модели
└── train.ipynb         # Jupyter-ноутбук с процессом обучения
```

## Submission Results

| Name | Solution | Base Model | Features | Score | Comments |
|------|----------|------------|----------|-------|----------|
| YourName | [yolo_solution](solutions/your_name/yolo_solution) | YOLOv8n | 10-epoch training | 0.2144 | Fast inference, minimal training |
