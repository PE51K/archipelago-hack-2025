# Slicing

This document describes the process of creating a sliced YOLO UAV detection dataset from an existing YOLO UAV detection dataset. Slicing is useful for breaking large images into smaller patches, which can improve training efficiency and detection performance for UAV datasets.

## How to create sliced yolo uav detection dataset from default yolo uav detection dataset

1. Go to folder with uav detection dataset (`data/merged` in our case) and rename .yaml file to `dataset.yaml`.

2. In val path set path to folder that yo want to convert (dont ask why) (it means, that from now on we would convert val and train separetly). You can start from val folder.

3. Convert val and train to coco format:
```shell
fiftyone convert \                                           
    --input-dir data/merged \
    --input-type fiftyone.types.YOLOv5Dataset \
    --output-dir data/merged_coco_{val/train} \
    --output-type fiftyone.types.COCODetectionDataset;
```

4. Slice val and train:
```shell
sahi coco slice \
  --image_dir data/merged_coco_{val/train}/data \
  --dataset_json_path data/merged_coco_{val/train}/labels.json \
  --output_dir data/merged_coco_sliced_{val/train}/data \
  --slice_size 1536 \
  --overlap_ratio 0.2 
```

5. Merge sliced coco val and train into one yolo dataset using `scripts/data/slicing/coco_to_yolo.py`.

## References

- https://github.com/obss/sahi/discussions/755
- https://github.com/obss/sahi/blob/main/sahi/scripts/slice_coco.py
