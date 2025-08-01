from sahi.utils.coco import Coco, export_coco_as_yolo

# init Coco object
train_coco = Coco.from_coco_dict_or_path("data/merged_coco_sliced_train/data/labels.json", image_dir="data/merged_coco_sliced_train/data/data")
val_coco = Coco.from_coco_dict_or_path("data/merged_coco_sliced_val/data/labels.json", image_dir="data/merged_coco_sliced_val/data/data")

# export converted YoloV5 formatted dataset into given output_dir with given train/val split
data_yml_path = export_coco_as_yolo(
  output_dir="data/merged_sliced",
  train_coco=train_coco,
  val_coco=val_coco,
  disable_symlink=True
)
