import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# Paths
dataset_dir = Path("data/merged_sliced")
images_list_file = dataset_dir / "selected_train_images.txt"
train_dir = dataset_dir / "train"
sampled_train_dir = dataset_dir / "sampled_train"
original_yaml = dataset_dir / "data.yml"
sampled_yaml = dataset_dir / "sampled_data.yaml"

# Create output folder
sampled_train_dir.mkdir(exist_ok=True)

# Copy selected images + labels
with open(images_list_file, "r") as f:
    selected_images = [line.strip() for line in f if line.strip()]

for img_name in tqdm(selected_images, desc="Copying images"):
    img_path = train_dir / img_name
    label_path = img_path.with_suffix(".txt")

    if not img_path.exists():
        print(f"WARNING: Image not found: {img_path}")
        continue

    # Copy image
    shutil.copy(img_path, sampled_train_dir / img_path.name)

    # Copy label if exists
    if label_path.exists():
        shutil.copy(label_path, sampled_train_dir / label_path.name)

print(f"✅ Copied {len(selected_images)} images to {sampled_train_dir}")

# Create new YAML config
with open(original_yaml, "r") as f:
    data_cfg = yaml.safe_load(f)

data_cfg["train"] = str(sampled_train_dir.name)  # relative path inside dataset_dir

with open(sampled_yaml, "w") as f:
    yaml.dump(data_cfg, f)

print(f"✅ Created YAML: {sampled_yaml}")
