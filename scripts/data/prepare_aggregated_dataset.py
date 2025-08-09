import random
import shutil
from pathlib import Path
from tqdm import tqdm
import os
import yaml

def prepare_aggregated_dataset(source_paths, dest_path, neg_pos_ratio):

    """
    Aggregates, and balances a dataset from multiple sources for training YOLO models.

    This function solves three key problems when working with large, partitioned datasets:
    1.  **Aggregation:** Combines files from multiple input directories (`source_paths`) into a
        single, unified structure.
    2.  **Training Set Balancing:** Creates a training dataset with a controlled ratio of "negative" (no objects)
        to "positive" (with objects) examples to combat false positives and accelerate training.
    3.  **Preserving Real Validation Distribution:** The validation set remains untouched to ensure a fair
        and realistic evaluation of the model's performance.

    The process is executed in two passes for maximum reliability:
    - **Pass 1 (Aggregation):** Scans all specified `source_paths`, analyzes label files,
      and compiles complete lists of "positive" and "negative" images for the `train` and `val` splits.
    - **Pass 2 (Sampling and Linking):** Based on the aggregated lists, it creates the final dataset
      structure in `dest_path` using symbolic links (symlinks) to save disk space.

    Args:
        source_paths (list[Path]):
            A list of paths (`pathlib.Path` objects) to the root directories of the input datasets.
            Each dataset is expected to contain `train` and/or `val` subdirectories, which in turn
            contain a mix of image (.jpg, .jpeg, .png) and label (.txt) files.

        dest_path (Path):
            The path to the target directory where the final, ready-to-train dataset structure
            will be created. If the directory exists, it will be completely removed and recreated.

        neg_pos_ratio (float):
            The desired ratio of "negative" (background) to "positive" (with objects) examples
            in the **training** set.
            - `1.0` means a 1:1 ratio (one negative for each positive).
            - `0.25` means a 1:4 ratio (one negative for every four positives).
            - `0.0` will completely exclude background images from training.

    Returns:
        Path:
            The absolute path to the generated `dataset.yaml` file, which can be passed
            directly to the `train()` method of a YOLO model.

    Raises:
        ValueError: If `random.sample` cannot select the requested number of negative
                    examples (though the code has a safeguard against this).
    """
  
    print(f"\nINFO: Setting up final, BALANCED dataset directory at {dest_path}")
    if dest_path.exists(): shutil.rmtree(dest_path)
    images_train_dir, labels_train_dir = dest_path / "images/train", dest_path / "labels/train"
    images_val_dir, labels_val_dir = dest_path / "images/val", dest_path / "labels/val"
    for d in [images_train_dir, labels_train_dir, images_val_dir, labels_val_dir]:
        d.mkdir(parents=True)
        
    VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    print("--- PASS 1: Aggregating all data files ---")
    all_train_pos, all_train_neg = [], []
    all_val_pos, all_val_neg = [], []

    for source_path in source_paths:
        print(f"--> Analyzing {source_path.name}...")
        for split in ["train", "val"]:
            source_split_dir = source_path / split
            if not source_split_dir.exists(): continue
            
            image_files = [f for f in source_split_dir.glob("*") if f.suffix.lower() in VALID_IMAGE_EXTENSIONS]
            for img_file in tqdm(image_files, desc=f"  - Reading {split} manifest"):
                label_file = img_file.with_suffix('.txt')
                
                is_positive = is_truly_positive(label_file)
                if split == 'train':
                    (all_train_pos if is_positive else all_train_neg).append(img_file)
                else:
                    (all_val_pos if is_positive else all_val_neg).append(img_file)

    print("\n--- Aggregation Complete ---")
    print(f"Total Train Positives: {len(all_train_pos)}")
    print(f"Total Train Negatives: {len(all_train_neg)}")

    print("\n--- PASS 2: Sampling and Linking ---")
    num_neg_to_add = int(len(all_train_pos) * neg_pos_ratio)
    num_neg_to_add = min(num_neg_to_add, len(all_train_neg))
    print(f"Linking all {len(all_train_pos)} positives and {num_neg_to_add} sampled negatives for TRAIN...")
    final_train_paths = all_train_pos + random.sample(all_train_neg, num_neg_to_add)
    for img_path in tqdm(final_train_paths, desc="  - Linking train set"):
        lbl_path = img_path.with_suffix('.txt')
        os.symlink(img_path, images_train_dir / img_path.name)
        if lbl_path.exists(): os.symlink(lbl_path, labels_train_dir / lbl_path.name)
    final_val_paths = all_val_pos + all_val_neg
    print(f"Linking all {len(final_val_paths)} images for VAL...")
    for img_path in tqdm(final_val_paths, desc="  - Linking val set"):
        lbl_path = img_path.with_suffix('.txt')
        os.symlink(img_path, images_val_dir / img_path.name)
        if lbl_path.exists(): os.symlink(lbl_path, labels_val_dir / lbl_path.name)
        
    yaml_path = dest_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({'path': str(dest_path.resolve()), 'train': 'images/train', 'val': 'images/val', 'nc': 1, 'names': ['person']}, f)
    return yaml_path
