#!/usr/bin/env python3
"""
split_dataset.py

Move a random subset of YOLO-style dataset files from a root directory
(with `train` and `val` subfolders) into another directory, preserving the
same split structure. The operation can later be reversed using a manifest
file generated during the move.

Usage examples
--------------
# Move 20% of items from data/merged_sliced to data/heldout_20
python split_dataset.py move \
    --source data/merged_sliced \
    --dest   data/heldout_20 \
    --percent 20 \
    --seed 42

# Restore the files back to data/merged_sliced
python split_dataset.py restore \
    --source data/merged_sliced \
    --dest   data/heldout_20
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

DEFAULT_SUBDIRS = ("train", "val")
MANIFEST_NAME = "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move or restore a random dataset slice (YOLO format)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub‑parser: move
    move_p = subparsers.add_parser("move", help="Move a random subset of files.")
    move_p.add_argument("--source", type=Path, required=True, help="Path to original dataset root.")
    move_p.add_argument("--dest", type=Path, required=True, help="Destination where subset will be moved.")
    move_p.add_argument("--percent", type=float, default=20.0, help="Percent of items to move (0–100).")
    move_p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    # Sub‑parser: restore
    restore_p = subparsers.add_parser("restore", help="Restore files back using manifest.")
    restore_p.add_argument("--source", type=Path, required=True, help="Original dataset root (where to restore).")
    restore_p.add_argument("--dest", type=Path, required=True, help="Folder containing moved subset & manifest.")

    return parser.parse_args()


def valid_stems(directory: Path) -> List[str]:
    """Return list of sample stems (filename without extension) that have both an image and a .txt label."""
    stems: Dict[str, bool] = {}
    for img_path in directory.glob("*.[jp][pn]*g"):
        # Matches .jpg, .jpeg, .png
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            stems[img_path.stem] = True
    return list(stems)


def move_pair(stem: str, src_dir: Path, dst_dir: Path) -> None:
    """Move <stem>.jpg|png and <stem>.txt from src_dir to dst_dir if they exist."""
    for ext in (".jpg", ".jpeg", ".png", ".txt"):
        src_file = src_dir / f"{stem}{ext}"
        if src_file.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_file), str(dst_dir / src_file.name))


def move_subset(source: Path, dest: Path, percent: float, seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)

    moved: Dict[str, List[str]] = {}

    for sub in DEFAULT_SUBDIRS:
        src_sub = source / sub
        dst_sub = dest / sub

        stems = valid_stems(src_sub)
        if not stems:
            print(f"[WARN] No stems found in {src_sub}, skipping…")
            continue

        n_move = max(1, int(len(stems) * percent / 100))
        chosen = random.sample(stems, n_move)
        moved[sub] = chosen

        for stem in chosen:
            move_pair(stem, src_sub, dst_sub)
        print(f"Moved {len(chosen)} items from {src_sub} → {dst_sub}")

    # Write manifest for later restoration
    manifest_path = dest / MANIFEST_NAME
    manifest_path.write_text(json.dumps({"percent": percent, "moved": moved}, indent=2))
    print(f"Manifest saved to {manifest_path.relative_to(Path.cwd())}")


def restore_subset(source: Path, dest: Path) -> None:
    manifest_path = dest / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Cannot restore without it.")

    manifest = json.loads(manifest_path.read_text())
    moved: Dict[str, List[str]] = manifest.get("moved", {})

    for sub, stems in moved.items():
        src_sub = source / sub
        dst_sub = dest / sub
        for stem in stems:
            move_pair(stem, dst_sub, src_sub)
        print(f"Restored {len(stems)} items from {dst_sub} → {src_sub}")

    print("Restoration complete. You may safely delete the destination folder afterward if empty.")


def main():
    args = parse_args()
    if args.command == "move":
        move_subset(args.source, args.dest, args.percent, args.seed)
    elif args.command == "restore":
        restore_subset(args.source, args.dest)


if __name__ == "__main__":
    main()
