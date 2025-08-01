#!/usr/bin/env python
"""
Merge the three Human-Rescue datasets into Ultralytics/YOLO format.

Example
-------
python prepare_dataset.py \
    --raw-root  /data/raw \
    --out-root  /data/merged \
    --include-private
"""
from __future__ import annotations
import argparse, shutil, os
from pathlib import Path
from typing import Iterable, Tuple
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def gather_pairs(img_root: Path, lbl_root: Path | None = None) -> Iterable[Tuple[Path, Path]]:
    """
    Yield (image_path, label_path) pairs.

    Parameters
    ----------
    img_root : Path
        Root under which images reside (recursed with rglob).
    lbl_root : Path | None
        If provided, look for labels here using the *same relative path* as
        the image but with `.txt` suffix.  If None, search next to the image.

    Yields
    ------
    Tuple[Path, Path]
        (image, label) paths that both exist.
    """
    for img in img_root.rglob("*"):
        if img.suffix not in IMG_EXT:
            continue

        if lbl_root is None:
            lbl = img.with_suffix(".txt")
        else:
            rel  = img.relative_to(img_root).with_suffix(".txt")
            lbl  = lbl_root / rel

        if lbl.exists():
            yield img, lbl


def copy_pair(img: Path, lbl: Path, dst_img_dir: Path, dst_lbl_dir: Path, prefix: str):
    """
    Copy an (image, label) pair to destination dirs with a unique stem.

    Files are renamed   `<prefix>_<original-stem>.<ext>`   to avoid clashes.
    """
    stem = f"{prefix}_{img.stem}"
    dst_img = dst_img_dir / f"{stem}{img.suffix.lower()}"
    dst_lbl = dst_lbl_dir / f"{stem}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst_img)
    shutil.copy2(lbl, dst_lbl)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="Root folder with 01_dataset, 02_dataset, 03_dataset")
    ap.add_argument("--out-root", required=True, help="Destination folder (YOLO structure is created here)")
    ap.add_argument("--include-private", action="store_true",
                    help="Move *private* validation into train (⚠ may leak!)")
    args = ap.parse_args()

    raw  = Path(args.raw_root).expanduser().resolve()
    out  = Path(args.out_root).expanduser().resolve()
    (out / "images/train").mkdir(parents=True, exist_ok=True)
    (out / "labels/train").mkdir(parents=True, exist_ok=True)
    (out / "images/val").mkdir(parents=True, exist_ok=True)
    (out / "labels/val").mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────── 1 & 2 ──
    train_sources = [
        raw / "01_dataset" / "01_train-s1__DataSet_Human_Rescue",
        raw / "02_dataset" / "02_second_part_DataSet_Human_Rescue",
    ]
    for src in train_sources:
        img_dir = src / "images"
        lbl_dir = src / "labels"
        prefix  = src.name.replace("__", "_")
        pairs   = gather_pairs(img_dir, lbl_dir)

        for img, lbl in tqdm(pairs, desc=f"{prefix:35s}", unit="img"):
            copy_pair(img, lbl,
                      out / "images/train",
                      out / "labels/train",
                      prefix)

    # ──────────────────────────────────────────────────────────────────────── 3 ──
    public_val = raw / "03_dataset" / "03_validation__DataSet_Human_Rescue" / "public"
    for img, lbl in tqdm(gather_pairs(public_val / "images", public_val / "labels"),
                         desc="public_val", unit="img"):
        copy_pair(img, lbl,
                  out / "images/val",
                  out / "labels/val",
                  "pub")

    # ──────────────────────────────────────────────────────────────────────── 4 ──
    private_val = raw / "03_dataset" / "03_validation__DataSet_Human_Rescue" / "private"
    if args.include_private:
        for img, lbl in tqdm(gather_pairs(private_val / "images", private_val / "labels"),
                             desc="private→train", unit="img"):
            copy_pair(img, lbl,
                      out / "images/train",
                      out / "labels/train",
                      "priv")
    else:
        print("⚠  private validation kept out of training (no leak).")

    # ---------------------------------------------------------------- YAML ------
    yaml_path = out / "uav_people.yaml"
    yaml_path.write_text(
f"""path: {out}
train: images/train
val: images/val
names:
  0: person
""")
    print(f"\n✅ Dataset ready →  {out}\n   YAML saved   →  {yaml_path}")


if __name__ == "__main__":
    main()
