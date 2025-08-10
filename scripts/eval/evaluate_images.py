"""
Evaluate per-image metric with default model settings (no threshold search)
===========================================================================

- Loads images, runs `predict()` with its default config.
- Builds a predictions DataFrame in the required COLUMNS format.
- Computes a per-image F-beta score (averaged across IoU thresholds).
- Saves:
    * predictions.csv
    * per_image_metrics.csv (sorted desc by score)
    * visualizations/<image_id> with GT (green) and Pred (red) boxes
- Can limit the number of processed images and shuffle with a seed.

Usage
-----
python evaluate_images.py \
  --img_dir /path/to/images \
  --gt_csv /path/to/gt.csv \
  --out_dir runs/eval_default \
  --limit 200 --shuffle --seed 42
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Your solution entrypoints
from solution import predict  # predict(list_of_rgb_images) -> list[list[dict]]
                              # dict fields: label, xc, yc, w, h, score (all normalized except score/label)

# Import metric helpers so logic stays consistent with your competition metric
sys.path.append("solutions/examples")
from metric import (
    COLUMNS,
    df_to_bytes,
    set_types,
    bytes_to_df,
    get_box_coordinates,
    process_image,  # does IoU matching + tp/fp/fn per threshold
)

DEFAULT_THRESHOLDS = np.round(np.arange(0.3, 1.0, 0.07), 2)  # same as evaluate()
DEFAULT_BETA = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Per-image evaluation and visualization (no threshold search)."
    )
    p.add_argument("--img_dir", required=True, help="Directory with images")
    p.add_argument("--gt_csv", required=True, help="Path to ground truth CSV")
    p.add_argument("--out_dir", default="runs/eval_default", help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Process only first N images (after optional shuffle)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle image order before limiting")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (if --shuffle)")
    p.add_argument("--thresholds_start_stop_step", nargs=3, type=float,
                   metavar=("START", "STOP", "STEP"),
                   default=(0.3, 1.0, 0.07),
                   help="IoU thresholds grid (same as metric defaults)")
    p.add_argument("--beta", type=float, default=1.0, help="F-beta beta value")
    return p.parse_args()


def get_image_paths(img_dir: str, exts: Optional[List[str]] = None) -> List[Path]:
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".bmp"]

    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    paths = []
    for ext in exts:
        paths.extend(img_dir.glob(f"**/*{ext}"))
        paths.extend(img_dir.glob(f"**/*{ext.upper()}"))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(f"No images found in {img_dir}")
    return paths


def load_gt(gt_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    df = set_types(df)
    df["image_id"] = df["image_id"].astype(str)
    return df


def make_pred_rows(image_id: str, w_img: int, h_img: int, detections: List[dict], time_spent: float) -> List[dict]:
    rows = []
    if detections:
        for d in detections:
            rows.append({
                "image_id": image_id,
                "label": int(d["label"]),
                "xc": float(d["xc"]),
                "yc": float(d["yc"]),
                "w": float(d["w"]),
                "h": float(d["h"]),
                "w_img": int(w_img),
                "h_img": int(h_img),
                "score": float(d["score"]),
                "time_spent": float(time_spent),
            })
    else:
        # keep timing row even if no detections
        rows.append({
            "image_id": image_id,
            "label": 0,
            "xc": np.nan,
            "yc": np.nan,
            "w": np.nan,
            "h": np.nan,
            "w_img": int(w_img),
            "h_img": int(h_img),
            "score": np.nan,
            "time_spent": float(time_spent),
        })
    return rows


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    if denom <= 0:
        return 0.0
    return (1 + b2) * tp / denom


def compute_per_image_metric(
    pred_df_img: pd.DataFrame,
    gt_df_img: pd.DataFrame,
    thresholds: np.ndarray,
    beta: float
) -> float:
    """
    Robust per-image F-beta:
    - Drops NaN geometry in predictions before matching.
    - Handles empty GT / empty pred cases like the global metric.
    - Averages F-beta across IoU thresholds using the same greedy matching (process_image).
    """
    # Ensure no NaN geometry reaches process_image()
    if not pred_df_img.empty:
        pred_df_img = pred_df_img.dropna(subset=["xc", "yc", "w", "h"])

    # All empty
    if gt_df_img.empty and pred_df_img.empty:
        return 1.0

    # GT present, no predictions -> all FN
    if not gt_df_img.empty and pred_df_img.empty:
        num_gt = len(gt_df_img)
        return float(np.mean([fbeta_from_counts(0, 0, num_gt, beta) for _ in thresholds]))

    # Predictions present, no GT -> all FP
    if gt_df_img.empty and not pred_df_img.empty:
        num_pred = len(pred_df_img)
        return float(np.mean([fbeta_from_counts(0, num_pred, 0, beta) for _ in thresholds]))

    # Both present -> use contest matching
    res = process_image(pred_df_img, gt_df_img, thresholds)
    scores = [fbeta_from_counts(res[t]["tp"], res[t]["fp"], res[t]["fn"], beta) for t in thresholds]
    return float(np.mean(scores)) if scores else 0.0


def _fill_translucent_rect(img, pt1, pt2, color_bgr, alpha=0.15):
    """Fill rectangle with translucency."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def _draw_dashed_rect(img, pt1, pt2, color_bgr, thickness=1, dash_len=8, gap_len=6):
    """Dashed rectangle using short line segments."""
    x1, y1 = pt1
    x2, y2 = pt2

    def _draw_dashed_line(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            return
        vx = (x2 - x1) / length
        vy = (y2 - y1) / length
        pos = 0
        while pos < length:
            x_start = int(x1 + vx * pos)
            y_start = int(y1 + vy * pos)
            x_end   = int(x1 + vx * min(pos + dash_len, length))
            y_end   = int(y1 + vy * min(pos + dash_len, length))
            cv2.line(img, (x_start, y_start), (x_end, y_end), color_bgr, thickness, cv2.LINE_AA)
            pos += dash_len + gap_len

    # top, right, bottom, left
    _draw_dashed_line((x1, y1), (x2, y1))
    _draw_dashed_line((x2, y1), (x2, y2))
    _draw_dashed_line((x2, y2), (x1, y2))
    _draw_dashed_line((x1, y2), (x1, y1))

def _draw_translucent_circle(img, center, radius, color_bgr, alpha=0.18):
    """Translucent filled circle for a soft attention glow."""
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def _draw_ring(img, center, radius, color_bgr, thickness=2):
    """Crisp ring to mark the exact attention point."""
    cv2.circle(img, center, radius, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)


def draw_boxes(
    image_bgr: np.ndarray,
    gt_df_img: pd.DataFrame,
    pred_df_img: pd.DataFrame,
    out_path: Path
) -> None:
    """
    Z-order: GT (green, translucent fill + thin outline) UNDER predictions.
             Predictions (red, dashed) OVER GT.
    Adds yellow attention circles at bbox centers.
    """
    vis = image_bgr.copy()
    h_img, w_img = vis.shape[:2]

    # Slender lines
    gt_thickness = 1
    pred_thickness = 1

    # Colors (BGR)
    GREEN  = (0, 200, 0)
    RED    = (0, 0, 255)
    YELLOW = (0, 255, 255)

    def _center_and_radius(x1, y1, x2, y2):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        # radius ~ min dimension / 6, clamped
        r  = int(np.clip(min(bw, bh) / 6, 4, 18))
        return (cx, cy), r

    # 1) Draw GT first (background)
    if gt_df_img is not None and not gt_df_img.empty:
        for _, row in gt_df_img.iterrows():
            x1, y1, x2, y2 = get_box_coordinates(row)
            # translucent fill + thin outline
            _fill_translucent_rect(vis, (x1, y1), (x2, y2), GREEN, alpha=0.15)
            cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN, gt_thickness, cv2.LINE_AA)

            # yellow attention circle at center (subtle for GT)
            (cx, cy), r = _center_and_radius(x1, y1, x2, y2)
            _draw_translucent_circle(vis, (cx, cy), r, YELLOW, alpha=0.14)
            _draw_ring(vis, (cx, cy), r, YELLOW, thickness=2)

    # 2) Draw Predictions on top (foreground): dashed red
    if pred_df_img is not None and not pred_df_img.empty:
        for _, row in pred_df_img.iterrows():
            if any(np.isnan([row.get("xc", np.nan), row.get("yc", np.nan), row.get("w", np.nan), row.get("h", np.nan)])):
                continue
            x1, y1, x2, y2 = get_box_coordinates(row)
            _draw_dashed_rect(vis, (x1, y1), (x2, y2), RED, thickness=pred_thickness, dash_len=8, gap_len=6)

            # stronger attention circle for predictions
            (cx, cy), r = _center_and_radius(x1, y1, x2, y2)
            _draw_translucent_circle(vis, (cx, cy), r, YELLOW, alpha=0.22)
            _draw_ring(vis, (cx, cy), r, YELLOW, thickness=2)

            # label
            label = f"P{int(row['label'])}:{row['score']:.2f}"
            y_text = max(0, y1 - 5)
            cv2.putText(vis, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)




def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.round(
        np.arange(*args.thresholds_start_stop_step),
        2
    )
    beta = float(args.beta)

    # Load data
    gt_df = load_gt(args.gt_csv)
    # Index by image_id for quick slicing; keep a copy of columns
    gt_by_id = gt_df.set_index("image_id", drop=False).sort_index()

    image_paths = get_image_paths(args.img_dir)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(image_paths)

    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    all_rows = []
    image_metrics = []

    # Process sequentially for stable memory + per-image visualizations
    for img_path in tqdm(image_paths, desc="Evaluating images"):
        image_id = img_path.name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            # Skip unreadable images
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = bgr.shape[:2]

        start = time.time()
        try:
            dets_list = predict([rgb])  # list, one item for this image
            detections = dets_list[0] if dets_list else []
        except Exception as e:
            print(f"Prediction failed for {image_id}: {e}")
            detections = []
        elapsed = round(time.time() - start, 4)

        # Collect prediction rows (keeps a NaN stub row when empty)
        rows = make_pred_rows(image_id, w_img, h_img, detections, elapsed)
        all_rows.extend(rows)

        # Tiny DFs for metric & viz
        pred_df_img = pd.DataFrame(rows, columns=COLUMNS).set_index("image_id")

        # GT for this image_id (may be empty)
        if image_id in gt_by_id.index:
            sel = gt_by_id.loc[image_id]
            gt_df_img = sel.to_frame().T if isinstance(sel, pd.Series) else sel
        else:
            gt_df_img = pd.DataFrame(columns=gt_by_id.columns)

        # Compute per-image metric (drop NaN geometry inside the function)
        score = compute_per_image_metric(
            pred_df_img=pred_df_img,
            gt_df_img=gt_df_img.set_index("image_id") if "image_id" in gt_df_img.columns else gt_df_img,
            thresholds=thresholds,
            beta=beta
        )
        image_metrics.append({"image_id": image_id, "metric": float(score)})

        # Save visualization (this call skips NaN rows for drawing)
        try:
            draw_boxes(
                image_bgr=bgr,
                gt_df_img=gt_df_img if "image_id" in gt_df_img.columns else gt_df_img.reset_index(),
                pred_df_img=pred_df_img.reset_index(),
                out_path=vis_dir / image_id
            )
        except Exception as e:
            print(f"Viz failed for {image_id}: {e}")

        # Explicit cleanup
        del bgr, rgb, detections, pred_df_img, gt_df_img

    # Save predictions.csv (includes NaN stub rows for empty predictions)
    pred_df_all = pd.DataFrame(all_rows, columns=COLUMNS)
    pred_df_all.to_csv(out_dir / "predictions.csv", index=False)

    # Save per_image_metrics.csv (sorted desc)
    metrics_df = pd.DataFrame(image_metrics)
    metrics_df = metrics_df.sort_values("metric", ascending=False)
    metrics_df.to_csv(out_dir / "per_image_metrics.csv", index=False)

    # Save simple list
    with (out_dir / "images_sorted_by_metric.txt").open("w") as f:
        for _, row in metrics_df.iterrows():
            f.write(f"{row['image_id']}\t{row['metric']:.6f}\n")

    print("\nDone.")
    print(f"- Predictions saved to: {out_dir / 'predictions.csv'}")
    print(f"- Per-image metrics saved to: {out_dir / 'per_image_metrics.csv'}")
    print(f"- Sorted list saved to: {out_dir / 'images_sorted_by_metric.txt'}")
    print(f"- Visualizations in: {vis_dir}")


if __name__ == "__main__":
    main()
