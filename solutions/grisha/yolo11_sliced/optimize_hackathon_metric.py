"""
Grid-search optimal confidence threshold for hackathon metric
============================================================

This script optimizes the confidence threshold for YOLO11 sliced inference
by evaluating the hackathon metric on a test dataset.

Example
-------
python optimize_hackathon_metric.py \
    --img_dir /path/to/test/images \
    --gt_csv /path/to/gt.csv \
    --conf_range 0.05 0.95 0.05 \
    --save_csv --out_dir runs/hackathon_optimization
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Import the predict function and detection_model from the same directory
# Make sure to use sliced solution to non sliced dataset and vice versa
from solution import predict, detection_model

# Import metric evaluation from examples
sys.path.append("solutions/examples")
from metric import evaluate, df_to_bytes


# Expected CSV columns for hackathon metric
COLUMNS = ['image_id', 'label', 'xc', 'yc', 'w', 'h', 'w_img', 'h_img', 'score', 'time_spent']


#  ----- Command-line arguments -----
def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments and returns an argparse.Namespace object. Awaits
    the following arguments:

        --img_dir:   Directory containing test images (required).
        --gt_csv:    Path to ground truth CSV file (required).
        --conf_range: Grid of confidence thresholds (default: 0.05 0.95 0.05).
        --save_csv:  Flag to save full grid search results to CSV (default: False).
        --out_dir:   Output directory for results (default: runs/hackathon_optimization).
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Optimize confidence threshold for hackathon metric"
    )
    
    parser.add_argument('--img_dir', required=True, help='Directory containing test images')
    parser.add_argument('--gt_csv', required=True, help='Path to ground truth CSV file')
    parser.add_argument(
        '--conf_range', 
        nargs=3, 
        type=float, 
        metavar=('START', 'STOP', 'STEP'),
        default=(0.05, 0.95, 0.05),
        help='Grid of confidence thresholds'
    )
    parser.add_argument('--save_csv', action='store_true', help='Save full grid search results to CSV')
    parser.add_argument('--out_dir', type=str, default='runs/hackathon_optimization', help='Output directory for results')
    
    return parser.parse_args()


#  ----- Helpers -----
def frange(start: float, stop: float, step: float):
    """
    Floating-point range generator (inclusive of stop).

    Args:
        start (float): Start value of the range.
        stop (float): Stop value of the range.
        step (float): Step size for the range.

    Yields:
        float: Values in the range from start to stop, inclusive.
    """
    while start <= stop + 1e-9:
        yield start
        start += step


def load_ground_truth(gt_csv_path: str) -> bytes:
    """
    Load ground truth CSV and convert to bytes format for metric evaluation.

    Args:
        gt_csv_path (str): Path to the ground truth CSV file.

    Returns:
        bytes: Byte representation of the ground truth DataFrame.
    """
    # Check if the ground truth CSV exists
    if not os.path.exists(gt_csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_csv_path}")
    
    # Read the CSV into a DataFrame
    gt_df = pd.read_csv(gt_csv_path)
    return df_to_bytes(gt_df)


def get_image_paths(img_dir: str, img_exts: List[str]) -> List[Path]:
    """
    Get all image paths from the specified directory with given extensions.

    Args:
        img_dir (str): Directory containing images.
        img_exts (List[str]): List of image file extensions to search for.

    Returns:
        List[Path]: List of Paths to image files found in the directory.
    """
    img_dir = Path(img_dir)

    # Validate the image directory
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Ensure extensions are lowercase
    image_paths = []
    for ext in img_exts:
        image_paths.extend(img_dir.glob(f'**/*{ext}'))
        image_paths.extend(img_dir.glob(f'**/*{ext.upper()}'))
    
    # Filter out non-image files
    if not image_paths:
        raise ValueError(f"No images found in {img_dir} with extensions {img_exts}")
    
    # Sort for reproducible results
    image_paths.sort()
    return image_paths


def update_confidence_threshold(conf: float):
    """
    Update the global detection model's confidence threshold.

    Args:
        conf (float): New confidence threshold to set.
    """
    # Validate the confidence threshold
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf:.3f}")
    
    global detection_model
    detection_model.confidence_threshold = conf


def process_images_for_confidence(image_paths: List[Path], conf: float) -> pd.DataFrame:
    """
    Process images with the specified confidence threshold and return results as a DataFrame.

    Args:
        image_paths (List[Path]): List of image file paths to process.
        conf (float): Confidence threshold for predictions.

    Returns:
        pd.DataFrame: DataFrame containing predictions and metadata for each image.
    """
    # Update confidence threshold
    update_confidence_threshold(conf)
    
    # Iterate over images and run predictions
    results = []
    for image_path in tqdm(image_paths, desc=f"Processing conf={conf:.3f}", leave=False):
        image_id = image_path.stem
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        # Convert BGR to RGB for prediction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]
        
        # Measure inference time
        start_time = time.time()
        
        # Run prediction
        image_results = predict([image])
        
        # Calculate time spent
        elapsed_time = time.time() - start_time
        time_per_image = round(elapsed_time, 4)
        
        # Process results
        if image_results and image_results[0]:
            for detection in image_results[0]:
                result = {
                    'image_id': image_id,
                    'label': detection['label'],
                    'xc': detection['xc'],
                    'yc': detection['yc'],
                    'w': detection['w'],
                    'h': detection['h'],
                    'w_img': w_img,
                    'h_img': h_img,
                    'score': detection['score'],
                    'time_spent': time_per_image
                }
                results.append(result)
        else:
            # No detections - still need to record time spent
            result = {
                'image_id': image_id,
                'label': 0,
                'xc': np.nan,
                'yc': np.nan,
                'w': np.nan,
                'h': np.nan,
                'w_img': w_img,
                'h_img': h_img,
                'score': np.nan,
                'time_spent': time_per_image
            }
            results.append(result)
    
    return pd.DataFrame(results, columns=COLUMNS)


def evaluate_metric_for_confidence(pred_df: pd.DataFrame, gt_bytes: bytes, conf: float) -> float:
    """
    Evaluate the hackathon metric for a given confidence threshold.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predictions for the current confidence.
        gt_bytes (bytes): Byte representation of the ground truth DataFrame.
        conf (float): Confidence threshold used for predictions.

    Returns:
        float: Calculated hackathon metric score for the given confidence.
    """
    try:
        # Convert predictions to bytes format
        pred_bytes = df_to_bytes(pred_df)
        
        # Calculate metric using the hackathon evaluation function
        metric, accuracy, fp_rate, avg_time = evaluate(
            predicted_file=pred_bytes,
            gt_file=gt_bytes,
            thresholds=np.round(np.arange(0.3, 1.0, 0.07), 2),
            beta=1.0,
            m=500,
            parallelize=True
        )
        
        return float(metric)
        
    except Exception as e:
        print(f"Error evaluating metric for conf={conf:.3f}: {str(e)}")
        return 0.0


# ----- Main optimization routine -----
def main():
    """
    Main function to optimize confidence threshold for hackathon metric.
    Parses command-line arguments, loads ground truth, finds images, and performs grid search.
    """
    # Parse command-line arguments
    args = parse_args()
    
    print("🔍 Hackathon Metric Confidence Optimization")
    print("=" * 50)
    print(f"Image directory: {args.img_dir}")
    print(f"Ground truth CSV: {args.gt_csv}")
    print(f"Confidence range: {args.conf_range[0]:.3f} to {args.conf_range[1]:.3f} step {args.conf_range[2]:.3f}")
    
    # Load ground truth
    print("\n📋 Loading ground truth...")
    gt_bytes = load_ground_truth(args.gt_csv)
    
    # Get image paths
    print("🖼️  Finding images...")
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = get_image_paths(args.img_dir, img_exts)
    print(f"Found {len(image_paths)} images")
    
    # Generate confidence values
    conf_start, conf_stop, conf_step = args.conf_range
    conf_values = [round(c, 3) for c in frange(conf_start, conf_stop, conf_step)]
    
    print(f"\n⚡ Testing {len(conf_values)} confidence thresholds...")
    
    # Grid search
    best_score = -1.0
    best_conf = None
    grid_results = []
    
    for conf in tqdm(conf_values, desc="Optimizing confidence"):
        # Process images with current confidence
        pred_df = process_images_for_confidence(image_paths, conf)
        
        # Evaluate metric
        score = evaluate_metric_for_confidence(pred_df, gt_bytes, conf)
        grid_results.append((conf, score))
        
        if score > best_score:
            best_score = score
            best_conf = conf
        
        print(f"conf={conf:.3f} → metric={score:.5f}")
    
    # Results summary
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best confidence threshold: {best_conf:.3f}")
    print(f"Best metric score: {best_score:.5f}")
    print("="*50)
    
    # Save results
    if args.save_csv:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = out_dir / "hackathon_metric_optimization.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["conf", "hackathon_metric"])
            writer.writerows(grid_results)
        
        print(f"📊 Full results saved to: {csv_path.resolve()}")
        
        # Save summary
        summary_path = out_dir / "optimization_summary.txt"
        with summary_path.open("w") as f:
            f.write(f"Hackathon Metric Optimization Summary\n")
            f.write(f"=====================================\n")
            f.write(f"Image directory: {args.img_dir}\n")
            f.write(f"Ground truth CSV: {args.gt_csv}\n")
            f.write(f"Confidence range: {conf_start:.3f} to {conf_stop:.3f} step {conf_step:.3f}\n")
            f.write(f"Images processed: {len(image_paths)}\n")
            f.write(f"Best confidence: {best_conf:.3f}\n")
            f.write(f"Best metric score: {best_score:.5f}\n")
        
        print(f"📄 Summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
