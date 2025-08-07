"""
Convert YOLO-format annotations to hackathon ground-truth CSV
============================================================

The output columns match the metric script:
    image_id,label,xc,yc,w,h,w_img,h_img

•  *xc, yc, w, h* are **already normalised** (0‒1).  
•  w_img and h_img are the original pixel dimensions of each image.  
•  Images that have **no annotation file** or an *empty* .txt are skipped
   (the metric tolerates that).

Example
-------
python yolo_to_gt_csv.py \
    --img_dir /path/to/images \
    --txt_dir /path/to/yolo_txts \
    --output /path/to/public_gt_solution.csv
"""
from pathlib import Path
import argparse, csv
from typing import List, Tuple

from tqdm import tqdm
import cv2


#  ----- Command-line arguments -----
def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments and returns an argparse.Namespace object. Awaits
    the following arguments:
        --img_dir:   Root folder containing images (required).
        --txt_dir:   Root folder containing YOLO .txt files (required).
        --output:    Destination CSV file (default: 'public_gt_solution.csv').
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--img_dir', required=True, help='Root folder containing images')
    p.add_argument('--txt_dir', required=True, help='Root folder containing YOLO .txt files')
    p.add_argument('--output',   default='public_gt_solution.csv', help='Destination CSV file')

    return p.parse_args()


#  ----- Helpers -----
def read_txt(txt_path: Path) -> List[Tuple[int,float,float,float,float]]:
    """
    Parses a YOLO annotation file. Each line:
        cls xc yc w h

    All five values are space-separated, floats except cls (int).

    Args:
        txt_path (Path): Path to the YOLO .txt file.
    
    Returns:
        List[Tuple[int,float,float,float,float]]: List of tuples with class and normalized bbox coordinates.
    """
    # If no .txt file exists, return empty list
    if not txt_path.is_file():
        return []
    
    # Open the .txt file and read its contents
    with txt_path.open('r') as f:
        rows = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f'malformed line in {txt_path}: "{line}"')
            
            cls, xc, yc, w, h = parts
            rows.append((int(cls), float(xc), float(yc), float(w), float(h)))

    return rows


def img_size(img_path: Path) -> Tuple[int,int]:
    """
    Reads the dimensions of an image file.

    Args:
        img_path (Path): Path to the image file.

    Returns:
        Tuple[int,int]: Width and height of the image in pixels.
    """
    im = cv2.imread(str(img_path))

    # If the image cannot be opened, raise an error
    if im is None:
        raise FileNotFoundError(f'Cannot open image: {img_path}')
    
    # Get the dimensions of the image
    h, w = im.shape[:2]
    return w, h



# ----- Main -----
def main() -> None:
    """
    Main function to convert YOLO annotations to ground-truth CSV format.
    It scans the specified directory for images, reads their corresponding YOLO
    annotations, and writes the results to a CSV file.

    The output CSV will contain the following columns:
        image_id,label,xc,yc,w,h,w_img,h_img
    """
    # Parse command-line arguments
    opt = parse_args()

    # Prepare paths
    img_dir = Path(opt.img_dir)
    txt_dir = Path(opt.txt_dir)

    # Validate paths
    if not img_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {img_dir}')
    if not txt_dir.is_dir():
        raise FileNotFoundError(f'Text directory not found: {txt_dir}')
    
    # Supported image extensions
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    rows = []
    # iterate over all images in the directory (only those with the specified extensions)
    for img_path in tqdm(img_dir.glob('**/*'), desc='Processing images', unit='image'):
        # Skip if not a file or not an image with the specified extension
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in img_exts:
            continue
        
        # Read the corresponding YOLO .txt file
        txt_path = txt_dir / (img_path.stem + '.txt')
        annos = read_txt(txt_path)
        if not annos:
            # No ground truth for this image → skip (metric can handle it)
            continue
        
        # If we have annotations, add them to the rows list
        w_img, h_img = img_size(img_path)
        for label, xc, yc, w, h in annos:
            rows.append([img_path.name, label, xc, yc, w, h, w_img, h_img])

    if not rows:
        print('❗ No annotations found. Exiting.')
        return

    # Write the results to a CSV file
    out_path = Path(opt.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id','label','xc','yc','w','h','w_img','h_img'])
        writer.writerows(rows)

    print(f'✅ Wrote {len(rows):,} rows → {out_path.resolve()}')


if __name__ == '__main__':
    main()
