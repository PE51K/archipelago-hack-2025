"""Gradio YOLO Annotation Viewer (keyboard-safe)

If your installed Gradio doesn't support Blocks.add_event_listener (introduced in v4),
this script gracefully falls back to button-only navigation.

Usage:
  pip install --upgrade gradio pillow  # upgrade Gradio if you want ←/→ keys
  python viewer_gradio.py
"""
import os
import glob
from typing import Tuple, Optional

import gradio as gr
from PIL import Image, ImageDraw

# ==================== CONFIG ====================
DATA_DIR = "data/merged_sliced/val"  # change if your folder differs
IMG_EXT = ".jpg"

# Gather image list once
IMAGES = sorted(
    [os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, f"*{IMG_EXT}"))]
)
if not IMAGES:
    raise FileNotFoundError(f"No {IMG_EXT} images found inside {DATA_DIR}.")


def draw_yolo_boxes(image: Image.Image, labels_path: str) -> Image.Image:
    """Overlay YOLO bounding boxes onto the image."""
    if not os.path.exists(labels_path):
        return image
    w, h = image.size
    draw = ImageDraw.Draw(image)
    with open(labels_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 2, y1 + 2), str(int(cls)), fill="red")
    return image


def load_image(idx: int) -> Tuple[Image.Image, str]:
    idx = idx % len(IMAGES)
    filename = IMAGES[idx]
    img_path = os.path.join(DATA_DIR, filename)
    label_path = img_path.replace(IMG_EXT, ".txt")
    img = Image.open(img_path).convert("RGB")
    img = draw_yolo_boxes(img, label_path)
    caption = f"{idx + 1}/{len(IMAGES)} : {filename}"
    return img, caption


def next_image(idx: int) -> Tuple[Image.Image, str, int]:
    return (*load_image(idx + 1), idx + 1)


def prev_image(idx: int) -> Tuple[Image.Image, str, int]:
    return (*load_image(idx - 1), idx - 1)


with gr.Blocks(title="YOLO Annotation Viewer") as demo:
    gr.Markdown("## YOLO Annotated Image Viewer")
    state = gr.State(0)  # index
    init_img, init_cap = load_image(0)
    image_display = gr.Image(value=init_img, interactive=False)
    caption_display = gr.Textbox(value=init_cap, interactive=False, label="Info")
    with gr.Row():
        prev_btn = gr.Button("⭠ Prev")
        next_btn = gr.Button("Next ⭢")

    prev_btn.click(prev_image, inputs=state, outputs=[image_display, caption_display, state])
    next_btn.click(next_image, inputs=state, outputs=[image_display, caption_display, state])

    # Optional keyboard navigation for Gradio ≥4.0
    if hasattr(demo, "add_event_listener"):
        demo.add_event_listener(
            "keyup",
            lambda evt, idx: prev_image(idx) if evt.key == "ArrowLeft" else next_image(idx) if evt.key == "ArrowRight" else None,
            inputs=state,
            outputs=[image_display, caption_display, state],
        )

demo.launch()