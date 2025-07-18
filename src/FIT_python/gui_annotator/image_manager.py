"""Image loading and preprocessing utilities for the annotation GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from FIT_python.soft_config import SOFT_CONFIG

CFG = SOFT_CONFIG.get("gui_annotator", {})
RAW_DIR = Path(CFG.get("raw_image_dir", "data/raw/images"))
PROCESSED_DIR = Path(CFG.get("processed_image_dir", "data/processed/images"))
DEFAULT_SCALE = float(CFG.get("default_scale", 1.0))


def load_image(image_path: Path) -> Image.Image:
    """Load an image from the given path."""
    return Image.open(image_path)


def preprocess_image(
    img: Image.Image,
    rotate_deg: float = 0.0,
    scale: Optional[float] = None,
    flip: bool = False,
) -> Image.Image:
    """Apply rotation, scaling and optional horizontal flip."""
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=True)
    s = DEFAULT_SCALE if scale is None else scale
    if s != 1.0:
        w, h = img.size
        img = img.resize((int(w * s), int(h * s)))
    return img


def load_and_preprocess(
    image_id: str,
    rotate_deg: float = 0.0,
    scale: Optional[float] = None,
    flip: bool = False,
) -> tuple[Image.Image, Path]:
    """Load image by ID from ``RAW_DIR`` and apply preprocessing."""
    img_path = RAW_DIR / f"{image_id}.jpg"
    img = load_image(img_path)
    img = preprocess_image(img, rotate_deg=rotate_deg, scale=scale, flip=flip)
    return img, img_path


def save_processed_image(img: Image.Image, image_id: str) -> Path:
    """Save processed image to ``PROCESSED_DIR`` and return the path."""
    out_path = PROCESSED_DIR / f"{image_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
