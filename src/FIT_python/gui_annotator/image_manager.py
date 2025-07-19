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
DISPLAY_SIZE = tuple(int(v) for v in CFG.get("display_size", [1280, 720]))


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


def extract_metadata(path: Path) -> dict:
    """Extract basic metadata from ``path``.

    The expected directory layout is ``RAW_DIR/species/sex/individual_id/trail/image.jpg``.
    If the hierarchy does not match this pattern, missing fields are set to ``None``.
    EXIF information is parsed using :func:`PIL.Image.open` to obtain GPS
    coordinates, timestamp, camera model and original image size.
    """

    meta = {
        "species": None,
        "sex": None,
        "individual_id": None,
        "trail": None,
        "timestamp": None,
        "camera_model": None,
        "gps": None,
        "width": None,
        "height": None,
    }

    # --- folder based information ---
    try:
        rel = path.resolve().relative_to(RAW_DIR)
    except Exception:
        rel = path

    parts = rel.parts
    if len(parts) >= 5:
        meta["species"] = parts[-5]
        meta["sex"] = parts[-4]
        meta["individual_id"] = parts[-3]
        meta["trail"] = parts[-2]

    # --- EXIF information ---
    try:
        img = Image.open(path)
        meta["width"], meta["height"] = img.size
        exif = img._getexif() or {}
    except Exception:
        exif = {}

    if exif:
        from PIL import ExifTags

        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        meta["timestamp"] = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        meta["camera_model"] = tag_map.get("Model")

        gps_raw = tag_map.get("GPSInfo")
        if gps_raw:
            gps_data = {ExifTags.GPSTAGS.get(t, t): gps_raw[t] for t in gps_raw}

            def _conv(coord):
                d, m, s = coord
                return float(d) + float(m) / 60.0 + float(s) / 3600.0

            lat = lon = None
            if {
                "GPSLatitude",
                "GPSLatitudeRef",
                "GPSLongitude",
                "GPSLongitudeRef",
            } <= gps_data.keys():
                lat = _conv(gps_data["GPSLatitude"])
                if gps_data["GPSLatitudeRef"] in {"S", "W"}:
                    lat = -lat
                lon = _conv(gps_data["GPSLongitude"])
                if gps_data["GPSLongitudeRef"] in {"S", "W"}:
                    lon = -lon

            altitude = None
            if "GPSAltitude" in gps_data:
                alt = gps_data["GPSAltitude"]
                if isinstance(alt, tuple):
                    altitude = alt[0] / alt[1] if alt[1] else None
                else:
                    altitude = float(alt)

            meta["gps"] = {
                k: v
                for k, v in {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": altitude,
                }.items()
                if v is not None
            } or None

    return meta
