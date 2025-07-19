#!/usr/bin/env python
"""Validate saved annotation transforms by applying them to images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PyQt5 import QtGui

from FIT_python.utils.transformations import TransformationPipeline


def validate(annotation_dir: Path) -> None:
    for path in sorted(annotation_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        img_path = Path(data["image_path"])
        pixmap = QtGui.QPixmap(str(img_path))
        pipeline = TransformationPipeline.from_json(data.get("transforms", []))
        transformed = pipeline.apply_to_image(pixmap)
        print(f"{path.name}: {pixmap.size()} -> {transformed.size()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate coordinate reprojection")
    ap.add_argument("annotation_dir", type=Path)
    args = ap.parse_args()
    validate(args.annotation_dir)


if __name__ == "__main__":
    main()
