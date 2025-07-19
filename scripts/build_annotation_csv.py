#!/usr/bin/env python
"""Build a combined annotation CSV from JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import numpy as np

from FIT_python.soft_config import SOFT_CONFIG

# Paths ----------------------------------------------------------------------
ANNOTATION_DIR = Path(SOFT_CONFIG.get("gui_annotator", {}).get("annotation_dir", "data/processed/annotations"))
REF_FILE = (
    Path(
        SOFT_CONFIG.get("gui_annotator", {}).get(
            "reference_template_dir", "data/raw/reference_templates"
        )
    )
    / "landmarks_extended.json"
)


# Geometry helpers -----------------------------------------------------------
def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def _angle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Return angle ABC in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = math.hypot(*ba)
    norm_bc = math.hypot(*bc)
    if norm_ba == 0 or norm_bc == 0:
        return float("nan")
    cos = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _triangle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c
    return 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


# Derived landmark logic -----------------------------------------------------
_DERIVED_PAIRS = [
    (0, 1),  # -> landmark 12
    (1, 2),  # -> landmark 13
    (2, 3),  # -> landmark 14
    (3, 4),  # -> landmark 15
    (5, 6),  # -> landmark 16
    (6, 7),  # -> landmark 17
]


def _midpoint(p1: Iterable[float], p2: Iterable[float]) -> list[float]:
    x1, y1 = p1
    x2, y2 = p2
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _compute_derived(landmarks: list[list[float]]) -> list[list[float]]:
    if len(landmarks) != 11:
        raise ValueError(f"expected 11 manual landmarks, got {len(landmarks)}")
    return [_midpoint(landmarks[i], landmarks[j]) for i, j in _DERIVED_PAIRS]


# Load variable definitions --------------------------------------------------
with REF_FILE.open("r", encoding="utf-8") as fh:
    ref_data = json.load(fh)
VARIABLES: Dict[str, Dict[str, Any]] = ref_data.get("variables", {})


def _load_landmarks(data: Dict[str, Any]) -> list[list[float]]:
    lms = []
    for lm in data.get("landmarks", []):
        if isinstance(lm, dict):
            lms.append([lm.get("x", float("nan")), lm.get("y", float("nan"))])
        else:
            x, y = lm[:2]
            lms.append([x, y])
    if len(lms) == 11:
        lms += _compute_derived(lms)
    return lms


def _compute_measurements(lms: list[list[float]]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for name, spec in VARIABLES.items():
        idxs = [i - 1 for i in spec.get("landmarks", [])]
        if any(i >= len(lms) for i in idxs):
            feats[name] = float("nan")
            continue
        pts = [lms[i] for i in idxs]
        if spec.get("type") == "distance":
            feats[name] = _distance(pts[0], pts[1])
        elif spec.get("type") == "angle":
            feats[name] = _angle(pts[0], pts[1], pts[2])
        elif spec.get("type") == "triangle":
            feats[name] = _triangle(pts[0], pts[1], pts[2])
        else:
            feats[name] = float("nan")
    return feats


def build_records() -> list[Dict[str, Any]]:
    records = []
    for fp in sorted(ANNOTATION_DIR.glob("*.json")):
        with fp.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        lms = _load_landmarks(data)
        meas = _compute_measurements(lms)
        meta = data.get("metadata", {})
        record: Dict[str, Any] = {
            "file": fp.name,
            "image_path": data.get("image_path"),
            "rotation_deg": data.get("rotation_deg"),
            "pixels_per_cm": data.get("pixels_per_cm"),
            "pixels_per_cm_sd": data.get("pixels_per_cm_sd"),
        }
        record.update(meta)
        for i in range(len(lms)):
            record[f"lm{i+1}_x"] = lms[i][0]
            record[f"lm{i+1}_y"] = lms[i][1]
        record.update(meas)
        records.append(record)
    return records


def main() -> None:
    records = build_records()
    df = pd.DataFrame(records)
    out_path = Path("data/processed/annotations.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
