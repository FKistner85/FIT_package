#!/usr/bin/env python
"""Compute derived landmark coordinates and update annotation JSON.

This utility expects a JSON file produced by the GUI annotator containing
an ``image_path`` key and a ``landmarks`` list with the 11 manually
selected landmark coordinates.  It computes six additional derived
landmarks and appends them to the list in-place.

Derived landmarks are simple midpoints between predefined pairs of
manual landmarks.  The mapping is intentionally straightforward so the
script can run without external dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

# Pairs of manual landmark indices (0-based) used to compute derived points
# yielding landmarks 12..17.  Adjust as needed for other datasets.
_DERIVED_PAIRS: List[tuple[int, int]] = [
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


def compute_derived(landmarks: list[list[float]]) -> list[list[float]]:
    """Return the six derived points given the 11 manual landmarks."""
    if len(landmarks) != 11:
        raise ValueError(f"expected 11 manual landmarks, got {len(landmarks)}")
    derived = [_midpoint(landmarks[i], landmarks[j]) for i, j in _DERIVED_PAIRS]
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description="Add derived landmarks to JSON")
    parser.add_argument("json_file", type=Path, help="annotation JSON file")
    args = parser.parse_args()

    with args.json_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    landmarks = data.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError("JSON file missing 'landmarks' list")

    derived = compute_derived(landmarks)
    data["landmarks"] = landmarks + derived

    with args.json_file.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(f"[INFO] added {len(derived)} derived landmarks to {args.json_file}")


if __name__ == "__main__":
    main()
