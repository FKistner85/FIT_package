#!/usr/bin/env python3
"""scripts/create_landmark_map.py

Generate landmark_map and point_map for Eurasian Otter dataset.
"""
import json
from pathlib import Path

from FIT_python.config import (
    RAW_DIR,
    PROCESSED_DIR,
    OTTER_LANDMARK_MAP_PATH,
    OTTER_POINT_MAP_PATH,
    OTTER_META_COLS,
    DEFAULT_TARGETS,
)
from FIT_python.data_import_utils import load_raw_files

def main():
    raw_dir = RAW_DIR
    lm_path = OTTER_LANDMARK_MAP_PATH
    pm_path = OTTER_POINT_MAP_PATH

    # Ensure output directory exists
    lm_path.parent.mkdir(parents=True, exist_ok=True)

    # Load only Otter data
    dfs = load_raw_files(raw_dir)
    # Raw files are keyed by their filename stem with spaces replaced by
    # underscores.  The only available otter dataset is "Eurasian Otter.csv"
    # which becomes "Eurasian_Otter".
    otter_df = dfs.get("Eurasian_Otter")
    if otter_df is None:
        raise FileNotFoundError(f"Eurasian_Otter not found in {raw_dir}")

    # Determine feature columns: exclude meta + target
    feature_cols = [
        c for c in otter_df.columns
        if c not in OTTER_META_COLS + DEFAULT_TARGETS
    ]

    # Build landmark_map
    landmark_map = {}
    for col in feature_cols:
        parts = col.split("_")
        key   = parts[0]            # 'dist' / 'ang' / 't'
        nums  = [int(p) for p in parts[1:]]
        landmark_map[col] = {"type": key, "points": nums}

    # Invert to point_map
    point_map = {}
    for col, info in landmark_map.items():
        for pt in info["points"]:
            point_map.setdefault(str(pt), []).append(col)

    # Save maps
    with open(lm_path, "w", encoding="utf-8") as f:
        json.dump(landmark_map, f, indent=2)
    with open(pm_path, "w", encoding="utf-8") as f:
        json.dump(point_map, f, indent=2)

    print(f"Landmark map saved to {lm_path}")
    print(f"Point map     saved to {pm_path}")

if __name__ == "__main__":
    main()
