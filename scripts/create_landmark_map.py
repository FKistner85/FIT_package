#!/usr/bin/env python3
"""scripts/create_landmark_map.py

Generate landmark_map and point_map for Eurasian Otter dataset.
"""
import json
from pathlib import Path
from FIT_python.config import RAW_DIR, PROCESSED_DIR
from FIT_python.data_import_utils import load_raw_files

def main():
    raw_dir = RAW_DIR
    processed_dir = PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load only Otter data
    dfs = load_raw_files(raw_dir)
    otter_df = dfs.get('Eurasian_Otter')
    if otter_df is None:
        raise FileNotFoundError('Eurasian_Otter dataset not found in data/raw')

    # Determine feature columns: exclude meta+target
    meta_cols = ['id', 'date', 'location', 'dataorigin', 'substrate']
    target_cols = ['species', 'individual_id', 'trail', 'sex']
    feature_cols = [c for c in otter_df.columns if c not in meta_cols + target_cols]

    # Build landmark_map
    landmark_map = {}
    for col in feature_cols:
        parts = col.split('_')
        key = parts[0]  # dist, ang, t...
        nums = [int(p) for p in parts[1:]]
        landmark_map[col] = {'type': key, 'points': nums}

    # Invert to point_map
    point_map = {}
    for col, info in landmark_map.items():
        for pt in info['points']:
            point_map.setdefault(str(pt), []).append(col)

    # Save maps
    lm_path = processed_dir / 'otter_landmark_map.json'
    pm_path = processed_dir / 'otter_point_map.json'
    with open(lm_path, 'w') as f:
        json.dump(landmark_map, f, indent=2)
    with open(pm_path, 'w') as f:
        json.dump(point_map, f, indent=2)

    print(f'Landmark map saved to {lm_path}')
    print(f'Point map saved to {pm_path}')

if __name__ == '__main__':
    main()
