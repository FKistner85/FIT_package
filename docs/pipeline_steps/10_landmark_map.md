# Step 10: Landmark Mapping

`create_landmark_map.py` analyses feature column names of the Eurasian
Otter dataset to build a JSON map of landmarks and points. It relies on
`load_raw_files` and writes `otter_landmark_map.json` and
`otter_point_map.json`.

**Key Functions**
- `load_raw_files`

**Algorithmic Cost**
- String parsing over feature names; negligible for typical dataset size.


