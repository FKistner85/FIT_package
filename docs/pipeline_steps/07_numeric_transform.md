# Step 7: Numeric Conversion
## Overview

`transform_splits.py` invokes `TransformWrapper` which converts Parquet
splits into NumPy arrays and encodes labels as integers. During this step,
`convert_numeric` also performs one-hot encoding for string-based feature
columns so the resulting arrays are fully numeric.

## Key Functions
- `TransformWrapper.transform_dataset`
- `convert_numeric`
- `one_hot_encode_targets`

## Algorithmic Cost
- Linear in the number of rows.

