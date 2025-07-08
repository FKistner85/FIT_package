# transform_utils.py

## Overview
Utility functions to convert feature columns and encode targets.

## Key Components
- convert_numeric
- one_hot_encode_targets
- save_target_mapping

### convert_numeric
Replaces comma decimal separators and coerces selected columns to `float`. Use
this after reading raw CSV or Excel files when numeric values may have been
parsed as strings. Columns that cannot be converted become `NaN`, so subsequent
imputation or filtering may be required.

### one_hot_encode_targets
Performs one‑hot encoding of categorical target columns. Returns both the numpy
array and a mapping dictionary which records the generated column names. Useful
for feeding multiple target variables into machine learning models. Beware of
memory growth when many classes are present.

### save_target_mapping
Stores the mapping dictionary from `one_hot_encode_targets` to a JSON file.
Keeping this file alongside model artefacts ensures that class ordering remains
consistent when predictions are decoded later.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Handles comma decimal separators and JSON mapping files.
