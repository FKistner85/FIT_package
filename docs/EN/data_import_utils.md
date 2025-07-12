# data_import_utils.py

## Overview
Helper routines for loading raw CSV/Excel files and normalising column names.

## Key Components
- clean_columns
- load_raw_files
- coerce_numeric_columns
- sanitize_labels

### clean_columns
Normalises column labels by converting them to lower case, stripping spaces and
replacing non-word characters with underscores. Run this routine immediately
after loading raw tables so subsequent code can rely on consistent naming. It is
helpful when handling heterogeneous exports but may produce duplicate names if
two columns clean to the same label.

### load_raw_files
Reads every CSV or Excel file in the given directory, applies
`clean_columns` and optionally inserts an `id` column. The `id` values are
simple sequential integers starting at ``1`` for each file. All returned data
frames share the same normalised naming scheme. Unsupported file extensions are
skipped which means inconsistent layouts can easily be missed.

### coerce_numeric_columns
Searches object columns for values that resemble numbers with comma decimal
separators and converts them to floats. This should be called before scaling or
modelling so that numeric operations work as expected. Thousand separators are
not handled which can lead to mis‑parsed values.

### sanitize_labels
Cleans string labels by filling missing entries, mapping aliases and replacing
punctuation with underscores. Use this before training classification models to
ensure label categories match. If the mapping dictionary is incomplete the
result may contain unexpected placeholder values.

## References
- https://pandas.pydata.org/docs/
- https://scikit-learn.org/stable/modules/preprocessing.html

## Assumptions and Limitations
Assumes mostly numeric data with optional comma decimals.
