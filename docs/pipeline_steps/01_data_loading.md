# Step 1: Data Loading
## Overview

The pipeline begins by importing raw CSV or Excel files located under `data/raw`.
`load_raw_files` from `FIT_python.data_import_utils` reads all files, normalises
column names and identifier columns and optionally inserts an `id` column.
This `id` column is the only reliable identifier for individual footprints and
should be kept throughout all preprocessing steps.

## Key Functions
- `load_raw_files`
- `DataImporter.load`
- `DataPipeline.load`

## Algorithmic Cost
- File I/O dominates; complexity is linear in the number of rows.

