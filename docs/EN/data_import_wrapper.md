# data_import_wrapper.py

## Overview
Wraps the import/clean/convert workflow for all raw datasets.

## Key Components
- DataImporter
- DataImportWrapper

### DataImporter
Instantiate this class with the raw data directory and call `run()` to obtain a
dictionary of cleaned `DataFrame` objects. It sequentially loads files,
sanitises label columns and converts numeric features. All methods assume that
pandas can read the source files. Large folders can take time to process and
memory usage grows with the number of loaded files.

### DataImportWrapper
Provides the `clean_all()` convenience method which writes the processed data to
parquet files under the configured output directory. Use this wrapper for the
one‑off preparation step before running any pipelines. Existing files will be
overwritten and missing directories cause the process to abort.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Designed for parquet output and numeric conversion before scaling.
