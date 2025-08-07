# summary_data_wrapper.py

## Overview
Runs summary creation for all split files and optional plotting.

## Key Components
- run_summary
- SummaryWrapper.summarize_all

### run_summary
Command style function that scans a directory of train/test splits, computes a
summary for each and optionally generates plots. Overwrite protection can be
disabled with the `force` flag. This is typically invoked via the wrapper class
but can also be called directly from scripts.

### SummaryWrapper.summarize_all
Convenient wrapper that calls `run_summary` using paths defined in the
configuration. It handles exceptions and returns a status code so it can be used
in automated pipelines. The method writes the final Parquet summary and any
figures to the results directory.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Writes Parquet summary and figures under results directory.
