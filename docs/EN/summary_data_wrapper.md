# summary_data_wrapper.py

## Overview
Runs summary creation for all split files and optional plotting.

## Key Components
- run_summary
- SummaryWrapper.summarize_all

### run_summary
Command style function that scans a directory of train/test splits for a single species, computes a
summary for each split and optionally generates plots. Output paths are resolved via
`get_species_paths(section="dataprocessing", species)` so tables and figures end up in the species
subdirectories. Overwrite protection can be disabled with the `force` flag.

### SummaryWrapper.summarize_all
Convenient wrapper that iterates over all species split directories and calls
`run_summary` for each one. It handles exceptions and returns a status code so it can be used
in automated pipelines. The method writes the Parquet summaries and any
figures to the corresponding species result directories.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Writes Parquet summary and figures under results directory.
