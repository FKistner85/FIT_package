# Step 9: Summary and Visualisation
## Overview

`create_summary.py` compiles statistics for each dataset split. The
`summary.py` CLI calls `run_summary` for each species, which in turn uses functions from
`summary_utils` to aggregate counts and optionally create plots. Output files are
written to the species-specific `figures` and `tables` directories resolved by
`get_species_paths`.

## Key Functions
- `summarize_dataset`
- `summary_stats`
- `run_summary`
- `plot_summary_table`

## Algorithmic Cost
- Linear in the number of rows for summaries; plotting cost depends on
  number of categories.

