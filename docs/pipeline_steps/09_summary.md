# Step 9: Summary and Visualisation
## Overview

`create_summary.py` compiles statistics for each dataset split. The
`summary.py` CLI calls `run_summary` which in turn uses functions from
`summary_utils` to aggregate counts and optionally create plots.

## Key Functions
- `summarize_dataset`
- `summary_stats`
- `run_summary`
- `plot_summary_table`

## Algorithmic Cost
- Linear in the number of rows for summaries; plotting cost depends on
  number of categories.

