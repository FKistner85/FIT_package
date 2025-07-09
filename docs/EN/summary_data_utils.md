# summary_data_utils.py

## Overview
Functions to summarise split statistics and generate bar plots.
Plot styling is configured via `FIT_python.plot_style.apply_style()`.

## Key Components
- compute_summary
- plot_summary_table
- plot_split_proportions

### compute_summary
Aggregates split statistics such as the number of footprints and individuals per
sex. Pass the split DataFrame along with dataset and origin strings. The
function returns a table ready for further reporting. Empty inputs result in an
empty DataFrame so downstream code should handle this case.

### plot_summary_table
Creates bar plots visualising the summary statistics. Requires Matplotlib and
writes figures to the specified directory. This is useful for quick exploratory
analysis but the colour scheme and layout are somewhat opinionated.

### plot_split_proportions
Takes the summary table and displays the relative fraction of footprints in the
train, test and inference splits for each species. The plot is saved to the
given directory and complements the absolute counts from `plot_summary_table`.

## References
- https://pandas.pydata.org/docs/
- https://matplotlib.org/

## Assumptions and Limitations
Expects columns like "sex" and "individual_id".
