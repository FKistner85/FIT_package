# FIT_start_to_finish.ipynb Steps

This file records the sequence of operations executed in `FIT_start_to_finish.ipynb`. Each entry summarises what a notebook block does so future additions can simply append new descriptions.

## Dataset Summary

The notebook first calls `generate_dataset_overview` to build a CSV table describing all raw datasets. The function loops through each species, counts footprints, individuals and trails, and stores the results in `dataset_overview.csv`.

## Data Splits

Next, `SplitWrapper().split_all()` writes train/test splits for each species under `data/splits`. Afterwards `run_summary(SPLITS_DIR, EXP_DIR/'split_summary.csv', EXP_DIR/'split_fig')` generates a CSV with split statistics and saves bar charts. Only the Eurasian otter summary and the overall footprint fraction plots are shown in the notebook output.

## Correlation Matrix

Next, `plot_feature_correlation_matrix` creates a 2×2 grid of Pearson correlation heatmaps for all numeric features and for the subsets of distance, angle and triangle features. The figure is saved in the experiment directory with a shared colourbar.
