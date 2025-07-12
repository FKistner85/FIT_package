# feature_corr_utils.py

## Overview
Provides a helper to visualise correlations between feature groups.

## Key Components
- `plot_feature_correlations`

### plot_feature_correlations
Computes a correlation matrix using aggregated feature groups and
saves a heatmap. Features starting with `d` or `dist` map to the
*distance* group, `ang` to *angle*, and `t` (excluding `trail`) to
*triangles*. When fewer than two groups are found, the heatmap is
created for all numeric features without axis labels.
