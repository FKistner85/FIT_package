# FIT_start_to_finish.ipynb Steps

This file records the sequence of operations executed in `FIT_start_to_finish.ipynb`. Each entry summarises what a notebook block does so future additions can simply append new descriptions.

## Dataset Summary

The notebook first calls `generate_dataset_overview` to build a CSV table describing all raw datasets. The function loops through each species, counts footprints, individuals and trails, and stores the results in `dataset_overview.csv`. If the CSV already exists it is loaded instead of recomputed.

## Data Cleaning

`DataImportWrapper().clean_all()` converts the raw CSV files into cleaned Parquet
tables under `data/cleaned`. The notebook creates the files
`data/cleaned/*.parquet` before any of the subsequent plotting steps are
executed.

## Data Splits

Next, `SplitWrapper().split_all(reuse_splits=True)` writes train/test splits for each species under `data/splits`. When the files already exist they are reused. Afterwards `run_summary(SPLITS_DIR, EXP_DIR/'split_summary.csv', EXP_DIR/'split_fig')` generates a CSV with split statistics and saves bar charts. Only the Eurasian otter summary and the overall footprint fraction plots are shown in the notebook output.

## Correlation Matrix

Next, `plot_feature_correlation_matrix` creates a 2×2 grid of Pearson correlation heatmaps for all numeric features and for the subsets of distance, angle and triangle features. The figure is saved in the experiment directory with a shared colourbar.

## Top Features and UMAP Visualisation

`select_top_features` is used twice to find the four most discriminative measurements for sex and for individual identification. `plot_sex_boxplots` and `plot_individual_boxplots` then produce box plots for these features. Afterwards a supervised `DimensionalityReducerTransformer` computes a UMAP embedding of the otter data and `plot_umap_by_individual` visualises it with markers for each individual. All plots are written to the experiment directory.

## Baseline Sex Classification

The notebook then calls `run_simple_baseline_all_species(EXP_DIR, n_jobs=-1, progress=True, max_features=10)` to train stepwise LDA sex classifiers for each species. Accuracy and majority-vote comparison plots are produced and stored in the same experiment directory. The optional `progress=True` argument displays progress bars during cross-validation and `max_features` sets the number of features selected during forward selection. After training, `predict_simple_baseline('eurasian_otter', EXP_DIR)` loads the saved model and writes baseline predictions for the otter splits.

## Baseline Evaluation

Finally, `predict_all('eurasian_otter')` applies the best otter classifier to the held-out test data. The subsequent calls to `plot_confusion_and_inference`, `plot_quality`, `plot_quality_heatmaps` and `plot_individual_probabilities` create confusion matrices, quality plots and individual-level probability charts. Each figure is saved to the experiment directory for later inspection.


## Sex Feature Experiment

After benchmarking the baseline individual identification pipeline, the notebook calls `run_sex_prediction_experiment(EXP_DIR/'sex_feature')` to evaluate models with and without appended sex predictions. The resulting summary tables are loaded and passed to `plot_sex_feature_boxplots`, which compares BCR and count differences across setups. Example boxplots for the Eurasian otter and the aggregated results are displayed in the notebook.
## Sequential Holdout and Individual-ID Baseline

The notebook then runs `run_simple_baseline_otter` to determine the best feature
count for sequential holdouts on otter data. The chosen `k` and per‑species Ward
cut‑offs are passed to `run_baseline_all_species`, which evaluates all species.
Pair‑example and dendrogram plots are produced for each otter split. Every
species directory contains a `summary.csv`, the collected metrics are written to
`raw_results.csv`, and the resulting BCR comparison plot is saved as
`id_baseline/fig/bcr_comparison.png`.

## Distance Metric Comparison

The notebook concludes with an evaluation of the distance metrics computed for
each pair. Besides the centroid distance, it stores the mean and median values
across all projected points of both trails. The function
`compute_overlap_jsl_style` classifies pairs whose 50 % chi-square ellipses
overlap. Ward clustering is applied with metric-specific cut-offs selected by
`optimal_cutoff` to estimate population sizes. For Manhattan distances the
alternative `compute_overlap_rhombus` compares rhombus confidence regions. The
resulting figures include a BCR bar chart, a scatter plot of predicted versus
true population size and example pair visualisations.

