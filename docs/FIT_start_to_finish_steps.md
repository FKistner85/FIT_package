# FIT_start_to_finish.ipynb Steps

This file records the sequence of operations executed in `FIT_start_to_finish.ipynb`. Each entry summarises what a notebook block does so future additions can simply append new descriptions.

## Dataset Summary

The notebook first calls `generate_dataset_overview` to build a CSV table describing all raw datasets. The function loops through each species, counts footprints, individuals and trails, and stores the results in `dataset_overview.csv`.

## Data Cleaning

`DataImportWrapper().clean_all()` converts the raw CSV files into cleaned Parquet
tables under `data/cleaned`. The notebook creates the files
`data/cleaned/*.parquet` before any of the subsequent plotting steps are
executed.

## Data Splits

Next, `SplitWrapper().split_all()` writes train/test splits for each species under `data/splits`. Afterwards `run_summary(SPLITS_DIR, EXP_DIR/'split_summary.csv', EXP_DIR/'split_fig')` generates a CSV with split statistics and saves bar charts. Only the Eurasian otter summary and the overall footprint fraction plots are shown in the notebook output.

## Correlation Matrix

Next, `plot_feature_correlation_matrix` creates a 2×2 grid of Pearson correlation heatmaps for all numeric features and for the subsets of distance, angle and triangle features. The figure is saved in the experiment directory with a shared colourbar.

## Top Features and UMAP Visualisation

`select_top_features` is used twice to find the four most discriminative measurements for sex and for individual identification. `plot_sex_boxplots` and `plot_individual_boxplots` then produce box plots for these features. Afterwards a supervised `DimensionalityReducerTransformer` computes a UMAP embedding of the otter data and `plot_umap_by_individual` visualises it with markers for each individual. All plots are written to the experiment directory.

## Baseline Sex Classification

The notebook then calls `run_simple_baseline_all_species(EXP_DIR, n_jobs=-1)` to train stepwise LDA sex classifiers for each species. Accuracy and majority-vote comparison plots are produced and stored in the same experiment directory.

## Baseline Evaluation

Finally, `predict_all('eurasian_otter')` applies the best otter classifier to the held-out test data. The subsequent calls to `plot_confusion_and_inference`, `plot_quality`, `plot_quality_heatmaps` and `plot_individual_probabilities` create confusion matrices, quality plots and individual-level probability charts. Each figure is saved to the experiment directory for later inspection.

## Individual Identification Baseline

`run_simple_baseline_otter` evaluates the otter splits with different numbers of
selected features. For each value in the provided ``k_range`` the helper
`sequential_holdout.run` performs several iterations and writes a
``summary_k<k>.csv`` file. The ``k`` yielding the highest mean BCR is returned
as the optimal feature count【F:src/FIT_python/pipeline_individual_id/simple_baseline.py†L88-L126】.

`run_baseline_all_species` then runs one sequential holdout for every species
using this ``best_k`` (or species-specific overrides) and stores a
``summary.csv`` in each subdirectory【F:src/FIT_python/pipeline_individual_id/simple_baseline.py†L129-L160】【F:src/FIT_python/pipeline_individual_id/sequential_holdout.py†L189-L198】.

Example pair plots and Ward dendrograms can be produced from the resulting CSV
files via `plot_pair_examples` and `plot_dendrogram`【F:src/FIT_python/Visualisations/plots_utils.py†L422-L490】.
