# Configuration Reference

This file lists the keys in `src/FIT_python/config.yaml`.

## paths
Relative directories used throughout the project.
- `data_dir`, `raw_dir`, `cleaned_dir`, `splits_dir`, `processed_dir`, `processed_splits_dir`
- `scaled_dir`, `feature_selected_dir`, `dim_reduced_dir`, `numeric_dir`
- `otter_landmark_map_path`, `otter_point_map_path`
- `results_dir`, `results_data_dir`, `figures_dir`
- `scripts_dir`, `notebooks_dir`

## data_split_and_summary
Options for the data preparation utilities.
- `otter_meta_cols`: columns kept for otter datasets.
- `default_targets`: target label columns.
- `group_col`, `stratify_col`: columns used for splitting.
- `global_random_seed`: RNG seed.
- `test_size`: fraction for test split.
- `num_folds`: number of folds for cross validation.

## pipeline_sex
Hyperparameter options for the sex classification pipeline.
- `allowed_fs`, `allowed_impute`, `allowed_outliers`, `allowed_scalers`, `allowed_reds`
- `outlier_clip_quantiles`, `outlier_zscore_threshold`
- `dim_reduction_n_components`

## pipeline_individual_id
Defaults for the individual identification utilities.
- `chunk_size`, `trail_size_list`
- `max_individuals`, `max_trails_per_animal`
- `n_folds`, `random_state`
- `fallback_col`, `sampling_mode`

## general_pipeline_steps
Reserved for future general settings.
