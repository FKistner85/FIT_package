# Internal Functions and Classes

## FIT_python.data_split_and_summary.data_import_utils.clean_columns
Clean column names: strip spaces, lowercase, replace non-alphanumeric with underscore.

## FIT_python.data_split_and_summary.data_import_utils.coerce_numeric_columns
Convert numeric-looking object columns with comma decimal separator.

## FIT_python.data_split_and_summary.data_import_utils.load_csv
Load a CSV file into a DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.load_excel
Load an Excel file into a DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.load_raw_files
Load all CSV and Excel files from 'folder', clean columns, optionally add 'id' column.
Returns a dict mapping file stem to DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.sanitize_labels
Clean string labels in target_cols:
  - fillna('unknown') except skip_fillna
  - strip & lower
  - replace non-word with underscore
  - apply mapping dict
Returns a new DataFrame.

## FIT_python.data_split_and_summary.data_import_wrapper.DataImportWrapper
High-level wrapper to clean all raw datasets and persist Parquet files.

## FIT_python.data_split_and_summary.data_import_wrapper.DataImporter
Class to import, clean, and convert data from raw files.

## FIT_python.data_split_and_summary.split_utils._check_valid
Check that each fold contains both classes 0 and 1.

## FIT_python.data_split_and_summary.split_utils._make_folds
Return ``(fold_ids, method_name)`` using the following strategy:
1) existing ``Fold`` column (predefined)
2) ``StratifiedGroupKFold`` via :func:`group_stratified_kfold`
3) ``GroupKFold``
4) ``KFold``

## FIT_python.data_split_and_summary.split_utils.create_train_test_split_otter
Create deterministic train/test/inference splits for Otter.
Returns (train_df, test_df, inference_df).

## FIT_python.data_split_and_summary.split_utils.ensure_valid_splits
Validate that each species in ``SPLITS_DIR`` has a valid ``Fold`` column
with both classes present in every fold.  Warns if any split is invalid.

## FIT_python.data_split_and_summary.split_utils.group_stratified_kfold
Assigns a 'Fold' column via StratifiedGroupKFold on groups,
stratified by stratify_col.

## FIT_python.data_split_and_summary.split_utils.sample_individuals
Randomly select `n` unique individuals for a dataset/sex combination.
Used for the fixed Otter test split.

## FIT_python.data_split_and_summary.split_utils.splits_available
Return ``True`` if at least one valid train/test pair exists.

## FIT_python.data_split_and_summary.split_utils.stratified_individual_split
Split ``df`` by ``group_col`` while stratifying by ``stratify_col``.
- All rows for one individual are kept together in train or test
- Invalid or missing groups are placed into ``inference_df``
- Mit ``add_folds=True`` erhält der Train-Split zusätzlich eine ``Fold``-Spalte
  aus einem stratifizierten Gruppen-KFold.

## FIT_python.data_split_and_summary.split_utils.train_test_group_split
Simple group-based train/test split.

## FIT_python.data_split_and_summary.split_wrapper.SplitWrapper
Wrapper, der Rohdaten bereinigt, splittet und Folds robust erzeugt.

## FIT_python.data_split_and_summary.summary_data_utils._lighten

## FIT_python.data_split_and_summary.summary_data_utils.compute_summary
Return summary per Split (dataset+origin) and Sex:
- NumberOfFootprints
- UniqueIndividuals
- UniqueTrails
- Mean & SD footprints per individual
- Mean & SD trails per individual
Skips empty splits.

## FIT_python.data_split_and_summary.summary_data_utils.discover_splits

## FIT_python.data_split_and_summary.summary_data_utils.load_split_data

## FIT_python.data_split_and_summary.summary_data_utils.plot_summary_table
1) "Summary All" plot: one small stacked train/test bar chart per species.
2) Individual plots: female and male bars stacked (train below, test above).
Titles use italic scientific names.

## FIT_python.data_split_and_summary.summary_data_wrapper.SummaryWrapper

## FIT_python.data_split_and_summary.summary_data_wrapper.run_summary
Create CSV and plots summarising each split separately.

## FIT_python.pipeline_individual_id.dimensionality_reduction_wrapper.DimensionalityReducerTransformer
Apply various dimensionality reduction techniques.

Supported methods
-----------------
- PCA (unsupervised)
- UMAP (unsupervised or supervised)
- t-SNE (unsupervised)
- LDA (supervised)
- MDS (unsupervised)
- Isomap (unsupervised)

## FIT_python.pipeline_individual_id.distance_metrics.compute_distances
Return several distance metrics between vectors ``a`` and ``b``.

Parameters
----------
a, b : array-like
    Input vectors.
VI : array-like, optional
    Inverse covariance matrix for Mahalanobis distance.

## FIT_python.pipeline_individual_id.feature_scaler_wrapper.FeatureScalerTransformer
Scale numerical features using either a standard or robust approach.

## FIT_python.pipeline_individual_id.feature_selection_wrapper.FeatureSelectionTransformer

## FIT_python.pipeline_individual_id.feature_selection_wrapper._forward_ranking

## FIT_python.pipeline_individual_id.generate_trails_and_trailpairs.generate_pairwise_comparisons_from_df
Generate all pairwise trail comparisons.

Steps
-----
1. ``group_id`` wird aus ``id_col`` oder ``fallback_col`` gebildet, wenn ersteres fehlt.
2. ``trails_per_animal`` stammt entweder aus vorgegebenen Pools oder wird
   über vielfältige Zufallsteilsets mit Jaccard-Distanz ausgewählt.
3. Erstelle alle Cross- und Within-Individual-Paare aus unterschiedlichen Fenstern.
4. ``same_individual`` und ``same_sex`` kennzeichnen Gleichheit oder ``"unknown"``.
5. ``StratifiedKFold`` erfolgt nach ``trail_size_a``.
6. Eine Summary-Tabelle zeigt Mittelwert und Standardabweichung der Pair-Anzahl.

## FIT_python.pipeline_individual_id.geometric_pairwise_projection.generate_pairwise_comparisons_from_df
Create trail pair comparisons with metadata:
- ``samples_a``/``samples_b``: Listen von ``id``-Werten
- ``trail_a_id``/``trail_b_id``: unique identifiers
- ``same_individual``: boolean flag
- ``fold``: stratified k-fold based on ``same_individual``

## FIT_python.pipeline_individual_id.geometric_pairwise_projection.run_all_pairwise_projections_parallel
Run projections for all pairings.

Steps
-----
0. If ``use_sexmodel_prediction`` is ``True``, load the sex model and precompute ``predict_proba`` on the entire base DataFrame.
1. Clean the base DataFrame.
2. Perform feature selection once with ``k_max``.
3. Use an RCV set comprising the remaining indices.
4. Extract ``predict_proba`` for A, B and R and compute their averages.
5. For every combination of ``reducer``, ``n_components`` and ``k``:
   - apply the dimensionality reducer
   - compute distances
   - record results including averaged probabilities

Parameters
----------
batch_size : int or None, optional
    If set, comparisons are processed in batches of this size.

## FIT_python.pipeline_individual_id.outlier_wrapper.OutlierCleanerTransformer
Clean numerical outliers using clipping or z-score limiting.

Methods
-------
- ``'clip'``: clip all features to ``[q_low, q_high]`` (percentile clipping)
- ``'zscore'``: set values outside ``±z_thresh*σ`` to the boundary (Winsorizing)

Parameters
----------
method : str
    ``'clip'`` or ``'zscore'``
lower_quantile, upper_quantile : float
    Quantiles used for clipping mode, e.g. ``0.01`` and ``0.99``
z_thresh : float
    Z-score threshold for ``'zscore'`` mode, e.g. ``3.0``

## FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline.run_all_pairwise_projections_parallel
Process all pairwise projections.

Steps
-----
0. If ``use_sexmodel_prediction`` is ``True`` load the sex model and
   precompute ``predict_proba``. ``sexmodel_path`` must point to a valid ``.joblib`` file.
1. Clean the base DataFrame.
2. Apply pipeline steps: outlier cleaning and feature scaling.
3. Perform feature selection once with ``k_max``.
4. Use an RCV set as the complement.
5. Extract and average sex probabilities.
6. For each combination of ``outlier``, ``scaler``, ``reducer``, ``n_components`` and ``k``:
   - apply the dimensionality reduction
   - compute distances
   - record results including average probabilities for A/B/R

## FIT_python.pipeline_individual_id.rcv_sampling.generate_rcv
Create the **R**ecaptured **C**ontrol **V**ariation dataset.

Used to project comparison samples into a reference space created from all other data.

Parameters
----------
full_df : pd.DataFrame
    Complete dataset containing all samples.
exclude_indices : list
    List of indices that should be excluded (e.g., used in current comparison).

Returns
-------
pd.DataFrame
    Subset of ``full_df`` excluding ``exclude_indices`` and with
    ``'individual_id'`` and ``'Trail'`` set to ``"RCV"``.

## FIT_python.pipeline_sex.dimensionality_reduction_wrapper.DimensionalityReducerTransformer
Apply different dimensionality reduction techniques.

Supported methods
-----------------
- ``None``: identity transformation
- PCA
- UMAP
- t-SNE
- LDA
- MDS
- Isomap

## FIT_python.pipeline_sex.feature_scaler_wrapper.FeatureScalerTransformer
Scale numerical features using either a standard or robust approach.

## FIT_python.pipeline_sex.feature_selection_wrapper.FeatureSelectionTransformer

## FIT_python.pipeline_sex.feature_selection_wrapper._forward_ranking

## FIT_python.pipeline_sex.grouped_metrics.individual_accuracies

## FIT_python.pipeline_sex.grouped_metrics.individual_majority_stats

## FIT_python.pipeline_sex.imputation_wrapper.ImputationWrapper

## FIT_python.pipeline_sex.outlier_wrapper.OutlierCleanerTransformer
Clean numerical outliers using clipping or z-score limiting.

Methods
-------
- ``'clip'``: clip all features to ``[q_low, q_high]`` (percentile clipping)
- ``'zscore'``: set values outside ``±z_thresh*σ`` to the boundary (Winsorizing)

Parameters
----------
method : str
    ``'clip'`` or ``'zscore'``
lower_quantile, upper_quantile : float
    Quantiles used for clipping mode, e.g. ``0.01`` and ``0.99``
z_thresh : float
    Z-score threshold for ``'zscore'`` mode, e.g. ``3.0``

## FIT_python.pipeline_sex.pipeline_wrapper_sex.PipelineWrapper
Wraps one-time preparation (import, split, summary) and
training/evaluation of all model variants.

## FIT_python.pipeline_sex.pipeline_wrapper_sex.get_pipeline_steps
Construct the list of ``(name, transformer)`` steps based on the chosen hyperparameters.

## FIT_python.pipeline_sex.transform_utils.convert_numeric
Convert feature columns to float, replacing comma decimal separators.

Non-convertible values are coerced to NaN so that mixed columns do not
raise errors during conversion.

## FIT_python.pipeline_sex.transform_utils.one_hot_encode_targets
One-hot encode target columns.
Returns:
  - y: numpy array of shape (n_samples, total_classes)
  - mapping: dict target_col -> list of classes

## FIT_python.pipeline_sex.transform_utils.save_target_mapping
Save target mapping dict to JSON.

## FIT_python.pipeline_sex.transform_wrapper.NumericTransformer
Convert feature columns to numeric ``numpy`` arrays.

Steps
-----
1. Replace commas with dots in all feature columns.
2. Coerce values to ``float`` (non-convertible values become ``NaN``).
3. Return a plain ``numpy`` array ``X``.

Target columns (``DEFAULT_TARGETS``) remain untouched and should be
extracted outside of this transformer.
