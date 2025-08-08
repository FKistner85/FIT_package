# Internal Functions and Classes

## FIT_python.Visualisations.display_mapping.apply_display_mapping
Return ``df`` with ``display_id`` and ``display_trail`` columns added.

## FIT_python.Visualisations.display_mapping.generate_display_mapping
Return mapping from original labels to anonymised display labels.

## FIT_python.Visualisations.id_style._build_markers

## FIT_python.Visualisations.id_style._build_palette

## FIT_python.Visualisations.id_style._load_unique

## FIT_python.Visualisations.id_style.sanitize_id_trail
Return ``df`` with cleaned ``id_col`` and ``trail_col`` values.

## FIT_python.Visualisations.plot_style._lighten
Return a lighter shade of ``color``.

``amount`` specifies the blend ratio with white where ``0`` returns the
original colour and ``1`` returns white.

## FIT_python.Visualisations.plot_style.apply_style
Apply consistent plot styling across notebooks and modules,
including sex-specific mappings and color palette.

## FIT_python.Visualisations.plot_style.map_sex
Map raw sex values ('f','F',0,'m','M',1) to standardized labels.
Also handles pandas Series by mapping elementwise.

## FIT_python.Visualisations.plots_utils._circle_from_row
Return center and radius for the ``prefix`` coordinates in ``row``.

## FIT_python.Visualisations.plots_utils._parse_coords
Return ``val`` as a 1D float array.

``val`` may either be a sequence of numbers or a string representation
of such a sequence. Invalid inputs yield an empty array.

## FIT_python.Visualisations.plots_utils._rhombus_from_row
Return center and L1-based radius for the ``prefix`` coordinates in ``row``.

## FIT_python.Visualisations.plots_utils._scatter_points
Helper to plot scatter points with consistent styling.

## FIT_python.Visualisations.plots_utils.make_marker_map
Return marker mapping from :data:`ID_MARKERS`.

## FIT_python.Visualisations.plots_utils.plot_dendrogram
Save a dendrogram based on ``dist_matrix``.

Samples originating from the same individual are coloured consistently using
:data:`~FIT_python.Visualisations.id_style.ID_COLORS`. Trail labels are
mapped via :data:`~FIT_python.Visualisations.id_style.TRAIL_TO_ID` to obtain
the parent ID. Tick labels fall back to black when an ID is missing from the
mapping. Labels are rotated to avoid overlaps. Optional horizontal cut-off
lines can be drawn via ``cutoff_low`` and ``cutoff_high``.

## FIT_python.Visualisations.plots_utils.plot_embedding_by_individual
Scatter embeddings for train/test splits grouped by individual.

## FIT_python.Visualisations.plots_utils.plot_feature_correlation_matrix
Create and save a 2×2 panel of correlation heatmaps with a single, external colorbar:
 - a) full correlation (distance/angle/triangles)
 - b) distance-only features
 - c) angle-only features
 - d) triangles-only features
Returns the path to the saved image.

## FIT_python.Visualisations.plots_utils.plot_feature_correlations
Backward compatible wrapper for :func:`plot_feature_correlation_matrix`.

## FIT_python.Visualisations.plots_utils.plot_feature_distributions
Plot histograms of numeric features before and after cleaning.

Parameters
----------
raw_df : pandas.DataFrame
    Data prior to cleaning.
cleaned_df : pandas.DataFrame
    Data after cleaning.
fig_dir : pathlib.Path
    Directory to store the generated plots.
bins : int, default=30
    Number of histogram bins.

Returns
-------
list[pathlib.Path]
    Paths of the saved plot images.

## FIT_python.Visualisations.plots_utils.plot_individual_boxplots
Zeichnet 2×2 Boxplots der vier Features in top4_feats,
geordnet nach sex_mapped (erst alle 'Female', dann 'Male'),
ohne x-Ticks und ohne Legende, speichert das Bild und gibt den Pfad zurück.

## FIT_python.Visualisations.plots_utils.plot_individual_boxplots
Plot 2×2 boxplots grouped by individual and sex.

## FIT_python.Visualisations.plots_utils.plot_pair_examples
Plot example 50% confidence areas for TP/FP/FN/TN categories.

Parameters
----------
df_res : pandas.DataFrame
    Result rows containing coordinate columns.
out_dir : pathlib.Path
    Directory where the output image is written.
rhombus : bool, default=False
    If ``True``, draw rhombus confidence regions based on Manhattan
    distances instead of circular ones.

## FIT_python.Visualisations.plots_utils.plot_pred_true_counts
Scatter true vs predicted counts for different result variants.

The x-axis displays ``true_count`` values while the y-axis shows
``pred_count``.  When ``n_train`` is provided it will be appended to
the ``true_count`` label.

Parameters
----------
results : dict[str, pandas.DataFrame]
    Mapping of variant name to DataFrame containing ``pred_count`` and
    ``true_count`` columns.
out_file : pathlib.Path
    Destination path for the image.

regression : bool, optional
    If ``True``, plot a simple linear regression for each variant in
    addition to the scatter points.
n_train : int, optional
    Number of unique training individuals used for the sequential holdout
    run. When provided the value is appended to the ``true_count`` axis
    label.

Returns
-------
pathlib.Path
    The saved file path.

## FIT_python.Visualisations.plots_utils.plot_sex_boxplots
Plot and save boxplots for given features split by sex (Female/Male).
Filters out Unknown.

## FIT_python.Visualisations.plots_utils.plot_sex_feature_boxplots
Plot BCR and count difference boxplots for pipelines with/without sex.

Parameters
----------
df_dict : dict[str, pandas.DataFrame]
    Mapping ``"with_sex"`` and ``"without_sex"`` to summary tables. Each
    table must contain the columns ``species``, ``bcr``, ``pred_count`` and
    ``true_count``.
fig_dir : pathlib.Path
    Directory to store the generated plots.

Returns
-------
list[pathlib.Path]
    Paths of the generated images in the order
    ``[bcr_species, bcr_all, count_species, count_all]``.

## FIT_python.Visualisations.plots_utils.plot_umap_by_individual
Scatter UMAP coordinates coloured and marked per individual.

``marker_map`` was a positional parameter in older versions.  When a ``dict``
is passed as ``fig_dir`` this function assumes the old calling convention
``(df, marker_map, fig_dir, filename)`` and adjusts parameters
accordingly.

Parameters
----------
legend : bool, optional
    If ``True`` (default), draw a legend showing the individuals.
use_sex_colors : bool, optional
    If ``True``, color individuals according to their sex using
    :data:`~FIT_python.Visualisations.plot_style.SEX_COLORS` instead of
    :data:`~FIT_python.Visualisations.id_style.ID_COLORS`.

## FIT_python.Visualisations.plots_utils.plot_umap_centroid_outliers

## FIT_python.Visualisations.plots_utils.plot_umap_centroid_outliers
Small-multiple plots of UMAP points with centroid-based outliers.

## FIT_python.Visualisations.plots_utils.plot_umap_centroid_outliers
Small‑multiple UMAP plots with KDE background and optional centroid‑based outlier markers.

Parameters
----------
df : pd.DataFrame
    Must contain columns 'UMAP1', 'UMAP2', 'individual_id', 'sex_mapped',
    and optionally 'is_outlier_centroid'.
title : str
    Title to display above the grid.
cols : int, default=6
    Number of columns in the facet grid.

Returns
-------
fig : plt.Figure
    The matplotlib Figure object containing the small multiples.

## FIT_python.Visualisations.plots_utils.plot_umap_scatter
Plot and save a UMAP scatter colored by ``sex_mapped``.

Parameters
----------
df : pandas.DataFrame
    Unused placeholder for API compatibility.
emb : pandas.DataFrame
    Must contain the columns ``UMAP1`` and ``UMAP2`` for the coordinates
    as well as ``sex_mapped`` for coloring.
legend : bool, optional
    If ``True`` (default), draw a legend for the sex categories.

Returns
-------
pathlib.Path
    Path to the saved image file.

## FIT_python.Visualisations.plots_utils.select_top_features
Return the ``k`` highest scoring feature names using ANOVA F-test.

## FIT_python.caption_utils.save_caption
Write ``text`` to ``path`` with ``.txt`` suffix and print the caption.

## FIT_python.data_split_and_summary.data_import_utils.clean_columns
Clean column names: strip spaces, lowercase, replace non-alphanumeric with underscore.

## FIT_python.data_split_and_summary.data_import_utils.coerce_numeric_columns
Convert numeric-looking object columns with comma decimal separator.

## FIT_python.data_split_and_summary.data_import_utils.convert_numeric
Convert feature columns to float, replacing comma decimal separators.

Non-convertible values are coerced to NaN so that mixed columns do not
raise errors during conversion.

This implementation performs the conversion on all columns at once which
avoids DataFrame fragmentation warnings when many columns are processed.

## FIT_python.data_split_and_summary.data_import_utils.get_feature_cols
Return the numeric feature columns of ``df``.

Feature columns are detected either by common prefixes or, if no
such columns exist, by selecting all numeric columns starting from
column index 5.

## FIT_python.data_split_and_summary.data_import_utils.load_and_prep_df_individual
Lädt das Train-Parquet für eine Spezies, filtert nur Männchen/Weibchen
und entfernt die Fold-Spalte, falls vorhanden.

## FIT_python.data_split_and_summary.data_import_utils.load_csv
Load a CSV file into a DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.load_excel
Load an Excel file into a DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.load_raw_files
Load all CSV and Excel files from 'folder', clean columns, optionally add 'id' column.
Returns a dict mapping file stem to DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.one_hot_encode_targets
One-hot encode target columns.
Returns:
  - y: numpy array of shape (n_samples, total_classes)
  - mapping: dict target_col -> list of classes

## FIT_python.data_split_and_summary.data_import_utils.sanitize_labels
Clean string labels in target_cols:
  - fillna('unknown') except skip_fillna
  - strip & lower
  - replace non-word with underscore
  - apply mapping dict
Returns a new DataFrame.

## FIT_python.data_split_and_summary.data_import_utils.save_target_mapping
Save target mapping dict to JSON.

## FIT_python.data_split_and_summary.data_import_wrapper.DataImportWrapper
High-level wrapper to clean all raw datasets and persist Parquet files.

## FIT_python.data_split_and_summary.data_import_wrapper.DataImporter
Class to import, clean, and convert data from raw files.

## FIT_python.data_split_and_summary.dataset_summary._species_label
Return a human readable species label.

## FIT_python.data_split_and_summary.dataset_summary.generate_dataset_overview
Create a Parquet summary of all raw datasets and return it as a DataFrame.

Parameters
----------
raw_dir:
    Directory containing the raw footprint tables.
out_path:
    Destination of the generated overview table.
reuse:
    When ``True`` and ``out_path`` already exists, the Parquet file is loaded and
    returned instead of recomputing the statistics.

## FIT_python.data_split_and_summary.datasplit_and_summary_wraper.SplitWrapper
High-level interface to create train/test/inference splits and assign folds.

## FIT_python.data_split_and_summary.datasplit_and_summary_wraper.prepare_all_splits
Lädt alle CSVs in RAW_DIR und erstellt für jede Art Splits:

- Wenn `species_filter` angegeben ist, nur für diese Arten.
- Für 'Eurasian Otter.csv' wird `create_train_test_split_otter` verwendet.
- Für alle anderen Arten `stratified_individual_split`.

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

## FIT_python.data_split_and_summary.split_utils.prepare_all_splits
Lädt alle CSVs in RAW_DIR und erstellt für jede Art Splits:

- Wenn `species_filter` angegeben ist, nur für diese Arten.
- Für 'Eurasian Otter.csv' wird `create_train_test_split_otter` verwendet.
- Für alle anderen Arten `stratified_individual_split`.

## FIT_python.data_split_and_summary.split_utils.sample_individuals
Randomly select `n` unique individuals for a dataset/sex combination.
Used for the fixed Otter test split.

## FIT_python.data_split_and_summary.split_utils.splits_available
Return ``True`` if at least one valid train/test pair exists.

## FIT_python.data_split_and_summary.split_utils.stratified_individual_split
Split ``df`` by ``group_col`` while stratifying by ``stratify_col``.
- All rows for one individual are kept together in train or test
- Invalid or missing groups are placed into ``inference_df``
- Optionally assign a ``Fold`` column to ``train_df`` using
  stratified group k-fold.

## FIT_python.data_split_and_summary.split_utils.train_test_group_split
Simple group-based train/test split.

## FIT_python.data_split_and_summary.summary_data_utils.compute_summary
Return summary per Split (dataset+origin) and Sex.

## FIT_python.data_split_and_summary.summary_data_utils.discover_splits

## FIT_python.data_split_and_summary.summary_data_utils.load_split_data

## FIT_python.data_split_and_summary.summary_data_utils.plot_split_proportions
Plot fraction of footprints per split for each species.

## FIT_python.data_split_and_summary.summary_data_utils.plot_summary_table
1) "Summary All" plot: one small stacked train/test bar chart per species.
2) Individual plots: female and male bars stacked (train below, test above).
Titles use italic scientific names.

## FIT_python.data_split_and_summary.summary_data_wrapper.SummaryWrapper

## FIT_python.data_split_and_summary.summary_data_wrapper.run_summary
Create per-species Parquet summaries and plots. Output paths are resolved via
`get_species_paths(section="dataprocessing", species)`.

## FIT_python.data_split_and_summary.transform_utils.convert_numeric
Convert feature columns to float, replacing comma decimal separators.

Non-convertible values are coerced to NaN so that mixed columns do not
raise errors during conversion.

Updated to convert all columns together to avoid DataFrame fragmentation
warnings when many columns are present.

## FIT_python.data_split_and_summary.transform_utils.one_hot_encode_targets
One-hot encode target columns.
Returns:
  - y: numpy array of shape (n_samples, total_classes)
  - mapping: dict target_col -> list of classes

## FIT_python.data_split_and_summary.transform_utils.save_target_mapping
Save target mapping dict to JSON.

## FIT_python.data_split_and_summary.transform_wrapper.NumericTransformer
Convert feature columns to numeric ``numpy`` arrays.

Steps
-----
1. Replace commas with dots in all feature columns.
2. Coerce values to ``float`` (non-convertible values become ``NaN``).
3. Return a plain ``numpy`` array ``X``.

Target columns (``DEFAULT_TARGETS``) remain untouched and should be
extracted outside of this transformer.

## FIT_python.general_pipeline_steps.dimensionality_reduction_wrapper.DimensionalityReducerTransformer
Apply various dimensionality reduction techniques.

Supported methods
-----------------
- PCA (unsupervised)
- UMAP (supervised)
- t-SNE (unsupervised)
- LDA (supervised)
- MDS (unsupervised)
- Isomap (unsupervised)

## FIT_python.general_pipeline_steps.feature_scaler_wrapper.FeatureScalerTransformer
Scale numerical features using either a standard or robust approach.

## FIT_python.general_pipeline_steps.feature_selection_wrapper.FeatureSelectionTransformer
Flexible feature selection supporting several strategies.

## FIT_python.general_pipeline_steps.feature_selection_wrapper._forward_ranking

## FIT_python.general_pipeline_steps.imputation_wrapper.ImputationWrapper
Impute missing numeric values using ``IterativeImputer``.

The wrapper configures :class:`~sklearn.impute.IterativeImputer` with a
:class:`~sklearn.ensemble.RandomForestRegressor` estimator. Default values
for ``n_estimators``, ``max_iter`` and ``random_state`` are taken from the
:data:`CONFIG` dictionary.

## FIT_python.general_pipeline_steps.outlier_wrapper.CentroidOutlierTransformer
Mark or remove outliers per individual in UMAP space.
Expects a DataFrame with columns ['UMAP1','UMAP2','individual_id'].

## FIT_python.general_pipeline_steps.outlier_wrapper.OutlierCleanerTransformer
Clean numerical outliers using clipping or z-score limiting.

Methods
-------
- ``'clip'``: clip all features to ``[q_low, q_high]`` (percentile clipping)
- ``'zscore'``: set values outside ``±z_thresh*σ`` to the boundary (Winsorizing)

## FIT_python.general_pipeline_steps.outlier_wrapper.UMAPtoDF
Convert arrays to a DataFrame with UMAP coordinates and IDs.

## FIT_python.general_pipeline_steps.outlier_wrapper.mark_outliers_centroid
Berechnet für jede individual_id separat:
  1. Den Centroid der UMAP-Punkte.
  2. Eine Gauß‑KDE um diesen Centroid (Varianz = bandwidth^2).
  3. Die Dichte jedes echten Punktes unter diesem Kernel.
  4. Markiert als Outlier alle Punkte unterhalb des gegebenen Perzentils.

## FIT_python.general_pipeline_steps.outlier_wrapper.print_filter_stats
Print summary statistics about removed records.

## FIT_python.general_pipeline_steps.outliers.mark_outliers_centroid
Berechnet für jede individual_id separat:
  1. Den Centroid der UMAP-Punkte.
  2. Eine Gauß‑KDE um diesen Centroid (Varianz = bandwidth^2).
  3. Die Dichte jedes echten Punktes unter diesem Kernel.
  4. Markiert als Outlier alle Punkte unterhalb des gegebenen Perzentils.

## FIT_python.gui_annotator.annotation_canvas.AnnotationCanvas
Interactive canvas allowing placement of landmarks and scale refs.

## FIT_python.gui_annotator.annotation_canvas.Landmark
Single annotated landmark with visibility flag.

## FIT_python.gui_annotator.gui_app.AnnotatorApp

## FIT_python.gui_annotator.gui_app._angle
Return angle ABC in degrees.

## FIT_python.gui_annotator.gui_app._compute_derived

## FIT_python.gui_annotator.gui_app._compute_measurements

## FIT_python.gui_annotator.gui_app._distance

## FIT_python.gui_annotator.gui_app._midpoint

## FIT_python.gui_annotator.gui_app._triangle

## FIT_python.gui_annotator.gui_app.run

## FIT_python.gui_annotator.image_manager.extract_metadata
Extract basic metadata from ``path``.

The expected directory layout is ``RAW_DIR/species/sex/individual_id/trail/image.jpg``.
If the hierarchy does not match this pattern, missing fields are set to ``None``.
EXIF information is parsed using :func:`PIL.Image.open` to obtain GPS
coordinates, timestamp, camera model and original image size.

## FIT_python.gui_annotator.image_manager.load_and_preprocess
Load image by ID from ``RAW_DIR`` and apply preprocessing.

## FIT_python.gui_annotator.image_manager.load_image
Load an image from the given path.

## FIT_python.gui_annotator.image_manager.preprocess_image
Apply rotation, scaling and optional horizontal flip.

## FIT_python.gui_annotator.image_manager.save_processed_image
Save processed image to ``PROCESSED_DIR`` and return the path.

## FIT_python.pipeline_individual_id.__init__._load_metadata
Return the docstring and :class:`inspect.Signature` of ``func_name``.

The information is extracted statically from ``simple_baseline.py`` so no
heavy dependencies need to be imported when this module is loaded. The
wrapper signatures and docstrings therefore automatically reflect the
current implementation of the underlying helpers.

## FIT_python.pipeline_individual_id.__init__._make_lazy_wrapper
Create a lazy wrapper for ``func_name`` from ``simple_baseline``.

## FIT_python.pipeline_individual_id.__init__.run_id_search
Lazy wrapper around :func:`search.run_species_search`.

All parameters – including ``cv`` for the cross-validation strategy – are
forwarded to :func:`~FIT_python.pipeline_individual_id.search.run_species_search`
which performs a :class:`skopt.BayesSearchCV` over the pairwise ID pipeline.

## FIT_python.pipeline_individual_id.baseline_pipeline.DistanceBaseline
Utility class to compute baseline distances on validation comparisons.

## FIT_python.pipeline_individual_id.distance_metrics.compute_distances
Return several distance metrics between vectors ``a`` and ``b``.

Parameters
----------
a, b : array-like
    Input vectors.

## FIT_python.pipeline_individual_id.evaluation.average_trail_stats
Return the column-wise mean of ``summary_tables``.

The function expects each table to contain a ``sub_size == 'Total'`` row.
Only numeric columns of these rows are averaged.

## FIT_python.pipeline_individual_id.evaluation.compute_bcr
Return the Balanced Classification Rate (BCR) for ``cm``.

BCR as described in the FIT documentation is the mean of the true
positive rate and the true negative rate calculated from the confusion
matrix produced by :func:`compute_confusion`.

## FIT_python.pipeline_individual_id.evaluation.compute_confusion
Return a confusion matrix for ``results_df``.

Parameters
----------
results_df:
    DataFrame containing ground-truth and prediction columns.
true_col:
    Column name with the true ``same_individual`` labels.
pred_col:
    Column name with the predicted labels.

The function maps string labels ``"True"``/``"False"`` to ``1``/``0`` and
returns a ``pandas.DataFrame`` with labelled rows/columns.

## FIT_python.pipeline_individual_id.evaluation.compute_overlap_jsl_style
Return True if the 50% confidence ellipses of A/B overlap.

## FIT_python.pipeline_individual_id.evaluation.compute_overlap_jsl_style_isotropic
Return True if the p‑level isotropic confidence circles of A/B overlap.

Die Funktion berechnet aus den 2D‑Koordinaten von A und B jeweils
den radialen Standardabstand vom Mittelpunkt und multipliziert mit
sqrt(chi2.ppf(p, df=2)), um einen Kreisradius zu erhalten.

## FIT_python.pipeline_individual_id.evaluation.compute_overlap_jsl_style_vec
Return boolean overlap predictions for all rows in ``df``.

The function performs the same ellipse overlap check as
:func:`compute_overlap_jsl_style`, but processes the entire dataframe in a
vectorised manner. The coordinate columns may contain either sequences of
numeric values or their string representations.

## FIT_python.pipeline_individual_id.evaluation.compute_overlap_rhombus
Return ``True`` if two rhombus confidence areas overlap.

Parameters
----------
row : pandas.Series
    Must contain ``coords_a_x``, ``coords_a_y``, ``coords_b_x`` and
    ``coords_b_y`` columns with list-like coordinate values (or string
    representations).
p : float, default=0.5
    Probability mass used for the chi-square radius calculation.

The function computes Manhattan (L1) distances of each point from its
centroid, estimates a radius using ``chi2.ppf(p, df=2)`` and returns
``True`` if the distance between centroids does not exceed the sum of the
two radii.

## FIT_python.pipeline_individual_id.evaluation.parse_list
Return ``value`` as a NumPy ``float`` array if possible.

Parameters
----------
value:
    Either a string representation of a list or an actual sequence
    of numeric values.

This helper previously only accepted strings and returned an empty
array when called with list objects.  During the sequential holdout
evaluation the coordinate lists are still Python ``list`` instances,
which resulted in failed parsing and consequently ``False``
predictions.  By accepting lists and arrays directly the overlap
logic works correctly regardless of whether results are read from a
CSV file or computed on the fly.

## FIT_python.pipeline_individual_id.evaluation.report_skipped
Return a multi-line summary of skipped validation counts.

## FIT_python.pipeline_individual_id.evaluation.separation_score
Return a scoring callable comparing mean diff/same distances.

## FIT_python.pipeline_individual_id.evaluation.sequential_holdout_ids
Generate sequential train/validation splits based on unique IDs.

For each iteration a new validation set is drawn for every entry in
``val_sizes``.
Each split is therefore independent and does not build on the previous one
within the same iteration.

## FIT_python.pipeline_individual_id.generate_trails_and_trailpairs._mode
Return the first mode of ``values`` or ``pd.NA`` when empty.

## FIT_python.pipeline_individual_id.generate_trails_and_trailpairs.generate_pairs
Create all trail pair combinations excluding same base trail.

## FIT_python.pipeline_individual_id.generate_trails_and_trailpairs.generate_pairwise_comparisons_from_df
Create pairwise trail comparisons.

The function uses :func:`select_or_generate_trails` and
:func:`generate_subsamples` to obtain trail definitions. ``fold_a`` and
``fold_b`` are read directly from ``fold_col`` in ``df`` with
``same_fold`` indicating equality. If a ``substrate`` column exists or the
species is ``"eurasian_otter"`` then ``substrate_a`` and ``substrate_b`` are
added to each comparison.

When ``evaluation`` is ``True`` the ``fold`` column is ignored and the
returned comparisons contain ``None`` for ``fold_a``/``fold_b`` and do not
set ``same_fold``.

## FIT_python.pipeline_individual_id.generate_trails_and_trailpairs.generate_subsamples
Create diverse subsamples for each trail.

## FIT_python.pipeline_individual_id.generate_trails_and_trailpairs.select_or_generate_trails
Return existing trails or generate new ones.

Parameters
----------
df : pd.DataFrame
    Input data with one row per footprint.
strategy : str, optional
    ``"select"`` to keep existing ``trail`` values or ``"generate``" to
    create new trails. Defaults to ``"generate"``.
individual_col : str, optional
    Column identifying individuals. Defaults to ``"individual_id"``.
trail_col : str, optional
    Column containing trail identifiers. Defaults to ``"trail"``.
sample_size : int, optional
    Number of observations per generated trail. Defaults to
    ``CONFIG['pipeline_individual_id']['trail_generation_defaults']['sample_size']``.
random_state : int, optional
    Seed for random sampling. Defaults to ``GLOBAL_RANDOM_SEED``.

Returns
-------
pd.DataFrame
    ``df`` with updated ``trail_col`` values.

## FIT_python.pipeline_individual_id.holdout_helper.generate_holdout_sets
Return sequential holdout splits for every species.

Parameters
----------
split_dir:
    Directory containing per-species ``train.parquet`` files.
val_sizes:
    Validation set sizes passed to :func:`sequential_holdout_ids`.
iterations:
    Number of sequential holdout iterations.
seed:
    Random seed for reproducible splits. Defaults to ``GLOBAL_RANDOM_SEED``.

Returns
-------
dict[str, list[dict[str, pd.DataFrame]]]
    Mapping from species codes to lists of split dictionaries. Each
    dictionary contains ``train_df`` and ``val_df`` along with
    ``iteration`` and ``n_val`` information.

## FIT_python.pipeline_individual_id.search.PairwiseEstimator
Estimator that computes pairwise distances using the embedding pipeline.

All preprocessing steps including outlier cleaning, scaling, feature
selection and dimensionality reduction are handled inside
:func:`run_all_pairwise_projections_parallel`.

## FIT_python.pipeline_individual_id.search.run_species_search
Run BayesSearchCV for all species in ``SPLITS_DIR``.

Parameters
----------
species_filter : list[str], optional
    If given, restrict the search to these species directory names.
n_iter : int, optional
    Number of parameter samples drawn by :class:`skopt.BayesSearchCV`.
cv : int or str, optional
    Cross-validation strategy. ``"fold"`` uses the ``Fold`` column via
    :class:`~sklearn.model_selection.PredefinedSplit`. Any other value is
    forwarded to :class:`skopt.BayesSearchCV` as-is.
random_state : int, optional
    Random seed controlling the search. Defaults to ``GLOBAL_RANDOM_SEED``.

Notes
-----
Preprocessing (outlier cleaning, scaling, feature selection and dimensional
reduction) is performed inside :class:`PairwiseEstimator` by
:func:`run_all_pairwise_projections_parallel`.

## FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline.run_all_pairwise_projections_parallel
Process all pairwise projections.

Steps
-----
0. If ``use_sexmodel_prediction`` is ``True`` load the sex model and
   precompute ``predict_proba``. When no ``sexmodel_path`` is given, the
   classifier location is derived from the ``species`` column via
   ``PATHS['random_search']/balanced_test_acc/<species>.joblib``.
1. Clean the base DataFrame.
2. Apply pipeline steps: outlier cleaning and feature scaling.
3. Perform feature selection once with ``k_max`` on the **scaled** data.
4. Use an RCV set as the complement.
5. Extract and average sex probabilities.
6. For each combination of ``outlier``, ``scaler``, ``reducer``, ``n_components`` and ``k``:
   - apply the dimensionality reduction
   - compute distances
   - record results including average probabilities for A/B/R

``fit_df`` optionally specifies the DataFrame used to fit the preprocessing
steps.  When ``None`` the full ``df`` is used for fitting as before.

Parameters ``checkpoint_path`` and ``resume`` allow long runs to be continued.
When ``checkpoint_path`` is given the current index and partial results are
written to that file after each pair. Setting ``resume=True`` loads the
checkpoint if it exists and processing continues from the saved index.

## FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline.run_embedding_once_pipeline
Convenience wrapper that processes comparisons sequentially.

Parameters ``checkpoint_path`` and ``resume`` mirror the arguments of
:func:`run_all_pairwise_projections_parallel`.

## FIT_python.pipeline_individual_id.population_estimation.cluster_population
Return the number of clusters given a distance threshold.

Parameters
----------
dist_matrix : pd.DataFrame
    Square pairwise distance matrix.
cutoff : float
    Distance threshold passed to :func:`scipy.cluster.hierarchy.fcluster`.

Returns
-------
int
    Detected cluster count.

## FIT_python.pipeline_individual_id.population_estimation.compute_erd
Compute the expected relative difference (ERD).

The ERD measures how far ``predicted`` deviates from ``true`` relative
to the ground truth. It is defined as ``abs(predicted - true) / true``.

Parameters
----------
predicted : int
    Estimated population size.
true : int
    Known population size.

Returns
-------
float
    Relative deviation between prediction and truth.

## FIT_python.pipeline_individual_id.population_estimation.concordance_correlation_coefficient
Return Lin's concordance correlation coefficient between ``x`` and ``y``.

## FIT_python.pipeline_individual_id.population_estimation.optimal_cutoff
Return the Ward distance most likely to yield ``true_n`` clusters.

The confidence interval describes the range of distances producing the
same cluster count. Its bounds correspond to the 25\% and 75\%
percentiles of that interval.

Parameters
----------
distances : pd.DataFrame
    Square pairwise distance matrix.
true_n : int
    Expected number of clusters in the data.

Returns
-------
tuple
    ``(cutoff, (low, high))`` where ``cutoff`` is the midpoint of the
    valid distance range and ``(low, high)`` is the 25\% confidence
    interval around it.

## FIT_python.pipeline_individual_id.population_estimation.silhouette_cluster_count
Return the Ward cluster count with the highest silhouette score.

``k`` values from ``2`` to ``len(dist_matrix) - 1`` are evaluated and the
number of clusters yielding the best silhouette score is returned.

Parameters
----------
dist_matrix : pd.DataFrame
    Square pairwise distance matrix.

Returns
-------
int
    Cluster count maximising the silhouette score.

## FIT_python.pipeline_individual_id.rcv_sampling.generate_rcv
Create the **R**ecaptured **C**ontrol **V**ariation dataset.

Used to project comparison samples into a reference space created from all other data.

Parameters
----------
full_df : pd.DataFrame
    Complete dataset containing all samples.
exclude_ids : list
    List of ``id`` values that should be excluded (e.g. used in the
    current comparison).

Returns
-------
pd.DataFrame
    Subset of ``full_df`` excluding ``exclude_ids`` and with
    ``'individual_id'`` and ``'Trail'`` set to ``"RCV"``.

## FIT_python.pipeline_individual_id.sequential_holdout._merge_predictions
Return ``df`` merged with ``preds`` and list of added columns.

## FIT_python.pipeline_individual_id.sequential_holdout._pairs_to_array
Return distance array and trail order via index mapping.

Benchmarking ``timeit`` on 5k pairs and 1k unique trails showed
roughly a 12x speed-up compared to the previous row-wise loop.

## FIT_python.pipeline_individual_id.sequential_holdout.compute_global_cutoffs
Return aggregate Ward cut-off statistics across splits.

Parameters
----------
all_splits_path:
    CSV file created by :func:`run` containing the pairwise
    results of all splits.

Returns
-------
dict
    Dictionary with the mean and median Ward cut-off as well as
    the mean lower and upper 25% bounds across splits.

## FIT_python.pipeline_individual_id.sequential_holdout.evaluate_with_cutoff
Re-evaluate splits with a fixed Ward cut-off.

Parameters
----------
result_dir:
    Directory containing ``all_splits.csv`` produced by :func:`run`.
cutoff:
    Ward distance passed to :func:`cluster_population`.

Returns
-------
pd.DataFrame
    Data frame with one row per split containing ``pred_count``,
    ``true_count`` and ``erd``.

## FIT_python.pipeline_individual_id.sequential_holdout.run
Evaluate pairwise pipeline using sequential holdouts.

Parameters
----------
df:
    Cleaned footprints with ``individual_id`` and feature columns.
feature_cols:
    Names of morphometric feature columns.
sex_predictions:
    Optional DataFrame with sex-model predictions to append as extra
    features. When provided, it must be indexed by ``sample_col`` or contain
    a column of that name.
sample_col:
    Column containing sample identifiers. Defaults to ``"id"``.
id_col:
    Column identifying individuals. Defaults to ``"individual_id"``.
iterations:
    Number of sequential holdout iterations.
val_sizes:
    Validation sizes passed to :func:`sequential_holdout_ids`. Defaults to
    ``CONFIG['pipeline_individual_id']['sequential_holdout_val_sizes']``.
random_state:
    Random seed for the split generator. Defaults to ``GLOBAL_RANDOM_SEED``.
out_dir:
    Directory to write per-split CSV results. Defaults to
    ``RESULTS_DATA_DIR / 'individual_id'``.
tag:
    Optional identifier stored with raw split results when ``master_fp`` is
    provided.
master_fp:
    Parquet file collecting raw split results. When given, each split's
    comparisons are appended to this file.
n_jobs:
    Parallel jobs for the pairwise projection step.  ``-1`` uses all cores.
reuse_summary:
    When ``True`` and ``out_dir/summary.csv`` exists, load that table instead
    of recomputing the sequential holdout.
k_features:
    Number of morphometric features to select. ``None`` uses the default
    from :func:`run_all_pairwise_projections_parallel`.
trail_col:
    Column name containing trail identifiers.
subsample:
    Whether to generate sub-trails when creating comparisons.
cutoff:
    Optional Ward distance used for population clustering. When ``None`` the
    cutoff is determined via :func:`optimal_cutoff` for each split.
overlap_prob:
    Probability mass for :func:`compute_overlap_jsl_style_vec` when predicting
    whether two trails belong to the same individual.
sexmodel_path : str, optional
    Path to a saved sex classifier.  If ``use_sexmodel_prediction`` is
    ``True`` and no path is provided, the model location is derived from the
    ``species`` column using
    ``PATHS['random_search']/balanced_test_acc/<species>.joblib``.

Returns
-------
pd.DataFrame
    Summary table with one row per split containing the BCR, ERD, predicted
    and true population sizes as well as the Ward cut-off statistics
    (``ward_cutoff``, ``cutoff_low``, ``cutoff_high``).  In addition to the
    per-split ``split_*.csv`` files, a combined ``all_splits.csv`` containing
    all pairwise results is written to ``out_dir``.

## FIT_python.pipeline_individual_id.siamese_pipeline.SiameseNet

## FIT_python.pipeline_individual_id.siamese_pipeline.TripletDataset
Randomly generate triplets from a dataframe.

Parameters
----------
seed : int, optional
    Seed for the internal random number generator. Defaults to
    ``GLOBAL_RANDOM_SEED``.

## FIT_python.pipeline_individual_id.siamese_pipeline._pairwise_dataset
Return pairwise distance features and labels.

## FIT_python.pipeline_individual_id.siamese_pipeline.build_dataloader

## FIT_python.pipeline_individual_id.siamese_pipeline.run
Train a siamese network and evaluate on validation comparisons.

Parameters
----------
seed : int, optional
    Seed for random sampling in the training process. Defaults to
    ``GLOBAL_RANDOM_SEED``.

## FIT_python.pipeline_individual_id.siamese_pipeline.train_siamese

## FIT_python.pipeline_individual_id.simple_baseline._load_splits
Return concatenated train and optionally test tables for ``species_dir``.

Parameters
----------
species_dir:
    Path pointing to the directory containing ``train.parquet`` and
    optionally ``test.parquet`` files.
include_test:
    When ``True`` (default) the ``test.parquet`` file is loaded as well.
    Set to ``False`` to work exclusively with the training split.

## FIT_python.pipeline_individual_id.simple_baseline._pairs_to_array
Return distance array and trail order via index mapping.

Using the vectorised approach is about an order of magnitude faster
than iterating over ``DataFrame`` rows.

## FIT_python.pipeline_individual_id.simple_baseline.add_sex_features
Append sex probabilities to ``df`` and compute trail-level aggregates.

## FIT_python.pipeline_individual_id.simple_baseline.collect_id_metrics
Return averaged metrics across baseline holdout splits.

Parameters
----------
exp_dir:
    Root directory containing one subdirectory per species with a
    ``summary.csv`` produced by the baseline helper.

Returns
-------
pandas.DataFrame
    Table with one row per species containing the mean BCR, mean ERD
    and the concordance correlation coefficient (CCC).

## FIT_python.pipeline_individual_id.simple_baseline.load_sex_predictions
Return sex-model predictions for ``species``.

This is a thin wrapper around :func:`predict_all` from the sex classification
pipeline.  The helper simply forwards the parameters and returns the
resulting DataFrame so that the individual ID baseline can load the
predictions without importing the full sex pipeline here.

## FIT_python.pipeline_individual_id.simple_baseline.plot_bcr_comparison
Plot a bar chart comparing BCR across species.

## FIT_python.pipeline_individual_id.simple_baseline.run_baseline_all_species
Evaluate the baseline for every species using sequential holdouts.

For each species in :data:`SPLITS_DIR` this helper invokes
:func:`sequential_holdout.run` once and writes ``summary.csv`` to
``exp_dir/<species>``.

Parameters
----------
exp_dir : Path
    Root directory for all results. One subdirectory per species will be
    created underneath this path.
best_k : int
    Feature count used for the otter dataset and as the default for all
    other species.
cutoff : dict[str, Any]
    Mapping of species name to a parameter dictionary with optional keys
    ``"k"`` (int) and ``"ward"`` (float) overriding ``best_k`` and the Ward
    clustering cut-off.
reuse_summary : bool, optional
    When ``True`` and ``exp_dir/<species>/summary.csv`` exists, the
    computation for that species is skipped.
n_jobs : int, optional
    Parallel jobs forwarded to :func:`sequential_holdout.run`.  ``-1`` uses
    all available CPU cores.

## FIT_python.pipeline_individual_id.simple_baseline.run_fold_cv
Evaluate pairwise pipeline using predefined folds.

The function iterates over unique values in ``fold_col`` and treats each
fold as validation set while the remaining data forms the training set.  The
results for every fold are written to ``out_dir`` as ``fold_<n>.csv`` with a
combined ``summary.csv`` containing the evaluation metrics.  Summary rows
now also include a ``pipeline`` identifier describing the preprocessing and
reduction steps chosen inside
:func:`run_all_pairwise_projections_parallel`.

Parameters
----------
selection_method, reducers, n_components, scaler_methods
    Parameters forwarded to ``run_all_pairwise_projections_parallel`` to
    control feature selection, dimensionality reduction and scaling.
sexmodel_path : str, optional
    Path to a saved sex classifier.  When ``use_sexmodel_prediction`` is
    ``True`` and no path is given, the function attempts to resolve the
    model path from the ``species`` column using
    ``PATHS['random_search']/balanced_test_acc/<species>.joblib``.

## FIT_python.pipeline_individual_id.simple_baseline.run_sex_prediction_experiment
Evaluate sequential holdouts with and without sex predictions.

For each species two runs of :func:`sequential_holdout.run` are performed:
one with appended sex-model probabilities and one without.  The resulting
summaries are written to ``exp_dir/<species>/with_sex`` and
``exp_dir/<species>/without_sex`` respectively.

Parameters
----------
exp_dir : Path
    Root directory for the output structure.
best_k : int
    Default number of features for all species.
cutoff : dict[str, Any]
    Mapping of species specific Ward cut-offs and feature counts.
models_dir : str or Path, optional
    Directory containing saved sex models loaded by
    :func:`load_sex_predictions`.
reuse_results : bool, optional
    When ``True`` (default) existing ``summary.csv`` files are loaded and
    the computation for that setup is skipped.

## FIT_python.pipeline_individual_id.simple_baseline.run_simple_baseline_all_species
Evaluate cross-validation folds for every species.

This helper mirrors :func:`run_baseline_all_species` but relies on the
``fold`` column of the training data instead of sequential holdouts.
It calls :func:`run_fold_cv` for each species and writes the results to
``exp_dir/<species>``.

Parameters
----------
exp_dir : Path
    Directory where per-species results will be stored.
best_k : int
    Default number of features used for the otter dataset and as fallback
    for other species.
cutoff : dict[str, Any]
    Mapping of species names to optional ``"k"`` and ``"ward"`` overrides.
subsample : bool, optional
    When ``True`` a subset of trail pairs is sampled for each fold via
    :func:`generate_pairwise_comparisons_from_df`.
reuse_summary : bool, optional
    Skip processing when ``exp_dir/<species>/summary.csv`` already exists.
n_jobs : int, optional
    Parallel jobs forwarded to :func:`run_fold_cv`.  ``-1`` uses all cores.
use_sex_predictions : bool, optional
    When ``True`` sex-model probabilities are loaded via
    :func:`load_sex_predictions` and appended before evaluation.
use_sexmodel_prediction : bool, optional
    Forwarded to :func:`run_fold_cv` to toggle usage of sex-model
    predictions inside the pairwise pipeline.
sexmodel_path : str or Path, optional
    Path to a saved sex model forwarded to :func:`run_fold_cv` when
    ``use_sexmodel_prediction`` is ``True``.  When ``None`` the path is
    resolved automatically for each species using
    ``PATHS['random_search']/balanced_test_acc/<species>.joblib``.
models_dir : Path, optional
    Directory containing the saved sex models used by
    :func:`load_sex_predictions`.
selection_method, reducers, n_components, scaler_methods : optional
    Parameters forwarded to :func:`run_fold_cv` controlling feature
    selection, dimensionality reduction and scaling.

## FIT_python.pipeline_individual_id.simple_baseline.run_simple_baseline_otter
Run sequential holdouts for the otter dataset.

The helper evaluates different feature counts using
:func:`sequential_holdout.run` and returns the ``k`` with the best mean BCR.

Parameters
----------
exp_dir : Path
    Directory used to store intermediate and summary results.
k_range : Iterable[int], optional
    Range of feature counts to evaluate.  Defaults to ``range(12, 21)``.
iterations : int, optional
    Number of sequential holdout iterations per ``k``.  Defaults to ``10``.
use_sex_predictions : bool, optional
    Append sex-model probabilities via :func:`load_sex_predictions` when
    ``True``.
reuse_results : bool, optional
    When ``True`` already existing summaries are loaded instead of
    recomputing the baseline.

Returns
-------
int
    The ``k`` value that achieved the best balanced correct recognition
    rate.

## FIT_python.pipeline_individual_id.supervised_umap_pipeline._compute_pair_features
Compute distance features for each comparison.

The ``embeddings`` DataFrame is indexed by ``id`` so ``samples_a`` and
``samples_b`` can be looked up directly via ``loc``.

## FIT_python.pipeline_individual_id.supervised_umap_pipeline._prepare_features
Return processed feature matrix and the fitted preprocessing pipeline.

## FIT_python.pipeline_individual_id.supervised_umap_pipeline.run
Fit a supervised UMAP pipeline and classify validation pairs.

Parameters
----------
train_df : pd.DataFrame
    Training footprints with ``individual_id`` and feature columns.
val_comparisons : sequence of dict
    Pair specifications containing ``samples_a``/``samples_b`` lists of
    ``id`` values and ``same_individual`` label.
feature_cols : sequence of str
    Columns used as numeric input features.
sex_features : sequence of str, optional
    Extra columns appended before dimensionality reduction.
val_df : pd.DataFrame, optional
    Validation samples referenced by ``val_comparisons``. When provided,
    embeddings are also computed for these samples so that their IDs are
    available during pair feature computation.

Returns
-------
tuple of (predictions_df, embeddings_df)

## FIT_python.pipeline_individual_id.train_individual_id_pipelines._add_sex_predictions
Append sex model predictions as additional features.

## FIT_python.pipeline_individual_id.train_individual_id_pipelines._confusion_from_results

## FIT_python.pipeline_individual_id.train_individual_id_pipelines._ensure_splits
Return directory containing train/test/inference splits.

## FIT_python.pipeline_individual_id.train_individual_id_pipelines._load_splits
Load train and test dataframes and print a short summary.

## FIT_python.pipeline_individual_id.train_individual_id_pipelines._select_features
Return the names of all morphometric feature columns.

## FIT_python.pipeline_individual_id.train_individual_id_pipelines.main

## FIT_python.pipeline_individual_id.train_individual_id_pipelines.scale_holdout_split
Scale ``morph_cols`` in ``train_df`` and ``val_df`` using ``StandardScaler``.

The scaler is fitted on the training dataframe and then applied to both
the training and validation dataframes. Copies of the inputs with the
transformed columns are returned.

## FIT_python.pipeline_individual_id.training_script.train_pipelines
Run the embedding pipeline and summarise results.

## FIT_python.pipeline_individual_id.utils.map_indices
Map positional indices to ID strings if necessary.

Parameters
----------
id_list:
    List of ID strings or positional indices.
idx_to_id:
    Series mapping positional indices to ID strings.
valid_index:
    Iterable of valid ID strings (e.g., ``DataFrame.index``).

Returns
-------
List[str]
    The mapped ID strings.

## FIT_python.pipeline_individual_id.utils.prepare_base_df
Return cleaned DataFrame and index mapping.

Parameters
----------
df:
    Input data frame with sample rows.
feature_cols:
    Names of numeric feature columns.
sample_col:
    Column containing sample identifiers.

Returns
-------
Tuple[pd.DataFrame, pd.Series]
    ``df`` indexed by ``sample_col`` and a positional index to id mapping.

## FIT_python.pipeline_sex.__init__.run_pipeline
Prepare data and train the sex-classification models.

This is a convenience wrapper around :class:`PipelineWrapper`.  All
keyword arguments are forwarded to :class:`~pipeline_sex.pipeline_wrapper_sex.PipelineWrapper`.

Parameters
----------
model_keys : list[str], optional
    Identifiers of models to train.  Available keys are defined in
    ``pipeline_wrapper_sex.MODELS``.
fs_method : str or None, optional
    Feature-selection algorithm. Options: ``forward``, ``random_forest``,
    ``variance``, ``univariate``, ``lasso`` or ``None``.
fs_k : int, optional
    Number of features selected when ``fs_method`` is not ``None``.
impute_method : str or None, optional
    Imputation strategy. Options: ``miss_forest`` or ``None``.
outlier_method : str or None, optional
    Outlier cleaning method. Options: ``clip``, ``zscore`` or ``None``.
scaler_method : str or None, optional
    Feature scaling approach. Options: ``standard``, ``robust`` or ``None``.
reduce_pre_method, reduce_post_method : str or None, optional
    Dimensionality reduction before/after feature selection. Options:
    ``pca``, ``umap``, ``tsne`` or ``None``.
n_jobs : int, optional
    Number of parallel jobs used for cross-validation. ``-1`` uses all
    available CPU cores.
debug : bool, optional
    If ``True`` additional debug information is printed during training.

Returns
-------
pandas.DataFrame
    The result table produced by :meth:`pipeline_sex.pipeline_wrapper_sex.PipelineWrapper.train`.

## FIT_python.pipeline_sex.baseline_sex.collect_best_metrics
Return metrics for the best accuracy model of each species.

Parameters
----------
exp_dir:
    Directory searched recursively for ``best_models.csv`` files.

Returns
-------
pandas.DataFrame
    DataFrame with one row per species containing the metrics of the
    best model w.r.t ``accuracy_test``.

## FIT_python.pipeline_sex.baseline_sex.plot_accuracy_by_sex
Plot female vs. male individual accuracy for each species.

## FIT_python.pipeline_sex.baseline_sex.plot_accuracy_comparison
Plot a bar chart comparing accuracy across species.

The function accepts data frames using either the ``*_test`` suffix as
produced by :func:`collect_best_metrics` or the shorter ``accuracy`` / ``f1``
naming used by :func:`run_simple_baseline_all_species`.

## FIT_python.pipeline_sex.baseline_sex.plot_majority_comparison
Plot the fraction of majority-correct individuals per species.

Accepts both ``maj_test_pct`` and the abbreviated ``maj_pct`` column names
as produced by :func:`run_simple_baseline_all_species`.

## FIT_python.pipeline_sex.baseline_sex.run_baseline_all_species
Train baseline LDA classifiers for each species.

A :class:`PipelineWrapper` with ``model_keys=['lda']`` and
``fs_method='forward'`` is used to train one model per species.  The
resulting ``raw_results.csv`` and the best models are written below
``exp_dir``.

Parameters
----------
exp_dir : Path
    Directory where ``raw_results.csv`` and best models will be stored.

## FIT_python.pipeline_sex.grouped_metrics.individual_accuracies
Return female, male and balanced accuracy per individual.

## FIT_python.pipeline_sex.grouped_metrics.individual_majority_stats
Count individuals with majority correct predictions.

## FIT_python.pipeline_sex.pipeline_wrapper_sex.PipelineWrapper
Wraps one-time preparation (import, split, summary) and
training/evaluation of all model variants.

## FIT_python.pipeline_sex.pipeline_wrapper_sex.get_pipeline_steps
Construct the list of ``(name, transformer)`` steps based on the chosen hyperparameters.

## FIT_python.pipeline_sex.pipeline_wrapper_sex.plot_pipeline_timings
Create a bar chart of average seconds per preprocessing step.

## FIT_python.pipeline_sex.search.EstimatorWrapper
Simple container for an estimator without ``__len__``/``__iter__``.

The wrapper proxies all estimator methods/attributes so it can be used
transparently inside a :class:`~sklearn.pipeline.Pipeline` while ensuring
that optimization libraries treat it as an atomic object.

## FIT_python.pipeline_sex.search._run_species_search
Run the hyperparameter search for a single species.

Parameters
----------
reuse_results:
    When ``True`` and result CSVs exist in ``base_dir_suffix`` the search
    is skipped and the files are loaded instead.

## FIT_python.pipeline_sex.search.prepare_eurasian_otter
Prepare splits only for the Eurasian otter dataset.

## FIT_python.pipeline_sex.search.run_other_species_search
Run the search for all species except the Eurasian otter.

Parameters
----------
n_iter : int, optional
    Number of iterations for :class:`skopt.BayesSearchCV`.
cv : int or str, optional
    Cross-validation strategy forwarded to :func:`run_species_search`.
random_state : int, optional
    Random seed controlling the search.

## FIT_python.pipeline_sex.search.run_otter_search_sex
Run BayesSearchCV for the Eurasian otter dataset.

Parameters
----------
n_iter : int, optional
    Number of iterations for :class:`skopt.BayesSearchCV`.
cv : int or str, optional
    Cross-validation strategy forwarded to :func:`run_species_search`.
random_state : int, optional
    Random seed controlling the search.
reuse_results : bool, optional
    When ``True`` previously saved search results are loaded from
    ``PATHS['random_search']``.

## FIT_python.pipeline_sex.search.run_species_search
Run the search for all species in ``SPLITS_DIR``.

Parameters
----------
species_filter : list[str], optional
    Restrict the search to these species names.
n_iter : int, optional
    Number of iterations for :class:`skopt.BayesSearchCV`.
cv : int or str, optional
    Cross-validation strategy (``"fold"`` or integer).
random_state : int, optional
    Random seed controlling the search.
reuse_results : bool, optional
    When ``True`` previously saved results are reused if present.

Returns
-------
tuple[pandas.DataFrame, pandas.DataFrame]
    Combined results of all searches and the best parameters per species.

## FIT_python.pipeline_sex.sex_predict_and_visualisation._base_paths
Return split dir, model dir and output CSV for ``species``.

If ``models_dir`` is provided, it is used directly.  Otherwise the function
falls back to the species-specific directory under
``results/data``. When ``prefer_generic`` is ``True`` or the species
directory does not exist, ``random_search_standard_metrics`` is used
instead.  This enables notebooks that work on a single species to load
custom models while the all-species notebook can still rely on the generic
directory.

## FIT_python.pipeline_sex.sex_predict_and_visualisation._plot_quality_heatmaps_single
Plot prediction quality heatmaps for a single model and split.

When ``group_by`` is provided, predictions are aggregated by this column
(e.g. ``"trail"`` or ``"individual_id"``) before computing the quality
categories.  This ensures each group is counted only once using a majority
vote across its predictions.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_confusion
Plot CV vs. test confusion matrices for each model.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_confusion_and_inference
Backward compatible wrapper calling :func:`plot_confusion` and :func:`plot_inference`.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_hyperparam_heatmap
Plot a heatmap visualising mean CV accuracy across preprocessing options.

Parameters
----------
df:
    DataFrame with the preprocessing options and CV scores.  Columns using
    the names produced by :class:`skopt.BayesSearchCV` (e.g.
    ``select__method`` and ``mean_test_score``) are mapped automatically.
out_dir:
    Directory where the plot will be saved.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_individual_probabilities
Plot distribution of predicted sex probabilities for each individual.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_inference
Plot predicted sex counts for the inference split.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_model_quality_heatmaps
Deprecated wrapper for :func:`plot_quality_heatmaps`.

This function now simply calls :func:`plot_quality_heatmaps` without
additional parameters so that all available models are plotted.  It will
be removed in a future version.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_quality

## FIT_python.pipeline_sex.sex_predict_and_visualisation.plot_quality_heatmaps
Plot prediction-quality heatmaps.

Parameters ``pred_col`` and ``proba_cols`` can be used to display the
heatmap for a specific model and split.  When they are omitted, the
function behaves like the other plotting helpers and iterates over all
available models and the ``train``/``test`` splits automatically.
If ``group_by`` is provided, predictions are aggregated per group before
counting cells in the heatmap.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.predict_all
Return dataframe with model predictions for ``species``.

Parameters
----------
species:
    The species folder under ``data/splits``.
prefer_generic:
    If ``True``, models are loaded from the shared
    ``random_search_standard_metrics`` directory when present. Ignored when
    ``models_dir`` is given.
models_dir:
    Custom directory containing the trained models. The predictions CSV is
    also written to this directory. When ``None`` (default) the directory is
    determined automatically based on ``prefer_generic`` and the existence of
    a species-specific directory.
include_inference:
    Include the ``inference`` split if the corresponding parquet exists.
reuse_csv:
    When ``True`` the existing prediction CSV must be present, otherwise a
    ``FileNotFoundError`` is raised. Set to ``False`` to recompute the
    predictions if the file is missing.
use_cv_train_predictions:
    If ``True`` use out-of-fold predictions already stored in
    ``train.parquet`` instead of computing new predictions for the
    training split.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.predict_all_species
Predict sex for all species and combine into a single CSV.

Parameters
----------
species_list:
    Optional list of species names. When ``None`` the function iterates over
    the sub-directories of ``data/splits``.

## FIT_python.pipeline_sex.sex_predict_and_visualisation.predict_simple_baseline
Return predictions of the simple baseline model for ``species``.

Parameters
----------
species:
    Species folder under ``data/splits``.
exp_dir:
    Directory containing the trained models produced by
    :func:`run_simple_baseline_all_species`.
include_inference:
    Include the ``inference`` split when it exists.
reuse_csv:
    When ``True`` the function expects ``{species}_baseline_predictions.csv``
    to exist in ``exp_dir``. If the file is missing a ``FileNotFoundError``
    is raised. Set to ``False`` to recompute the predictions.

Returns
-------
pandas.DataFrame
    DataFrame with the original splits and three additional columns:
    ``pred_baseline_sex``, ``pred_baseline_proba_f`` and
    ``pred_baseline_proba_m``.

## FIT_python.pipeline_sex.simple_baseline._load_split

## FIT_python.pipeline_sex.simple_baseline.predict_simple_baseline
Delegate to :func:`sex_predict_and_visualisation.predict_simple_baseline`.

## FIT_python.pipeline_sex.simple_baseline.run_simple_baseline_all_species
Train a simple LDA baseline for each species.

Each species split is loaded from :data:`SPLITS_DIR` and an LDA model with
forward feature selection is trained using cross-validation.  Results are
stored in ``exp_dir`` and optionally the per-sample predictions are written
for later analysis.

Parameters
----------
exp_dir : Path
    Directory where ``raw_results.csv`` and models will be stored.
n_jobs : int, optional
    Number of CPU cores to use during cross-validation. ``-1`` uses all
    available cores.
progress : bool, optional
    Show a progress bar for species and cross-validation when ``True``.
max_features : int, optional
    Maximum number of features to select during forward selection.
reuse_results : bool, optional
    When ``True`` existing ``raw_results.csv`` and model files in
    ``exp_dir`` are loaded and returned instead of training new models.
save_predictions : bool, optional
    When ``True`` baseline predictions for each species are written to
    ``{exp_dir}/{species}_baseline_predictions.csv`` after training.

Returns
-------
pandas.DataFrame
    Table with one row per species containing the evaluation metrics.

## FIT_python.utils.aggregate_all_folds.aggregate_all_folds
Combine per-species ``master_pairs.parquet`` tables.

Parameters
----------
base_dir:
    Directory containing one sub-folder per species with a
    ``master_pairs.parquet`` file.
out_path:
    Optional location where the combined parquet should be written.
    When not provided, the file is saved as ``all_species_folds.parquet``
    inside ``base_dir``.
origin:
    ``origin`` label to filter by (e.g. ``"cv"`` or ``"test"``).

Returns
-------
pandas.DataFrame
    Concatenated data frame of all pairs with an added ``species`` column.

## FIT_python.utils.debug_utils.debug_report
Print debug statistics if ``config.DEBUG_MODE`` is True.

## FIT_python.utils.transformations.TransformationPipeline
Manage a sequence of ``QTransform`` objects.
