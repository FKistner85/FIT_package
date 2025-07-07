# Individual Identification Pipeline Methodology

This document summarises the pairwise identification workflow implemented in the modules under `src/FIT_python/pipeline_individual_id`. Three versions are provided below: a brief overview, a detailed description (~20 pages when printed) and a medium-length synopsis.

## Condensed (1–2 pages)

1. **Pair Generation** – `generate_pairwise_comparisons_from_df` constructs trail pairs from the raw footprint table. Individual IDs are grouped, short trails are sampled and every pair receives a label `same_individual` along with a cross‑validation `fold`.
2. **Projection Pipeline** – `run_all_pairwise_projections_parallel` applies outlier cleaning, optional scaling, feature selection and dimensionality reduction to each pair. Distances between trail centres are computed using several metrics.
3. **RCV Sampling** – For each comparison a complementary reference set is created with `generate_rcv` to stabilise projections.
4. **Results** – The function returns a list of dictionaries containing coordinates, distance measures and optional sex model probabilities, which can be aggregated into a DataFrame for further analysis.

## Detailed (~20 pages)

### 1. Trail and Pair Construction
- **Grouping** – Input tables must contain an `individual_id` column. Missing IDs fall back to the `trail` label so that every record belongs to some group.
- **Chunks and Trails** – `generate_trails_and_trailpairs.py` samples non‑overlapping chunks of footprints per individual. From these chunks trails of configurable sizes are drawn, optionally limiting the number per animal.
- **Cross‑ and Within‑Individual Pairs** – All combinations of trails from different individuals are generated alongside within‑individual pairs from distinct chunks. Each entry stores the original row indices (`samples_a`, `samples_b`), unique trail IDs and whether both sides originate from the same individual.
- **Stratified Folds** – A `StratifiedKFold` on the trail size assigns a `fold` value to every pair so that validation splits maintain similar trail length distributions.
- **Summary Output** – The helper returns both the list of comparisons and a summary table reporting the number of animals and trails for each size, including averages of within‑ and between‑individual comparisons.

### 2. Outlier Cleaning and Scaling
- **OutlierCleanerTransformer** – This wrapper supports percentile clipping (`clip`) or z‑score winsorisation (`zscore`). Boundaries are learned during `fit` and applied in `transform` while preserving DataFrame columns.
- **FeatureScalerTransformer** – Features may be scaled with `StandardScaler` or `RobustScaler`. The transformer records the original column order and returns the same type as received.

### 3. Feature Selection
- **FeatureSelectionTransformer** – Several strategies are available: forward selection, random forest importance, variance thresholding, univariate `f_classif` scoring or a Lasso regression approach. The transformer keeps a ranking of all features and outputs the top `k`.

### 4. Dimensionality Reduction
- **DimensionalityReducerTransformer** – Supported methods include PCA, UMAP, t‑SNE, LDA, MDS and Isomap. Supervised variants (UMAP and LDA) require class labels. Component counts are clamped to the allowable range.

### 5. Pairwise Projection Workflow
- **Preparation** – `run_all_pairwise_projections_parallel` copies the data set, ensures numeric dtype for feature columns and optionally precomputes sex model probabilities.
- **Processing** – For every pair the function:
  1. Extracts feature matrices for trail A, trail B and the RCV set.
  2. Performs feature selection once with the maximum requested `k`.
  3. Iterates over outlier and scaler options, reducers, component counts and numbers of selected features.
  4. Appends sex probabilities as features if provided.
  5. Applies dimensionality reduction and computes distances (Euclidean, Manhattan, cosine, Chebyshev, Canberra, Bray‑Curtis and optionally Mahalanobis).
  6. Records centre coordinates, per‑pair distances and summary statistics for within‑ and between‑trail distances.
- **Parallelisation** – Pairs are processed in parallel via joblib with progress bars from `tqdm_joblib`. Intermediate results can be cached by `joblib.Memory`.

### 6. Output and Evaluation
- The returned list of dictionaries can be transformed into a DataFrame for statistical evaluation. Each row describes one projection configuration for a trail pair, including metadata (`ind_a`, `ind_b`, trail IDs, fold), distance metrics and the coordinates of all projected points.
- Results may be stored under `results/data` for further analysis or visualisation.

### 7. Reproducibility
- **Caching** – Expensive functions such as pair generation and pairwise computations are cached on disk to avoid recomputation.
- **Random Seeds** – Functions use a configurable `random_state` to make sampling and cross‑validation splits repeatable.

### 8. References
- Breiman, L. “Random Forests.” *Machine Learning* 45 (2001): 5–32.
- Hotelling, H. “Analysis of a Complex of Statistical Variables into Principal Components.” *Journal of Educational Psychology* 24 (1933): 417–441.
- McInnes, L., Healy, J., Melville, J. “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.” arXiv:1802.03426 (2018).
- Tibshirani, R. “Regression Shrinkage and Selection via the Lasso.” *Journal of the Royal Statistical Society* B 58 (1996): 267–288.
- van der Maaten, L., Hinton, G. “Visualizing Data using t‑SNE.” *JMLR* 9 (2008): 2579–2605.

## Compressed Detailed (5–10 pages)

1. **Generate Pairs** – Use `generate_trails_and_trailpairs.py` to build trail pairs with labels and folds. Missing IDs fall back to the trail name.
2. **Outlier & Scaling** – Clean extreme values and optionally scale numeric features.
3. **Select Features** – Rank all features and keep the best `k` for each configuration.
4. **Reduce Dimensions** – Project the selected features with PCA, UMAP, LDA and others.
5. **Compute Distances** – Measure distances between trail centres; store per‑pair statistics and coordinates.
6. **Aggregate Results** – Combine all result dictionaries into a table for analysis and visualisation.
7. **Reproducibility** – Cached computations and fixed seeds make experiments repeatable.

