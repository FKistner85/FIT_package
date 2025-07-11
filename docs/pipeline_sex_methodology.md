# Sex Classification Pipeline Methodology

This document summarises the workflow implemented in the notebook `pipeline_sex_with grph.ipynb` and the helper modules under `src/FIT_python`. Three versions are provided: a short overview, a detailed explanation (~20 pages when printed), and a medium-length synopsis.

## Condensed (1–2 pages)

1. **Data Loading** – CSV files in `data/raw` are imported through `DataImportWrapper`, which normalises column names and converts numeric fields using `transform_utils.convert_numeric()`. Purely textual features are one-hot encoded during this step.
2. **Splitting** – `SplitWrapper.split_all()` creates species-specific directories with `train.parquet` and `test.parquet`. Footprints from the same individual are grouped together in a single split. Stratification on sex keeps male/female ratios balanced.
3. **Cross‑Validation (optional)** – If enabled, `group_stratified_kfold` assigns fold numbers that preserve sex distribution and prevent individuals from appearing in multiple folds.
4. **Pipeline Assembly** – `get_pipeline_steps()` builds a scikit‑learn pipeline with optional imputation, outlier cleaning, scaling, dimensionality reduction, feature selection and a classifier from `models.py`.
5. **Training & Evaluation** – `PipelineWrapper.train()` performs five‑fold cross-validation (via `RandomizedSearchCV`) and records balanced accuracy. The best models are saved under `results/data/sex_models*`.
6. **Prediction** – Saved pipelines can be reloaded with `joblib` to classify the sex of new footprints.

## Detailed (~20 pages)

### 1. Data Acquisition and Cleaning
- **Import** – Raw CSVs are stored in `data/raw`. `DataImportWrapper` iterates over these files, loads them into pandas DataFrames and harmonises column names. Metadata such as individual ID, sampling location and substrate type are preserved.
- **Numeric Conversion** – `convert_numeric()` from `transform_utils.py` replaces comma decimal separators with dots, converts available numbers and one-hot encodes columns that contain only strings. Duplicate rows are dropped, while missing values remain for imputation.

### 2. Train/Test/Inference Splitting
- **Core Logic** – `SplitWrapper.split_all()` reads the cleaned tables and writes per‑species folders under `data/splits`. For most datasets, `stratified_individual_split` ensures all footprints of an individual occur either entirely in the training or testing set, stratified by sex.
- **Special Case for Otters** – `create_train_test_split_otter` separates Portuguese individuals into an inference set. The remaining otter data is split by individual and sex using the same mechanism as above.
- **Cross-Validation Folds** – When cross-validation is required, `group_stratified_kfold` (based on `StratifiedGroupKFold`) adds a `Fold` column to the training set so that individuals remain grouped and each fold shows similar sex proportions. If not needed, this column may be ignored.
- **Split Summary** – `SummaryWrapper.summarize_all()` checks row counts, individual counts and sex distributions, saving bar charts to `results/figures`.

### 3. Pipeline Configuration
`pipeline_wrapper_sex.py` composes modular steps for preprocessing and modelling.

1. **Numeric Transformation** – `NumericTransformer` drops or encodes categorical columns and ensures purely numeric matrices for scikit‑learn.
2. **Imputation** – `ImputationWrapper` uses an iterative imputer with a RandomForestRegressor backbone (Breiman, 2001) to fill missing values.
3. **Outlier Handling** – `OutlierCleanerTransformer` performs percentile clipping or z‑score winsorisation, with user-defined thresholds.
4. **Scaling** – `FeatureScalerTransformer` applies either `StandardScaler` or `RobustScaler`.
5. **Dimensionality Reduction** – `DimensionalityReducerTransformer` supports PCA (Hotelling, 1933), UMAP (McInnes et al., 2018) and t‑SNE (van der Maaten & Hinton, 2008), among others. Parameters like component count and neighbours are configurable.
6. **Feature Selection** – `FeatureSelectionTransformer` offers forward selection, variance thresholding, univariate tests, random forest importance or Lasso-based ranking (Tibshirani, 1996).
7. **Classifier** – `models.py` defines logistic regression, SVM, random forest, extra trees, k‑NN, LDA and several boosting methods (XGBoost, LightGBM, CatBoost). Their parameters are exposed for tuning.

### 4. Training Workflow
- **PipelineWrapper** – This class handles preparation (data import, splitting, summary generation) and model training. The `.train()` method loops over species and model configurations.
- **Cross‑Validation** – Each configuration undergoes five‑fold cross-validation via `cross_val_score` with `balanced_accuracy_score` to handle class imbalance.
- **Metrics** – Test results include a classification report and balanced accuracy. Helper functions in `grouped_metrics.py` compute per-individual accuracy and majority-vote metrics.
- **Timing** – Transformation, imputation and prediction durations are measured using `perf_counter` to profile pipeline stages.

### 5. Hyperparameter Search
- **RandomizedSearchCV** – The notebook explores a predefined search space covering preprocessing options and classifier parameters. Sampled combinations are ranked by cross-validation balanced accuracy.
- **Parameter Examples** – Search grids include UMAP neighbour counts, clipping quantiles for outlier removal and the number of trees in a random forest.
- **Evaluation** – For each sample the mean cross-validation balanced accuracy and the test-set balanced accuracy are stored in `results/data/raw_results.csv`.

### 6. Model Persistence
- **Saving** – Pipelines are serialised via `joblib.dump()` and stored under `results/data/sex_models`. The best model per species is also copied to `results/data/sex_models_best`.
- **Loading** – Models can be reloaded with `joblib.load()` to classify new footprints without repeating preprocessing.

### 7. Reproducibility
- **Caching** – `joblib.Memory` caches intermediate results, accelerating repeated runs.
- **Random Seeds** – A project-wide seed from `config.py` is used by all stochastic operations (imputation, random forest, etc.) to ensure repeatability.
- **Data Integrity** – Intermediate data sets are saved as Parquet files to maintain schema and type information.

### 8. References
- Breiman, L. “Random Forests.” *Machine Learning* 45 (2001): 5–32.
- Hotelling, H. “Analysis of a Complex of Statistical Variables into Principal Components.” *Journal of Educational Psychology* 24 (1933): 417–441.
- Kohavi, R. “Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.” *IJCAI* (1995).
- McInnes, L., Healy, J., Melville, J. “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.” arXiv:1802.03426 (2018).
- Tibshirani, R. “Regression Shrinkage and Selection via the Lasso.” *Journal of the Royal Statistical Society* B 58 (1996): 267–288.
- van der Maaten, L., Hinton, G. “Visualizing Data using t-SNE.” *JMLR* 9 (2008): 2579–2605.

## Compressed Detailed (5–10 pages)

1. **Load and Clean Data** – Import CSVs with `DataImportWrapper` and clean numeric fields while retaining individual identifiers for stratification.
2. **Create Splits** – Use `SplitWrapper.split_all()` so each individual’s footprints live entirely in train or test. Sex stratification keeps class ratios stable. Otter data reserves Portuguese individuals for inference. Optional `group_stratified_kfold` adds fold numbers for cross-validation.
3. **Check Splits** – `SummaryWrapper.summarize_all()` confirms row counts and sex distributions, outputting bar charts.
4. **Build Pipeline** – `get_pipeline_steps()` assembles numeric conversion, imputation, outlier cleaning, scaling, dimensionality reduction, feature selection and the classifier. Arguments are validated against allowed choices.
5. **Train with Cross‑Validation** – `PipelineWrapper.train()` executes five-fold cross-validation, measures balanced accuracy and records processing time for each step.
6. **Hyperparameter Search** – In the notebook `RandomizedSearchCV` samples pipeline configurations and model parameters. Results are logged to `raw_results.csv`.
7. **Save Models** – Trained pipelines are saved under `results/data/sex_models` and duplicated in `sex_models_best` when they outperform previous runs.
8. **Predict** – Reload a saved model with `joblib.load()` to classify the sex of new footprints without repeating the full pipeline.
9. **Reproducibility** – Consistent random seeds and cached intermediate results ensure that experiments can be reproduced exactly.

