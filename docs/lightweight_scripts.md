# Lightweight Scripts Reference

This document summarises the small utility scripts in the `scripts/` folder. Each script performs a single preprocessing or analysis step so that they can be chained together.  The descriptions below outline the purpose of each script, list the key options and, where relevant, briefly mention the algorithms used.

## `clean_data.py`
Copies raw CSV files to Parquet for faster loading.  The script calls
`load_raw_files` from `FIT_python.data_import_utils` to read all raw
datasets and then writes each one to the `cleaned` directory.

- **Uses**: `load_raw_files`
- **Complexity**: purely I/O bound; depends on file size and format conversion.

## `dataload.py`
Loads all raw data files and prints their shape and first rows for
inspection. It is mainly intended for interactive inspection of the
imported data.

- **Uses**: `load_raw_files`
- **Complexity**: linear in the number of rows read from disk.

## `create_splits.py`
Creates train/test splits for every dataset under `data/raw` using group-aware splitting. Output is written as Parquet files under `data/processed/splits`.

- **Uses**: `all_splits`, `train_test_group_split`, `split_path`
- **Complexity**: dominated by Pandas operations and the split logic; roughly linear in dataset size.

## `scale_splits.py`
Applies `StandardScaler` from scikit-learn to each train/test split. The script reads every split, scales feature columns and writes the result back.

- **Uses**: `StandardScaler`, `split_path`, `scaled_path`
- **Method**: Standardisation (zero mean, unit variance). See [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
- **Complexity**: O(n \* p) where *n* is number of rows and *p* the number of features.

## `create_feature_selection.py`
Selects the top features by variance on each scaled dataset and saves the reduced tables.

- **Uses**: `feature_selected_path`, `scaled_path`, `pandas.DataFrame.var`
- **Method**: Variance thresholding (here implemented by sorting variances and keeping the highest ones). This is a very light-weight form of feature selection.
- **Complexity**: O(n \* p) to compute variances.

## `create_dim_reduction.py`
Runs Principal Component Analysis (PCA) with a fixed number of components (two) on each feature-selected training split.

- **Uses**: `PCA`, `feature_selected_path`, `dim_reduced_path`
- **Method**: PCA finds orthogonal directions maximising variance. A canonical reference is Jolliffe "Principal Component Analysis" (Springer, 2002).
- **Complexity**: dominated by SVD, approximately O(min(n p^2, p n^2)).

## `transform_splits.py`
Converts Parquet train/test splits to NumPy arrays for modelling. Meta columns are dropped and labels are encoded numerically.

- **Uses**: `TransformWrapper.transform_dataset`
- **Complexity**: linear in the number of rows; mostly spent on file I/O and conversion.

## `compare_models.py`
Small example comparing `LogisticRegression` and `SVC` on the scaled splits. Accuracy for each dataset is written to CSV.

- **Uses**: `LogisticRegression`, `SVC`, `accuracy_score`
- **Methods**:
  - **Logistic Regression**: a linear classifier optimised via maximum likelihood with optional regularisation. See the scikit-learn user guide.
  - **Support Vector Machine (SVC)**: maximises the margin between classes using kernel functions (Cortes & Vapnik, 1995).
- **Complexity**:
  - Logistic Regression: typically O(n p) per iteration.
  - SVM with RBF kernel: between O(n^2) and O(n^3) depending on solver.

## `model_comparison_sex.py`
Comprehensive model comparison for the target `sex`. Supports many estimators including logistic regression, SVM, random forest, k-NN, LDA, Naive Bayes, AdaBoost and optional gradient boosting models. Hyperparameters are tuned using `GridSearchCV`.

- **Uses**: a large set of scikit-learn models (`LogisticRegression`, `SVC`, `RandomForestClassifier`, `KNeighborsClassifier`, `LinearDiscriminantAnalysis`, `GaussianNB`, `AdaBoostClassifier`) and optionally `CatBoostClassifier`, `LGBMClassifier`, `XGBClassifier` if installed. Hyperparameter search is performed with `GridSearchCV` and `PredefinedSplit`.
- **Complexity**: depends on the estimator. Grid search scales with the number of parameter combinations \* cross‑validation folds.

## `create_summary.py`
Computes dataset summary statistics such as number of individuals, trails and features. Summaries are stored under `results/data` for later plotting.

- **Uses**: `all_splits`, `summarize_dataset`, `summary_stats`
- **Complexity**: linear aggregation over the dataset.

## `summary.py`
Command-line interface that wraps the summary generation and optionally produces plots. It parses command line arguments and calls `run_summary`.

- **Uses**: `run_summary`
- **Complexity**: same as `create_summary.py` plus plotting if enabled.

## `plot_summary.py`
Reads `summary_datasets.csv` and produces stacked bar charts of individuals per split and sex. Plots are saved under `results/figures`.

- **Uses**: `load_summary`, `summary_for_species`, `plot_species`
- **Complexity**: dominated by matplotlib plotting functions; linear in the number of categories to display.

## `create_landmark_map.py`
For the Eurasian Otter dataset, analyses feature names to build mappings between landmarks and the columns representing distances, angles or times. Results are JSON files used by other parts of the pipeline.

- **Uses**: `load_raw_files`
- **Complexity**: string parsing over feature columns; negligible for typical dataset sizes.

---

### Notes on Algorithmic Cost
For simple scripts (cleaning, loading, splitting) the runtime is typically linear in dataset size. Feature selection by variance, scaling and PCA are also linear in the number of rows but depend on the number of features. Model comparison scripts can become expensive because cross‑validation needs to fit each model multiple times. Logistic regression scales roughly linearly with the number of samples, while kernel SVMs or ensemble methods (RandomForest, gradient boosting) can have quadratic or worse behaviour depending on hyperparameters and dataset dimensions.

