# FIT Modeling Pipeline

This document gives an overview of the end-to-end pipeline implemented in this repository. Each stage is realised as a lightweight script so that steps can be run independently or chained in notebooks and tests.

## 1. Data Loading
Raw CSV or Excel files reside in `data/raw`. The helper function `load_raw_files` cleans column names and normalises identifier fields. The optional script `dataload.py` prints the shapes and previews of these data sets for inspection.

## 2. Cleaning
`clean_data.py` copies each raw file to Parquet under `data/cleaned`. Using Parquet speeds up subsequent I/O while keeping the data untouched.

## 3. Train/Test Splitting
`create_splits.py` generates train and test splits with group awareness so that individuals never appear in both sets. The splits are stored as Parquet files in `data/processed/splits`.

## 4. Scaling
`scale_splits.py` standardises feature columns using scikit-learn's `StandardScaler`. Each dataset is processed separately and written to `data/processed/scaled`.

## 5. Feature Selection
`create_feature_selection.py` performs a very simple variance-based selection, keeping a fixed number of features with the highest variance. The resulting files go to `data/processed/feature_selected`.

## 6. Dimensionality Reduction
`create_dim_reduction.py` applies Principal Component Analysis (PCA) with two components. PCA is a classical method for projecting high dimensional data to a lower dimensional subspace while maximising retained variance. A detailed treatment can be found in Jolliffe (2002).

## 7. Conversion to NumPy
`transform_splits.py` converts the train/test splits to NumPy arrays and stores them under `data/processed/numeric`. Labels such as `sex` are mapped to integers for modelling.

## 8. Model Comparison
Two scripts illustrate modelling:

- `compare_models.py` trains a logistic regression and a support vector machine (SVM) on each dataset and reports accuracy.
- `model_comparison_sex.py` performs an extensive grid search over many classifiers (logistic regression, SVM, random forest, k‑NN, LDA, Naive Bayes, AdaBoost and optionally gradient boosting models). Results are written to `results/data`.

Logistic regression is a linear classifier; see the scikit-learn user guide for details. SVMs maximise the margin between classes and can employ kernel functions for non-linear boundaries (Cortes & Vapnik, 1995).

## 9. Summary and Visualisation
`create_summary.py` and `summary.py` aggregate counts of individuals, trails and features across splits. `plot_summary.py` creates stacked bar charts from these tables and saves them under `results/figures`.

## 10. Landmark Mapping
`create_landmark_map.py` analyses feature names in the Eurasian Otter data to generate JSON maps relating landmarks to feature columns. These mappings assist later analysis steps.

---

Together these scripts form a modular pipeline from raw data ingestion to model evaluation. Each stage reads its input from the previous step's output directory, enabling easy experimentation and incremental processing.

