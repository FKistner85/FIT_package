# FIT_python

**FIT Otter — footprint analysis and machine-learning pipeline**

`FIT_python` is the Python package powering the *FIT* research project.
It provides end-to-end tooling for importing and cleaning raw footprint
data, splitting datasets, running configurable preprocessing pipelines,
training sex-classification and individual-identification models, and
producing publication-ready visualisations.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Configuration & environment variables](#4-configuration--environment-variables)
5. [Data layout & directory structure](#5-data-layout--directory-structure)
6. [Pipelines](#6-pipelines)
   - 6.1 [Data import & cleaning](#61-data-import--cleaning)
   - 6.2 [Data splitting](#62-data-splitting)
   - 6.3 [Sex-classification pipeline](#63-sex-classification-pipeline)
   - 6.4 [Pairwise individual-identification pipeline](#64-pairwise-individual-identification-pipeline)
   - 6.5 [Sequential holdout & baseline evaluation](#65-sequential-holdout--baseline-evaluation)
7. [Reusable pipeline steps](#7-reusable-pipeline-steps)
8. [Visualisation utilities](#8-visualisation-utilities)
9. [GUI annotator](#9-gui-annotator)
10. [Scripts](#10-scripts)
11. [Notebooks](#11-notebooks)
12. [Tests](#12-tests)
13. [Supported species](#13-supported-species)
14. [Extending the package](#14-extending-the-package)

---

## 1. Project overview

The *FIT* project studies the feasibility of non-invasive individual
identification and sex classification from animal footprints.  Geometric
landmark coordinates are extracted from scanned footprint images, and a
series of configurable machine-learning pipelines is applied to answer two
core research questions:

| Task | Module |
|------|--------|
| Is a footprint from a male or female? | [`pipeline_sex`](src/FIT_python/pipeline_sex/) |
| Do two footprint trails belong to the same individual? | [`pipeline_individual_id`](src/FIT_python/pipeline_individual_id/) |

Both pipelines share a common set of modular preprocessing steps (see
[§7](#7-reusable-pipeline-steps)) and are driven by the central
configuration in [`FIT_python/config.py`](src/FIT_python/config.py).

---

## 2. Repository layout

```
FIT_package/
├── src/
│   └── FIT_python/               # main package
│       ├── config.py             # central configuration & paths
│       ├── data_split_and_summary/   # data import, cleaning, splitting
│       ├── general_pipeline_steps/   # reusable sklearn transformers
│       ├── gui_annotator/            # PyQt5 landmark annotation tool
│       ├── pipeline_individual_id/   # individual-ID pipeline
│       ├── pipeline_sex/             # sex-classification pipeline
│       ├── utils/                    # shared helpers (paths, captions …)
│       └── Visualisations/           # plot helpers and style sheets
├── docs/                         # detailed documentation per module
│   ├── pipeline_steps/           # one markdown per preprocessing step
│   ├── pipeline_wrappers/        # wrapper-level documentation
│   └── pipeline_individual_id/   # individual-ID specific docs
├── notebooks/                    # Jupyter analysis notebooks
├── scripts/                      # maintenance & utility scripts
├── tests/                        # pytest test suite
├── data/                         # raw data (not version-controlled)
├── results/                      # pipeline outputs (not version-controlled)
├── pyproject.toml                # build metadata & dependency list
├── requirements.txt              # pip-installable dependency list
└── environment.yml               # conda environment definition
```

---

## 3. Installation

**Requirements:** Python ≥ 3.11.

### pip

```bash
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yml
conda activate fit_python        # or the name defined in environment.yml
```

### Editable install (development)

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `coverage`, `flake8`, `mypy`, `jupyterlab`, and other
development helpers.

### Optional deep-learning extras

```bash
pip install torch          # required for siamese / supervised-UMAP pipelines
```

> The annotation GUI requires **PyQt5**, which is already listed in the
> main dependencies.  If you see `ModuleNotFoundError: No module named
> 'PyQt5'` run `pip install PyQt5`.

The `requirements.txt` and `environment.yml` files are generated from
`pyproject.toml` via [`scripts/sync_deps.py`](scripts/sync_deps.py).

---

## 4. Configuration & environment variables

All runtime configuration lives in
[`src/FIT_python/config.py`](src/FIT_python/config.py) and the `CONFIG`
dictionary.  The full reference is in
[`docs/config_reference.md`](docs/config_reference.md).

| Variable | Environment variable | Default | Purpose |
|----------|---------------------|---------|---------|
| `EXPERIMENT_ROOT` | `FIT_EXPERIMENT_ROOT` | `results/<timestamp>/` | Root for all output |
| `RAW_DIR` | `FIT_RAW_DIR` | `<EXPERIMENT_ROOT>/data/raw` | Immutable raw CSV files |
| `EXPERIMENT_DIR` | `FIT_EXPERIMENT_NAME` | `EXPERIMENT_ROOT/experiments/dissertation_notebook` | Per-experiment sub-folder |
| `DEBUG_MODE` | — | `False` | Print shapes & NaN counts after each step |

Set environment variables **before** importing the package or launching
Jupyter so that paths are initialised correctly:

```bash
export FIT_EXPERIMENT_ROOT=/path/to/my_experiment
export FIT_RAW_DIR=/path/to/raw_data
jupyter lab
```

Or set them inside a notebook cell:

```python
import os
os.environ["FIT_EXPERIMENT_ROOT"] = "/path/to/my_experiment"
# import FIT_python *after* setting the variable
from FIT_python import config
```

### Debug mode

```python
from FIT_python import config
config.DEBUG_MODE = True
```

Or pass `debug=True` to `PipelineWrapper` / `run_pipeline`.

---

## 5. Data layout & directory structure

```
<EXPERIMENT_ROOT>/
├── data/
│   ├── raw/           # immutable original CSV files (set via FIT_RAW_DIR)
│   ├── cleaned/       # output of DataImportWrapper.clean_all()
│   ├── splits/        # train / test / inference splits per species
│   └── processed/
│       ├── splits/
│       ├── scaled/
│       ├── feature_selected/
│       ├── dim_reduced/
│       └── numeric/
└── results/
    ├── dataprocessing/<species>/figures/ & tables/
    ├── sex_modelling/<species>/models/ figures/ tables/
    └── individual_id/<species>/models/ figures/ tables/
```

Path helpers are available via `FIT_python.utils.paths.get_species_paths`.
See [`docs/paths.md`](docs/paths.md) for the full canonical output layout.

---

## 6. Pipelines

### 6.1 Data import & cleaning

**Module:** [`data_split_and_summary`](src/FIT_python/data_split_and_summary/)  
**Docs:** [`docs/pipeline_steps/01_data_loading.md`](docs/pipeline_steps/01_data_loading.md),
[`docs/pipeline_steps/02_cleaning.md`](docs/pipeline_steps/02_cleaning.md)

```python
from FIT_python.data_split_and_summary.data_import_wrapper import DataImportWrapper

DataImportWrapper().clean_all()
# writes Parquet files to <EXPERIMENT_ROOT>/data/cleaned/
```

`clean_all()` reads every raw CSV from `RAW_DIR`, harmonises column
names, converts numeric fields via `transform_utils.convert_numeric`, and
stores cleaned tables as Parquet.

---

### 6.2 Data splitting

**Module:** [`data_split_and_summary`](src/FIT_python/data_split_and_summary/)  
**Docs:** [`docs/pipeline_steps/03_splitting.md`](docs/pipeline_steps/03_splitting.md),
[`docs/split_logic_details.md`](docs/split_logic_details.md)

```python
from FIT_python.data_split_and_summary.datasplit_and_summary_wraper import (
    SplitWrapper, SummaryWrapper
)

SplitWrapper().split_all(reuse_splits=True)
SummaryWrapper().summarize_all()
```

Key behaviour:
- `stratified_individual_split` keeps all footprints of one individual
  inside a single partition while preserving the male/female ratio.
- When fold cross-validation is needed, `_make_folds` generates
  `NUM_FOLDS` (default 5) stratified groups (see
  [`docs/pipeline_steps/03_splitting.md`](docs/pipeline_steps/03_splitting.md)).
- Split files are written as Parquet and optionally as CSV.

---

### 6.3 Sex-classification pipeline

**Module:** [`pipeline_sex`](src/FIT_python/pipeline_sex/)  
**Docs:** [`docs/pipeline_sex_methodology.md`](docs/pipeline_sex_methodology.md),
[`docs/pipeline_wrappers/pipeline_wrapper_sex.md`](docs/pipeline_wrappers/pipeline_wrapper_sex.md)

#### Quick start

```python
from FIT_python.pipeline_sex.pipeline_wrapper_sex import run_pipeline

results = run_pipeline(df, target_col="sex", n_jobs=-1, debug=True)
```

#### Training workflow

```python
from FIT_python.pipeline_sex.pipeline_wrapper_sex import PipelineWrapper

wrapper = PipelineWrapper(debug=True)
wrapper.prepare()   # assemble sklearn Pipeline from config
wrapper.train()     # five-fold CV → balanced accuracy logged to raw_results.csv
```

`PipelineWrapper.train()` iterates over species directories in
`data/splits`, performs cross-validation with
`cross_val_predict(method="predict_proba")`, and writes:

| Artifact | Location |
|----------|----------|
| `raw_results.csv` | `results/sex_modelling/<species>/tables/` |
| Fitted models | `results/sex_modelling/<species>/models/` |
| Best model per species | `results/sex_modelling/<species>/models/best/` |

#### Bayesian hyperparameter search

The search space (classifiers, outlier methods, scalers, feature
selectors, dimensionality reducers) is defined in
`CONFIG["pipeline_sex"]["search_spaces"]` — see
[`docs/config_reference.md`](docs/config_reference.md) and
[`docs/extend_bayes_search_config.md`](docs/extend_bayes_search_config.md).

```python
from FIT_python.pipeline_sex.search import run_otter_search_sex

run_otter_search_sex(n_iter=50, cv="fold", reuse_results=True)
```

#### Prediction & visualisation

```python
import os
os.environ["FIT_EXPERIMENT_ROOT"] = "/path/to/my_experiment"

from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all
predict_all("eurasian_otter", reuse_csv=False)
```

Produces confusion matrices, quality plots, heatmaps, and
individual-level probability charts.

---

### 6.4 Pairwise individual-identification pipeline

**Module:** [`pipeline_individual_id`](src/FIT_python/pipeline_individual_id/)  
**Docs:** [`docs/pairwise_individual_id_pipeline.md`](docs/pairwise_individual_id_pipeline.md),
[`docs/pipeline_individual_id/`](docs/pipeline_individual_id/)

The pipeline answers whether two movement trails originate from the same
individual.

#### Trail generation

```python
from FIT_python.pipeline_individual_id.generate_trails_and_trailpairs import (
    generate_pairwise_comparisons_from_df
)

pairs_df = generate_pairwise_comparisons_from_df(df)
# columns: trail_a, trail_b, same_individual, fold, sex, …
```

Each row is labelled `same_individual = True/False` and assigned a fold
based on individual-level stratified splitting.

#### Projection & distance computation

```python
from FIT_python.pipeline_individual_id.pairwise_individual_id_pipeline import (
    run_all_pairwise_projections_parallel
)

results = run_all_pairwise_projections_parallel(pairs_df, n_jobs=-1)
```

For each pair the function applies (in order):
1. Optional outlier cleaning — [`OutlierCleanerTransformer`](#outlier-cleaning)
2. Feature scaling — [`FeatureScalerTransformer`](#feature-scaling)
3. Feature selection — [`FeatureSelectionTransformer`](#feature-selection)
4. Dimensionality reduction — [`DimensionalityReducerTransformer`](#dimensionality-reduction)
5. Distance computation in the reduced space

Supported distance metrics (see
[`docs/pipeline_individual_id/distance_metrics.md`](docs/pipeline_individual_id/distance_metrics.md)):

| Metric | Description |
|--------|-------------|
| Euclidean | ℓ₂ distance |
| Manhattan | ℓ₁ distance |
| Cosine | Angle between vectors (magnitude-independent) |
| Chebyshev | Max component-wise difference |
| Canberra | Normalised absolute differences |
| Bray-Curtis | Compositional dissimilarity |

#### Population estimation

```python
from FIT_python.pipeline_individual_id.population_estimation import (
    cluster_population, compute_erd, optimal_cutoff
)

n_predicted = cluster_population(dist_matrix, cutoff=0.5)
erd = compute_erd(n_predicted, true_n=8)
cutoff, (low, high) = optimal_cutoff(dist_matrix, true_n=8)
```

See [`docs/pipeline_individual_id/population_estimation.md`](docs/pipeline_individual_id/population_estimation.md).

#### Overlap evaluation

```bash
python -m FIT_python.pipeline_individual_id.overlap_evaluation \
    experiments/fit_start_to_finish/id_baseline
```

Searches experiment directories for `all_splits.csv` files, finds the
chi-square probability `p` that maximises F1, and saves confusion matrix
plots.

---

### 6.5 Sequential holdout & baseline evaluation

**Module:** [`pipeline_individual_id`](src/FIT_python/pipeline_individual_id/)  
**Docs:** [`docs/FIT_start_to_finish_steps.md`](docs/FIT_start_to_finish_steps.md)

```python
from FIT_python.pipeline_individual_id.simple_baseline import (
    run_simple_baseline_otter, run_baseline_all_species
)

best_k = run_simple_baseline_otter(EXP_DIR)
run_baseline_all_species(EXP_DIR, best_k, cutoff)
```

Outputs per species:
- `summary.csv` with Ward cut-offs and confidence intervals
- `raw_results.csv` with collected metrics
- `id_baseline/fig/bcr_comparison.png`

The helper
[`scripts/aggregate_all_folds.py`](scripts/aggregate_all_folds.py) (or
`FIT_python.pipeline_individual_id.utils.aggregate_all_folds`) merges all
`master_pairs.parquet` files into `all_species_folds.parquet`.

---

## 7. Reusable pipeline steps

All steps are scikit-learn-compatible transformers located in
[`general_pipeline_steps`](src/FIT_python/general_pipeline_steps/).
They can be composed into any `sklearn.pipeline.Pipeline`.

### Imputation

**File:** [`imputation_wrapper.py`](src/FIT_python/general_pipeline_steps/imputation_wrapper.py)  
**Docs:** [`docs/pipeline_wrappers/imputation_wrapper.md`](docs/pipeline_wrappers/imputation_wrapper.md)

Uses an iterative random-forest imputer (`missingpy.MissForest` / sklearn
`IterativeImputer`).  Configured via
`CONFIG["general_pipeline_steps"]["imputation_defaults"]`.

### Outlier cleaning

**File:** [`outlier_wrapper.py`](src/FIT_python/general_pipeline_steps/outlier_wrapper.py)  
**Docs:** [`docs/pipeline_wrappers/outlier_wrapper.md`](docs/pipeline_wrappers/outlier_wrapper.md)

`OutlierCleanerTransformer` supports:
- `"clip"` — percentile clipping (default 1st–99th)
- `"zscore"` — z-score winsorising

### Feature scaling

**File:** [`feature_scaler_wrapper.py`](src/FIT_python/general_pipeline_steps/feature_scaler_wrapper.py)  
**Docs:** [`docs/pipeline_wrappers/feature_scaler_wrapper.md`](docs/pipeline_wrappers/feature_scaler_wrapper.md)

`FeatureScalerTransformer` accepts `method="standard"` or
`method="robust"`.

### Feature selection

**File:** [`feature_selection_wrapper.py`](src/FIT_python/general_pipeline_steps/feature_selection_wrapper.py)  
**Docs:** [`docs/pipeline_wrappers/feature_selection_wrapper.md`](docs/pipeline_wrappers/feature_selection_wrapper.md)

`FeatureSelectionTransformer` strategies:

| `method` | Algorithm |
|----------|-----------|
| `"forward"` | Stepwise forward selection |
| `"random_forest"` | Feature importances |
| `"lasso"` | LASSO-based selection |
| `"variance"` | Variance threshold |
| `"univariate"` | Univariate statistical tests |

### Dimensionality reduction

**File:** [`dimensionality_reduction_wrapper.py`](src/FIT_python/general_pipeline_steps/dimensionality_reduction_wrapper.py)  
**Docs:** [`docs/pipeline_wrappers/dimensionality_reduction_wrapper.md`](docs/pipeline_wrappers/dimensionality_reduction_wrapper.md)

`DimensionalityReducerTransformer` supports `"pca"`, `"umap"`, `"tsne"`,
and `"lda"`.  Defaults are in
`CONFIG["general_pipeline_steps"]["dim_reducer_defaults"]`.

### Models

**File:** [`models.py`](src/FIT_python/general_pipeline_steps/models.py)  
**Docs:** [`docs/pipeline_wrappers/models.md`](docs/pipeline_wrappers/models.md)

Available classifiers: LDA (`lda_svd`), XGBoost (`xgb_std`), and
others defined in the `MODELS` registry.

---

## 8. Visualisation utilities

**Module:** [`Visualisations`](src/FIT_python/Visualisations/)

| File | Purpose |
|------|---------|
| `plots_utils.py` | Confusion matrices, quality plots, probability charts |
| `display_mapping.py` | Label-to-display-name mapping for plots |
| `id_style.py` | Colour and marker styles for individual-ID figures |
| `plot_style.py` | Global matplotlib style helpers |

Colour palettes and sex-label maps are defined in
[`config.py`](src/FIT_python/config.py):

```python
SEX_COLORS   = {"Female": "#800000", "Male": "#000080", "Unknown": "#FFA500"}
TRAIN_COLORS = SEX_COLORS
TEST_COLORS  = {k: _lighten(v, 0.5) for k, v in SEX_COLORS.items()}
```

Generated figure inventory: [`docs/generated_figures.md`](docs/generated_figures.md)

---

## 9. GUI annotator

**Module:** [`gui_annotator`](src/FIT_python/gui_annotator/)  
**Docs:** [`docs/config_reference.md`](docs/config_reference.md#gui_annotator)

A PyQt5 desktop application for manually annotating landmark coordinates
on scanned footprint images.

```python
from FIT_python.gui_annotator.gui_app import launch_annotator
launch_annotator()
```

Paths used by the tool:

| Resource | Path |
|----------|------|
| Raw images | `RAW_DIR/images/` |
| Processed images | `PROCESSED_DIR/images/` |
| Annotations | `PROCESSED_DIR/annotations/` |
| Reference templates | `RAW_DIR/reference_templates/` |

---

## 10. Scripts

Utility scripts in [`scripts/`](scripts/):

| Script | Purpose |
|--------|---------|
| [`sync_deps.py`](scripts/sync_deps.py) | Regenerate `requirements.txt` and `environment.yml` from `pyproject.toml` |
| [`aggregate_all_folds.py`](scripts/aggregate_all_folds.py) | Merge per-species fold Parquet files |
| [`build_annotation_csv.py`](scripts/build_annotation_csv.py) | Convert annotation JSON files to CSV |
| [`compute_derived_landmarks.py`](scripts/compute_derived_landmarks.py) | Calculate derived geometric features from raw landmarks |
| [`generate_figures_markdown.py`](scripts/generate_figures_markdown.py) | Auto-generate `docs/generated_figures.md` |
| [`generate_function_docs.py`](scripts/generate_function_docs.py) | Generate per-function documentation stubs |
| [`validate_reprojection.py`](scripts/validate_reprojection.py) | Validate coordinate reprojection results |

---

## 11. Notebooks

The main analysis notebook is
[`notebooks/Dissertation_noteb_final .ipynb`](notebooks/).

It executes the full pipeline end-to-end:

1. Dataset summary → `dataset_overview.parquet`
2. Data cleaning → `data/cleaned/`
3. Data splits → `data/splits/`
4. Correlation matrix
5. Top-feature selection & UMAP visualisation
6. Sex-classification baseline (stepwise LDA)
7. Bayesian model search
8. Prediction & evaluation
9. Sex-feature experiment
10. Sequential holdout & individual-ID baseline
11. Fold-based cross-validation
12. Global cut-off evaluation
13. Distance metric comparison

A step-by-step description is in
[`docs/FIT_start_to_finish_steps.md`](docs/FIT_start_to_finish_steps.md).

---

## 12. Tests

Install all dependencies (including dev extras) and run:

```bash
pytest
```

The test suite in [`tests/`](tests/) covers:

- Outlier cleaning, feature selection, dimensionality reduction, scaling
- Data import utilities and numeric transforms
- Pairwise trail generation and estimator cloning
- Sequential holdout logic
- Sex-model prediction, reuse, and directory resolution
- Population estimation and silhouette scoring
- Visualisation helpers

Optional dependencies needed by the tests:

```bash
pip install PyQt5 torch
```

---

## 13. Supported species

| Key in code | Common name | Latin name |
|-------------|-------------|------------|
| `eurasian_otter` | Eurasian Otter | *Lutra lutra* |
| `amur_tiger` | Amur Tiger | *Panthera tigris altaica* |
| `bengal_tiger` | Bengal Tiger | *Panthera tigris tigris* |
| `cheetah` | Cheetah | *Acinonyx jubatus* |
| `giant_panda` | Giant Panda | *Ailuropoda melanoleuca* |
| `lowlandtapir` | Lowland Tapir | *Tapirus terrestris* |
| `mountain_lion` | Mountain Lion | *Puma concolor* |
| `white_rhino` | White Rhinoceros | *Ceratotherium simum* |

Species keys are defined in `SPECIES_MODEL_MAP` in
[`config.py`](src/FIT_python/config.py).

---

## 14. Extending the package

- **Add a new classifier:** register it in `MODELS` in
  [`general_pipeline_steps/models.py`](src/FIT_python/general_pipeline_steps/models.py)
  and add it to `CONFIG["pipeline_sex"]["model_keys"]`.
- **Add a new search space:** follow
  [`docs/extend_bayes_search_config.md`](docs/extend_bayes_search_config.md).
- **Add a new pipeline step:** subclass `sklearn.base.TransformerMixin`,
  place it in `general_pipeline_steps/`, and reference it in the
  pipeline order via `CONFIG`.
- **Add a new species:** insert it into `SPECIES_MODEL_MAP` and provide
  a matching directory under `RAW_DIR`.

All configurable values must be read from `FIT_python.config.CONFIG`.
Avoid hard-coded literals — see the full reference in
[`docs/config_reference.md`](docs/config_reference.md).
