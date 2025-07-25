# FIT_python

FIT Otter: Data import and analysis pipeline. This package bundles the tools and
pipelines used in the "FIT" project for preparing datasets, running
preprocessing steps and training models.

## Installation

Create a Python environment (e.g. using `venv` or conda) with Python 3.11 or
newer. Dependency versions are defined in `pyproject.toml`. The accompanying
`requirements.txt` and `environment.yml` files are generated from that list via
`scripts/sync_deps.py`.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively you can create a conda environment via `environment.yml`:

```bash
conda env create -f environment.yml
```

The annotation GUI relies on `PyQt5`, which is listed in the main
dependencies.

## Usage

The package exposes several pipeline utilities under `FIT_python`. Refer to the
`docs/` directory for detailed information on individual workflows.

A minimal example running the sex-classification pipeline could look like:

```python
from FIT_python.pipeline_sex.pipeline_wrapper_sex import run_pipeline

# prepare your training dataframe `df`
res = run_pipeline(df, target_col="sex", n_jobs=-1, debug=True)
```

### Debug output

Set `FIT_python.config.DEBUG_MODE = True` or pass `debug=True` when
creating a `PipelineWrapper` (or calling `run_pipeline`) to print shapes
and counts of missing or infinite values after each preprocessing step.
This aids troubleshooting
when building new pipelines.  For manual control you can instantiate
and run the wrapper directly:

```python
wrapper = PipelineWrapper(debug=True)
wrapper.prepare()
wrapper.train()
```

## Experiment root

Most paths used by the pipelines are built relative to
`FIT_python.config.EXPERIMENT_ROOT`.  The value of this variable is obtained
from the environment variable `FIT_EXPERIMENT_ROOT` and defaults to the current
working directory if not set.  Directories such as `data/` and `results/` are
therefore resolved inside the configured experiment root.

Individual experiments are stored under `FIT_python.config.EXPERIMENT_DIR`,
which defaults to `EXPERIMENT_ROOT/experiments/<SOFT_CONFIG["experiment"]["name"]>`.
Set the environment variable `FIT_EXPERIMENT_NAME` or adjust
``SOFT_CONFIG["experiment"]["name"]`` before importing ``FIT_python.config`` to
write results to a different subfolder.

Raw data is loaded from ``FIT_python.config.RAW_DIR``.  You can override this
path via the environment variable ``FIT_RAW_DIR``.  If unset it defaults to
``<EXPERIMENT_ROOT>/data/raw``.

After training has completed you can point the environment variable to the
experiment directory and run predictions:

```python
import os
os.environ["FIT_EXPERIMENT_ROOT"] = "/path/to/my_experiment"

from FIT_python.pipeline_sex.sex_predict_and_visualisation import predict_all
predict_all('eurasian_otter', reuse_csv=False)
```

## Running Tests

Install the requirements as shown above and then execute:

```bash
pytest
```

### Optional testing dependencies

Before executing the test suite make sure that optional GUI/testing
packages such as `pandas` and `PyQt5` are installed. Missing modules will
prevent pytest from collecting the tests and you will see import errors
like `ModuleNotFoundError: No module named pandas`.

Running the tests requires `PyQt5`. If you encounter a
`ModuleNotFoundError` when executing `pytest`, install the dependency via

```bash
pip install PyQt5
```

Deep learning utilities in `pipeline_individual_id` rely on `torch`. Install it
via

```bash
pip install torch
```

The unit tests located in `tests/` cover a small subset of the pipeline logic.

Further details on coordinate reprojection can be found in
`docs/coordinate_reprojection.md`.


### Overlap evaluation

The helper `FIT_python.pipeline_individual_id.overlap_evaluation` searches
experiment directories for `all_splits.csv` files, determines the chi-square
probability `p` that maximises F1 for the ellipse-overlap classifier and stores
a confusion matrix plot next to each CSV.

```bash
python -m FIT_python.pipeline_individual_id.overlap_evaluation \
    experiments/fit_start_to_finish/id_baseline
```
