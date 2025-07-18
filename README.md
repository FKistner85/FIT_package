# FIT_python

FIT Otter: Data import and analysis pipeline. This package bundles the tools and
pipelines used in the "FIT" project for preparing datasets, running
preprocessing steps and training models.

## Installation

Create a Python environment (e.g. using `venv` or conda) with Python 3.11 or
newer and install the dependencies:

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
res = run_pipeline(df, target_col="sex")
```

## Running Tests

Install the requirements as shown above and then execute:

```bash
pytest
```

The unit tests located in `tests/` cover a small subset of the pipeline logic.
