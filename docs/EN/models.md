# models.py

## Overview
Dictionary of predefined scikit-learn and gradient boosting classifiers.

## Key Components
- MODELS

### MODELS
Dictionary that maps short string keys to ready‑configured classifier instances
such as logistic regression, random forests and gradient boosting variants.
Select a model by key when building a pipeline. All dependencies like XGBoost,
LightGBM and CatBoost must be installed for the corresponding entries to work.
The presets cover a range from lightweight to heavy models; adjust parameters if
training time or accuracy do not meet expectations.

## References
- https://scikit-learn.org/stable/

## Assumptions and Limitations
Hyperparameters are set for diverse model complexities.
