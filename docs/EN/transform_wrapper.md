# transform_wrapper.py

## Overview
NumericTransformer converts DataFrame features to a numpy matrix.

## Key Components
- NumericTransformer

### NumericTransformer
Fits on a `DataFrame` to determine feature columns then converts those columns
to floating point numbers with comma replacement. The transformer returns a
plain numpy array suitable for scikit‑learn estimators. It assumes that metadata
and target columns have been defined in the configuration. Categorical features
should be encoded separately before using this transformer.

## References
- https://pandas.pydata.org/docs/

## Assumptions and Limitations
Ignores metadata and target columns defined in configuration.
