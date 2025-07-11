"""Shared transformer utilities for preprocessing pipelines."""

from .feature_selection_wrapper import FeatureSelectionTransformer, FEATURE_SELECTION_PRESETS
from .feature_scaler_wrapper import FeatureScalerTransformer, SCALER_PRESETS
from .outlier_wrapper import OutlierCleanerTransformer, OUTLIER_PRESETS, plot_feature_distributions
from .dimensionality_reduction_wrapper import DimensionalityReducerTransformer, REDUCER_PRESETS

__all__ = [
    "FeatureSelectionTransformer",
    "FEATURE_SELECTION_PRESETS",
    "FeatureScalerTransformer",
    "SCALER_PRESETS",
    "OutlierCleanerTransformer",
    "OUTLIER_PRESETS",
    "plot_feature_distributions",
    "DimensionalityReducerTransformer",
    "REDUCER_PRESETS",
]
