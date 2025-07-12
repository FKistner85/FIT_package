"""Preconfigured feature selection transformers."""

from .feature_selection_wrapper import FeatureSelectionTransformer

SELECTORS = {
    "forward_10": FeatureSelectionTransformer(method="forward", k=10),
    "random_forest_20": FeatureSelectionTransformer(method="random_forest", k=20),
    "variance": FeatureSelectionTransformer(method="variance"),
    "lasso_10": FeatureSelectionTransformer(method="lasso", k=10),
}
