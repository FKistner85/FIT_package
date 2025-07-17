"""Preconfigured feature selection transformers."""

from .feature_selection_wrapper import FeatureSelectionTransformer

SELECTORS = {
    "forward_10": FeatureSelectionTransformer(method="forward", k=10),
    "random_forest_20": FeatureSelectionTransformer(method="random_forest", k=20),
    "random_forest_10": FeatureSelectionTransformer(method="random_forest", k=10),
    "random_forest_8": FeatureSelectionTransformer(method="random_forest", k=8),
    "random_forest_50": FeatureSelectionTransformer(method="random_forest", k=50),
    "random_forest_30": FeatureSelectionTransformer(method="random_forest", k=30),
    "random_forest_100": FeatureSelectionTransformer(method="random_forest", k=100),
    "random_forest_200": FeatureSelectionTransformer(method="random_forest", k=200),
    "random_forest_150": FeatureSelectionTransformer(method="random_forest", k=150),
    "variance_10": FeatureSelectionTransformer(method="variance", k=10),
    "variance": FeatureSelectionTransformer(method="variance"),
    "lasso_10": FeatureSelectionTransformer(method="lasso", k=10),
    "forward_2": FeatureSelectionTransformer(method="forward", k=2),
    "forward_1": FeatureSelectionTransformer(method="forward", k=1),
    "forward_3": FeatureSelectionTransformer(method="forward", k=3),
    "forward_4": FeatureSelectionTransformer(method="forward", k=4),
    "forward_5": FeatureSelectionTransformer(method="forward", k=5),
    "forward_6": FeatureSelectionTransformer(method="forward", k=6),
    "forward_7": FeatureSelectionTransformer(method="forward", k=7),
    "random_forest_1": FeatureSelectionTransformer(method="random_forest", k=1),
    "random_forest_2": FeatureSelectionTransformer(method="random_forest", k=2),
    "random_forest_3": FeatureSelectionTransformer(method="random_forest", k=3),
    "random_forest_4": FeatureSelectionTransformer(method="random_forest", k=4),
    "random_forest_5": FeatureSelectionTransformer(method="random_forest", k=5),
    "random_forest_6": FeatureSelectionTransformer(method="random_forest", k=6),
    "random_forest_7": FeatureSelectionTransformer(method="random_forest", k=7),
    "lasso_1": FeatureSelectionTransformer(method="lasso", k=1),
    "lasso_2": FeatureSelectionTransformer(method="lasso", k=2),
    "lasso_3": FeatureSelectionTransformer(method="lasso", k=3),
    "lasso_4": FeatureSelectionTransformer(method="lasso", k=4),
    "lasso_5": FeatureSelectionTransformer(method="lasso", k=5),
    "lasso_6": FeatureSelectionTransformer(method="lasso", k=6),            

}
