"""Preconfigured outlier cleaning transformers."""

from .outlier_wrapper import OutlierCleanerTransformer

OUTLIERS = {
    "clip_90": OutlierCleanerTransformer(method="clip", lower_quantile=0.05, upper_quantile=0.95),
    "clip_98": OutlierCleanerTransformer(method="clip", lower_quantile=0.01, upper_quantile=0.99),
    "zscore_3": OutlierCleanerTransformer(method="zscore", z_thresh=3.0),
}
