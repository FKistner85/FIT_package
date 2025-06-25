#!/usr/bin/env python3
"""Full processing pipeline for the 'sex' target."""

from FIT_python.scaler_wrapper import ScalerWrapper
from FIT_python.feature_wrapper import FeatureSelector
from FIT_python.dim_reduction_wrapper import DimReducer
from FIT_python.model_comparator import ModelComparator
import FIT_python.config as config


def main() -> int:
    # Step 1: scaling
    for scaler in config.SCALER_PARAMS.keys():
        sw = ScalerWrapper(
            scaler_type=scaler,
            out_dir=config.SCALED_DIR / scaler,
            numeric_dir=config.NUMERIC_DIR / scaler,
        )
        sw.scale_all()

    # Step 2: feature selection on default scaled data
    fs = FeatureSelector(
        scaled_dir=config.SCALED_DIR / config.DEFAULT_SCALER,
        out_dir=config.FEATURE_SELECTED_DIR,
    )
    fs.select_all()

    # Step 3: dimensionality reduction
    dr = DimReducer()
    dr.reduce_all()

    # Step 4: modelling
    comp = ModelComparator()
    comp.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
