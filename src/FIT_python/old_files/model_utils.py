#!/usr/bin/env python3
"""General model utilities to train and evaluate classifiers on sex/species."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def train_and_evaluate(
    ds_folder: Path,
    models: Dict[str, Any],
    metrics: Dict[str, Any],
    tuner: Any,
    seed: int,
    debug: bool
) -> Dict[str, Any]:
    """Load numeric arrays, train each model, compute reports for one dataset."""
    # Load numpy arrays
    X_train = np.load(ds_folder / "X_train.npy", mmap_mode="r")
    y_train = np.load(ds_folder / "y_train.npy", mmap_mode="r")
    X_test  = np.load(ds_folder / "X_test.npy",  mmap_mode="r")
    y_test  = np.load(ds_folder / "y_test.npy",  mmap_mode="r")

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    ds_results: Dict[str, Any] = {}
    for name, estimator in models.items():
        if debug:
            print(f"[INFO] Training {name}")
        model = estimator.__class__(**estimator.get_params())
        # Hyperparameter tuning if provided
        if tuner is not None:
            best = tuner.tune(
                name,
                pd.DataFrame(X_train),
                pd.Series(y_train_enc),
            )
            model.set_params(**best)
            if debug:
                print(f"[INFO] Best params for {name}: {best}")
        # Fit & predict
        model.fit(X_train, y_train_enc)
        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        # Metrics
        reports = {
            m: ( metrics[m](y_test_enc, y_pred_enc) if callable(metrics[m]) else None )
            for m in metrics
        }
        cs_report = classification_report(y_test, y_pred, zero_division=0)
        ds_results[name] = {
            "model": model,
            "report": cs_report,
            "metrics": reports,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
    return ds_results
