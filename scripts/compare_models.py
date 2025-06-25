#!/usr/bin/env python3
"""Very small model comparison using scaled splits."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from FIT_python.config import (
    SCALED_DIR,
    RESULTS_DATA_DIR,
    DEFAULT_TARGETS,
    OTTER_META_COLS,
    DEBUG_MODE,
)
from FIT_python.path_utils import scaled_path


def main() -> None:
    if not SCALED_DIR.exists():
        msg = f"Required file not found: {SCALED_DIR}"
        if DEBUG_MODE:
            raise FileNotFoundError(msg)
        else:
            print("Skipping:", msg)
            return
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {p.stem.rsplit("_", 1)[0] for p in SCALED_DIR.glob("*_train.parquet")}
    rows = []
    for dataset in datasets:
        train_p = scaled_path(dataset, "train")
        test_p = scaled_path(dataset, "test")
        if not train_p.exists() or not test_p.exists():
            msg = f"Required file not found: {train_p if not train_p.exists() else test_p}"
            if DEBUG_MODE:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                continue
        df_train = pd.read_parquet(train_p)
        df_test = pd.read_parquet(test_p)
        meta_cols = OTTER_META_COLS if "otter" in dataset.lower() else ["id"]
        feature_cols = [
            c for c in df_train.columns if c not in meta_cols + DEFAULT_TARGETS
        ]
        X_train, y_train = df_train[feature_cols], df_train["sex"]
        X_test, y_test = df_test[feature_cols], df_test["sex"]
        models = {
            "LogReg": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            rows.append({"Dataset": dataset, "Model": name, "Accuracy": acc})
    out_path = RESULTS_DATA_DIR / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
