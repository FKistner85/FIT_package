"""Simple model comparison on scaled datasets."""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import FIT_python.config as config
from FIT_python.config import DEFAULT_TARGETS, OTTER_META_COLS
from FIT_python.path_utils import scaled_path
import sys


class ModelComparator:
    """Train small models on each scaled dataset and save accuracies."""

    def compare_all(self) -> int:
        if not config.SCALED_DIR.exists():
            msg = f"Required file not found: {config.SCALED_DIR}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            if config.DEBUG_MODE:
                raise FileNotFoundError(msg)
            return 1
        config.RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        datasets = {
            p.stem.rsplit("_", 1)[0]
            for p in config.SCALED_DIR.glob("*_train.parquet")
        }
        rows = []
        for dataset in datasets:
            train_p = scaled_path(dataset, "train")
            test_p = scaled_path(dataset, "test")
            if not train_p.exists() or not test_p.exists():
                missing = train_p if not train_p.exists() else test_p
                msg = f"Required file not found: {missing}"
                print(f"[ERROR] {msg}", file=sys.stderr)
                if config.DEBUG_MODE:
                    raise FileNotFoundError(msg)
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
                print(f"[REPORT] {dataset} {name} accuracy={acc:.3f}")

        out_path = config.RESULTS_DATA_DIR / "model_comparison.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[SUCCESS] results written to {out_path}")
        return 0
