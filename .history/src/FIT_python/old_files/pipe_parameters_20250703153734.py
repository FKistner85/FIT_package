# src/FIT_python/pipeline/pipeline_wrapper.py

"""Legacy pipeline using balanced accuracy for evaluation."""

from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import PredefinedSplit, cross_val_score

from FIT_python.config import SPLITS_DIR, RESULTS_DATA_DIR
from FIT_python.step_02_a_splitting_train_test.wrapper import SplitWrapper
from .pipe_parameters import get_pipeline_steps
from ..pipeline_sex.models import MODELS

class PipelineWrapper:
    """
    Führt für jede Spezies in SPLITS_DIR eine sklearn-Pipeline aus.
    Die Fold-Spalte wird **nur** für PredefinedSplit genutzt,
    danach aus den Trainingsdaten entfernt.
    """
    def __init__(
        self,
        model_keys: list[str] | None = None,
        fs_method: str = 'forward',
        fs_k: int = 20,
        impute_method: str | None = 'miss_forest',
        outlier_method: str | None = 'clip',
        scaler_method: str | None = 'standard',
        reduce_pre_method: str | None = None,
        reduce_post_method: str | None = None
    ):
        RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.model_keys = model_keys or list(MODELS.keys())
        self.fs_method = fs_method
        self.fs_k = fs_k
        self.impute_method = impute_method
        self.outlier_method = outlier_method
        self.scaler_method = scaler_method
        self.reduce_pre_method = reduce_pre_method
        self.reduce_post_method = reduce_post_method

    def run(self) -> pd.DataFrame:
        # 1) Splits erzeugen
        SplitWrapper().split_all()

        records: list[dict] = []
        for species_dir in sorted(Path(SPLITS_DIR).iterdir()):
            if not species_dir.is_dir():
                continue

            train_p = species_dir / 'train.parquet'
            test_p  = species_dir / 'test.parquet'
            if not train_p.exists() or not test_p.exists():
                continue

            # 2) Laden & Filter
            df_train = pd.read_parquet(train_p).dropna(subset=['sex'])
            df_test  = pd.read_parquet(test_p).dropna(subset=['sex'])
            df_train = df_train[df_train['sex'].isin(['f','m'])]
            df_test  = df_test[df_test['sex'].isin(['f','m'])]

            # 3) Ziel‐Encoding
            y_train = df_train['sex'].map({'f':0,'m':1})
            y_test  = df_test['sex'].map({'f':0,'m':1})

            # 4) PredefinedSplit per 'Fold'
            ps = PredefinedSplit(test_fold=df_train['Fold'].values)

            # **Wichtig**: Entferne 'Fold' aus X_train, bevor es in die Pipeline geht
            X_train = df_train.drop(columns=['Fold'])
            X_test  = df_test  # Test hat keine Fold-Spalte

            # 5) Pro Modell
            for key in self.model_keys:
                model = MODELS[key]

                # Baue die Schritte und hänge den Classifier an
                steps = get_pipeline_steps(
                    fs_method=self.fs_method,
                    fs_k=self.fs_k,
                    impute_method=self.impute_method,
                    outlier_method=self.outlier_method,
                    scaler_method=self.scaler_method,
                    reduce_pre_method=self.reduce_pre_method,
                    reduce_post_method=self.reduce_post_method
                )
                steps.append(('classifier', model))
                pipe = Pipeline(steps)

                # 6a) CV auf X_train, y_train
                try:
                    cv_scores = cross_val_score(
                        pipe,
                        X_train,
                        y_train,
                        cv=ps,
                        scoring='balanced_accuracy'
                    )
                    cv_mean = cv_scores.mean()
                except Exception:
                    cv_mean = None

                # 6b) Finales Fit & Test
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                test_acc = balanced_accuracy_score(y_test, y_pred)
                report   = classification_report(y_test, y_pred, output_dict=True)

                records.append({
                    'species': species_dir.name,
                    'model': key,
                    'fs_method': self.fs_method,
                    'fs_k': self.fs_k,
                    'impute_method': self.impute_method,
                    'outlier_method': self.outlier_method,
                    'scaler_method': self.scaler_method,
                    'reduce_pre_method': self.reduce_pre_method,
                    'reduce_post_method': self.reduce_post_method,
                    'cv_balanced_accuracy': cv_mean,
                    'test_balanced_accuracy': test_acc,
                    'classification_report': report
                })

        # 7) Ergebnisse speichern
        df_results = pd.DataFrame(records)
        out = Path(RESULTS_DATA_DIR) / 'model_results.csv'
        df_results.to_csv(out, index=False)
        return df_results
