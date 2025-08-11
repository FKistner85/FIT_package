# === MODELS: nur 4 Basis-Estimatoren ===
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

MODELS = {
    "lda_svd": LinearDiscriminantAnalysis(solver="svd"),
    "lda_lsqr": LinearDiscriminantAnalysis(solver="lsqr"),
    "lda_eigen": LinearDiscriminantAnalysis(solver="eigen"),
    "xgb_std": XGBClassifier(
        n_estimators=200, use_label_encoder=False, verbosity=0, tree_method="exact"
    ),
    "xgb_hist": XGBClassifier(
        n_estimators=200, use_label_encoder=False, verbosity=0, tree_method="hist"
    ),
}

