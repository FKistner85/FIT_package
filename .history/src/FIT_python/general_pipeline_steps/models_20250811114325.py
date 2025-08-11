# === MODELS: nur 4 Basis-Estimatoren ===
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

MODELS = {
    "logreg": LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=500
    ),
    "rf": RandomForestClassifier(
        n_estimators=100, max_depth=None, n_jobs=-1
    ),
    "xgb": XGBClassifier(
        n_estimators=200, use_label_encoder=False, verbosity=0, tree_method="hist"
    ),
    "lda": LinearDiscriminantAnalysis(solver="svd"),  # robust & ohne Shrinkage-Konflikte
}
