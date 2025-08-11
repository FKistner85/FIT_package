# === MODELS: nur robuste Basis-Estimatoren (LDA: nur svd) ===
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

MODELS = {
    # Linear Discriminant Analysis (robuste SVD-Variante)
    "lda_svd": LinearDiscriminantAnalysis(solver="svd"),
    
    # XGBoost mit Standard-Tree-Methode
    "xgb_std": XGBClassifier(
        n_estimators=200,
        use_label_encoder=False,
        verbosity=0,
        tree_method="exact"
    ),
    
    # XGBoost mit Histogram-basiertem Tree-Algorithmus
    "xgb_hist": XGBClassifier(
        n_estimators=200,
        use_label_encoder=False,
        verbosity=0,
        tree_method="hist"
    ),
}
