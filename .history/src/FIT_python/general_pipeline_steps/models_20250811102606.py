from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

MODELS = {
    # ─── Logistic Regression Variants ───────────────────────────────────────────
    "logreg_l2": LogisticRegression(penalty="l2", C=1.0, max_iter=200),
    "logreg_l1": LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=200),

    # ─── Random Forest Variants ────────────────────────────────────────────────
    "rf_small": RandomForestClassifier(n_estimators=20, max_depth=None, n_jobs=-1),
    "rf_med": RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1),
    "rf_large": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1),

    # ─── Extra Trees Variants ─────────────────────────────────────────────────
   # "et_med": ExtraTreesClassifier(n_estimators=200, max_depth=None, n_jobs=-1),
   # "et_sm": ExtraTreesClassifier(n_estimators=100, max_depth=10, n_jobs=-1),

    # ─── K-Nearest Neighbors Variants ────────────────────────────────────────
   #"knn_3": KNeighborsClassifier(n_neighbors=3),
   # "knn_5": KNeighborsClassifier(n_neighbors=5),
    #"knn_7": KNeighborsClassifier(n_neighbors=7),

    # ─── SVM Variants ─────────────────────────────────────────────────────────
   # "svm_linear": SVC(kernel="linear", C=1.0, probability=True),
   # "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),

    # ─── LDA Variants ───────────────────────────────────────────────────────────
    "lda_svd": LinearDiscriminantAnalysis(solver="svd"),  # Default, keine Shrinkage
    "lda_lsqr": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None),
    "lda_lsqr_shrinkage_auto": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "lda_eigen": LinearDiscriminantAnalysis(solver="eigen", shrinkage=None),
    "lda_eigen_shrinkage_auto": LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
    "lda_lsqr_shrinkage_point1": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.1),
    "lda_lsqr_shrinkage_point5": LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.5),
    "lda_ncomp1": LinearDiscriminantAnalysis(n_components=1),
    "lda_ncomp2": LinearDiscriminantAnalysis(n_components=2),
    # ─── XGBoost Variants ─────────────────────────────────────────────────────
    "xgb_std": XGBClassifier(n_estimators=100, use_label_encoder=False, verbosity=0),
    "xgb_hist": XGBClassifier(
        n_estimators=100, tree_method="hist", use_label_encoder=False, verbosity=0
    ),
    "xgb_deep": XGBClassifier(     # tiefere Bäume, für komplexere Muster
        n_estimators=200, max_depth=12, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, verbosity=0
    ),
    "xgb_shallow": XGBClassifier(  # flachere Bäume, schneller, weniger Overfitting
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        use_label_encoder=False, verbosity=0
    ),
    "xgb_regularized": XGBClassifier(  # stärkere Regularisierung
        n_estimators=150, max_depth=6, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=2.0,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, verbosity=0
    ),

    # ─── LightGBM Variants ────────────────────────────────────────────────────
   # "lgbm_std": LGBMClassifier(n_estimators=100, verbose=-1),
   # "lgbm_md10": LGBMClassifier(n_estimators=100, max_depth=10, verbose=-1),

    # ─── CatBoost Variants (auskommentiert) ───────────────────────────────────
    # "catb_std": CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0),
    # "catb_fast": CatBoostClassifier(iterations=50, learning_rate=0.2, verbose=0),
}



# Grouped presets to keep hyperparameter search manageable
CLASSIFIER_PRESETS = {
    "random_forest_fast": MODELS["rf_med"],
    "random_forest_accurate": MODELS["rf_large"],
    "xgboost_fast": MODELS["xgb_hist"],
    "xgboost_accurate": MODELS["xgb_std"],
    "logreg_l1": MODELS["logreg_l1"],
    "logreg_l2": MODELS["logreg_l2"],
}
