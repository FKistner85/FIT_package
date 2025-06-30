# src/FIT_python/pipeline/models.py

"""
Model definitions: mapping identifiers to sklearn estimator instances.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

MODELS = {
    "log_reg":   LogisticRegression(max_iter=200),
    "rf":        RandomForestClassifier(n_estimators=100),
    "et":        ExtraTreesClassifier(n_estimators=100),
    "knn":       KNeighborsClassifier(n_neighbors=3),
    "svm":       SVC(kernel='linear', probability=True),
    "lda":       LinearDiscriminantAnalysis(),
    "xgb":       XGBClassifier(n_estimators=100, verbosity=0, use_label_encoder=False),
    "lgbm":      LGBMClassifier(n_estimators=100, verbose=-1),
    "catboost":  CatBoostClassifier(iterations=100, verbose=0),
}
