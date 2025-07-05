import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def test_grid_search_balanced_accuracy():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(60, 4), columns=[f"f{i}" for i in range(4)])
    X["individual_id"] = np.repeat(np.arange(20), 3)
    y = np.random.randint(0, 2, size=60)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    search = GridSearchCV(
        pipe,
        param_grid={"clf__C": [0.1, 1.0]},
        scoring="balanced_accuracy",
        cv=3,
    )
    search.fit(X.drop(columns=["individual_id"]), y)
    assert 0 <= search.best_score_ <= 1
    print("Grid search completed with best_score=", search.best_score_)
