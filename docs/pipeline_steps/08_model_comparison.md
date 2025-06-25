# Step 8: Model Comparison

Two scripts provide modelling examples. `compare_models.py` trains a
logistic regression and an SVM. `model_comparison_sex.py` runs an
extensive grid search over multiple classifiers.

**Selectable Methods**
- Logistic Regression
- Support Vector Machine (SVC)
- Random Forest
- k-Nearest Neighbours
- Linear Discriminant Analysis
- Gaussian Naive Bayes
- AdaBoost
- (optional) CatBoost, LightGBM, XGBoost

**Algorithmic Cost**
- Depends on the chosen estimator. Grid search multiplies the base model
  cost by the number of parameter combinations and folds.

