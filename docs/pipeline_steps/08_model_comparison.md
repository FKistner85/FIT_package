# Step 8: Model Comparison

Two scripts illustrate model training.  `compare_models.py` fits a
simple logistic regression and a support vector machine (SVM).  The
more comprehensive `model_comparison_sex.py` performs a grid search
over several classifiers using `GridSearchCV`.

**Selectable Methods**
- **Logistic Regression** – a linear classifier optimising the
  cross-entropy loss with optional L1 or L2 regularisation.
- **Support Vector Machine (SVC)** – maximises the margin between
  classes; kernels provide non-linear decision boundaries.
- **Random Forest** – an ensemble of decision trees built on bootstrap
  samples with random feature selection at each split.
- **k‑Nearest Neighbours** – memory-based classification by majority vote
  among the closest training points.
- **Linear Discriminant Analysis** – assumes Gaussian class-conditional
  densities with identical covariance matrices.
- **Gaussian Naive Bayes** – treats each feature as independent and fits
  a Gaussian to each one; extremely fast.
- **AdaBoost** – sequentially fits weak learners and adjusts sample
  weights to focus on hard cases.
- *(Optional)* **CatBoost**, **LightGBM**, **XGBoost** – efficient
  gradient boosting tree implementations.

| Method | Idea | Typical Complexity |
| ------ | ---- | ------------------ |
| Logistic Regression | Linear model with gradient descent | O(n × p) per iter. |
| SVC | Margin maximisation with kernels | O(n²)–O(n³) |
| Random Forest | Bagging of decision trees | O(trees × n log n) |
| k-NN | Distance-based voting | O(n × p) per query |
| LDA | Gaussian generative model | O(n × p²) |
| Gaussian NB | Independent Gaussians | O(n × p) |
| AdaBoost | Weighted ensemble of weak learners | O(trees × n × p) |
| CatBoost/LightGBM/XGBoost | Gradient boosting | varies, ~O(trees × n log n) |

Grid search multiplies these base costs by the number of tested
hyperparameter combinations and cross-validation folds.

