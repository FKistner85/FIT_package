# Step 5: Feature Selection

`create_feature_selection.py` ranks features by variance but the
`FeatureSelector` wrapper in `FIT_python.feature_utils` exposes several
alternative strategies.

**Selectable Methods**
- `forward_count` / `forward_p` – stepwise forward selection based on
  ANCOVA statistics.  Features are added one by one if their p-value is
  below a threshold or they improve the model count the most.  This
  procedure requires re-evaluating the model many times and thus scales
  roughly quadratically with the number of candidate features.
- `random_forest` – trains a random forest and ranks features by the
  mean decrease in impurity.  Importance estimation is efficient once the
  forest is fit but the cost grows with the number of trees and depth.
- `anova_kbest` – performs an ANOVA F-test for each feature independently
  and keeps the top scoring ones.  Suitable for numerical targets and
  computationally light.
- `mutual_info` – computes mutual information between each feature and the
  target variable.  Captures non-linear dependencies but still processes
  features individually.
- `chi2_kbest` – uses the chi-square statistic for categorical data.
  Requires non-negative features and a discrete target.
- `l1_logistic` – fits a logistic regression with an L1 penalty and keeps
  only features with non-zero coefficients.  The underlying solver is
  typically liblinear.

| Method | Principle | Complexity |
| ------ | --------- | ---------- |
| forward_count / forward_p | Stepwise selection via repeated model fitting | ~O(p²) |
| random_forest | Mean decrease impurity from ensemble | O(trees × n × log p) |
| anova_kbest | Univariate F-test | O(n × p) |
| mutual_info | Mutual information estimate | O(n × p) |
| chi2_kbest | Chi-square statistic | O(n × p) |
| l1_logistic | Sparse logistic regression | O(n × p) per iteration |

