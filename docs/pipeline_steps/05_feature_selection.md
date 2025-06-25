# Step 5: Feature Selection

`create_feature_selection.py` uses a simple variance ranking but the
`FeatureSelector` wrapper exposes several methods implemented in
`FIT_python.feature_utils`.

**Selectable Methods**
- `forward_count` / `forward_p` – ANCOVA forward selection.
- `random_forest` – feature importance from a random forest.
- `anova_kbest` – ANOVA F-test `SelectKBest`.
- `mutual_info` – mutual information `SelectKBest`.
- `chi2_kbest` – chi-square `SelectKBest`.
- `l1_logistic` – L1-penalised logistic regression.

**Algorithmic Cost**
- Varies by method; forward selection is roughly O(p^2), others are O(n * p).

