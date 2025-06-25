# Step 4: Scaling

`scale_splits.py` standardises features using scikit-learn's
`StandardScaler`. The wrapper `ScalerWrapper` additionally supports
`RobustScaler`.

**Selectable Methods**
- `StandardScaler` – standardisation (zero mean, unit variance).
- `RobustScaler` – robust to outliers by using percentiles.

**Algorithmic Cost**
- O(n * p) with n samples and p features.

