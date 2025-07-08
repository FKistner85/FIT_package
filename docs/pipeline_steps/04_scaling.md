# Step 4: Scaling
## Overview

`scale_splits.py` standardises features of each split.  In the
library this is handled by the `ScalerWrapper` which can dispatch to
different scalers from scikit-learn.

## Selectable Methods
- `StandardScaler` – subtracts the mean and divides by the
  standard deviation of each feature.  This assumes a Gaussian-like
  distribution and can be sensitive to extreme values.
- `RobustScaler` – scales features according to the median and the
  inter-quartile range (IQR).  Because it relies on percentiles it is
  more resilient to outliers than simple standardisation.

The computational cost for both scalers is linear in the number of
samples *n* and features *p* because only simple statistics are
computed.

| Method | Summary | Complexity |
| ------ | ------- | ---------- |
| StandardScaler | Centre to zero mean and unit variance | O(n × p) |
| RobustScaler | Scale by median and IQR | O(n × p) |



