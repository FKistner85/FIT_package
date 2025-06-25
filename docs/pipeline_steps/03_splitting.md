# Step 3: Train/Test Splitting

Dataset splits are created with `all_splits` from
`FIT_python.splits_wrapper`, which itself calls `train_test_group_split`
from `FIT_python.split_utils` to ensure individuals never appear in both
splits.

**Key Functions**
- `all_splits`
- `train_test_group_split`
- `split_path`

**Algorithmic Cost**
- Approximately linear in dataset size.

