"""Legacy import wrapper for split utilities used in tests."""

from FIT_python.old_files.split_utils import (
    train_test_group_split,
    group_stratified_kfold,
    create_train_test_split_otter,
)

__all__ = [
    "train_test_group_split",
    "group_stratified_kfold",
    "create_train_test_split_otter",
]
